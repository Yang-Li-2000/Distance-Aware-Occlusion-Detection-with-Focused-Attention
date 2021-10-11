# ------------------------------------------------------------------------
# Licensed under the Apache License, Version 2.0 (the "License")
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
DETR model and criterion classes.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .hoi_matcher import build_matcher as build_hoi_matcher
from .transformer import build_transformer

from magic_numbers import *

# This will be modified by build() if train on 2.5vrd
num_humans = 2


class HoiTR(nn.Module):
    """ This is the DETR module that performs object detection """

    def __init__(self, backbone, transformer, num_classes, num_actions,
                 num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture.
                See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is
                the maximal number of objects DETR can detect in a single image.
                For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder
                layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim,
                                    kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

        self.human_cls_embed = nn.Linear(hidden_dim, num_humans + 1)
        self.human_box_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.object_cls_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.object_box_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.action_cls_embed = nn.Linear(hidden_dim, num_actions + 1)
        self.occlusion_cls_embed = nn.Linear(hidden_dim, num_actions + 1)

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape
                    [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W],
                    containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object)
                                for all queries. Shape =
                                [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries,
                                represented as
                                (center_x, center_y, height, width).
                                These values are normalized in [0, 1],  relative
                                to the size of each individual image
                                (disregarding possible padding).
                                See PostProcess for information on how to
                                retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are
                                activated. It is a list of
                                dictionnaries containing the two above keys
                                for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = \
        self.transformer(self.input_proj(src), mask, self.query_embed.weight,
                         pos[-1])[0]

        human_outputs_class = self.human_cls_embed(hs)
        human_outputs_coord = self.human_box_embed(hs).sigmoid()
        object_outputs_class = self.object_cls_embed(hs)
        object_outputs_coord = self.object_box_embed(hs).sigmoid()
        action_outputs_class = self.action_cls_embed(hs)
        occlusion_outputs_class = self.occlusion_cls_embed(hs)

        out = {
            'human_pred_logits': human_outputs_class[-1],
            'human_pred_boxes': human_outputs_coord[-1],
            'object_pred_logits': object_outputs_class[-1],
            'object_pred_boxes': object_outputs_coord[-1],
            'action_pred_logits': action_outputs_class[-1],
            'occlusion_pred_logits': occlusion_outputs_class[-1]
        }

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                human_outputs_class,
                human_outputs_coord,
                object_outputs_class,
                object_outputs_coord,
                action_outputs_class,
                occlusion_outputs_class
            )
        return out

    @torch.jit.unused
    def _set_aux_loss(self,
                      human_outputs_class,
                      human_outputs_coord,
                      object_outputs_class,
                      object_outputs_coord,
                      action_outputs_class,
                      occlusion_outputs_class
                      ):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{
            'human_pred_logits': a,
            'human_pred_boxes': b,
            'object_pred_logits': c,
            'object_pred_boxes': d,
            'action_pred_logits': e,
            'occlusion_pred_logits': f,
        } for
            a,
            b,
            c,
            d,
            e,
            f,
            in zip(
                human_outputs_class[:-1],
                human_outputs_coord[:-1],
                object_outputs_class[:-1],
                object_outputs_coord[:-1],
                action_outputs_class[:-1],
                occlusion_outputs_class[:-1]
            )]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the
            outputs of the model
        2) we supervise each pair of matched ground-truth / prediction
            (supervise class and box)
    """

    def __init__(self, num_classes, num_actions, matcher, weight_dict, eos_coef,
                 losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special
                no-object category
            matcher: module able to compute a matching between targets
                and proposals
            weight_dict: dict containing as key the names of the losses
                and as values their relative weight.
            eos_coef: relative classification weight applied to the
                no-object category
            losses: list of all the losses to be applied. See get_loss for
                list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes  # 91
        self.num_actions = num_actions
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses

        human_empty_weight = torch.ones(num_humans + 1)
        human_empty_weight[-1] = self.eos_coef
        self.register_buffer('human_empty_weight', human_empty_weight)

        object_empty_weight = torch.ones(num_classes + 1)
        object_empty_weight[-1] = self.eos_coef
        self.register_buffer('object_empty_weight', object_empty_weight)

        action_empty_weight = torch.ones(num_actions + 1)
        action_empty_weight[-1] = self.eos_coef
        self.register_buffer('action_empty_weight', action_empty_weight)

        occlusion_empty_weight = torch.ones(num_actions + 1)
        occlusion_empty_weight[-1] = self.eos_coef
        self.register_buffer('occlusion_empty_weight', occlusion_empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor
            of dim [nb_target_boxes]
        """
        assert 'human_pred_logits' in outputs
        assert 'object_pred_logits' in outputs
        assert 'action_pred_logits' in outputs
        assert 'occlusion_pred_logits' in outputs

        human_src_logits = outputs['human_pred_logits']
        object_src_logits = outputs['object_pred_logits']
        action_src_logits = outputs['action_pred_logits']
        occlusion_src_logits = outputs['occlusion_pred_logits']

        idx = self._get_src_permutation_idx(indices)

        human_target_classes_o = torch.cat(
            [t["human_labels"][J] for t, (_, J) in zip(targets, indices)])
        object_target_classes_o = torch.cat(
            [t["object_labels"][J] for t, (_, J) in zip(targets, indices)])
        action_target_classes_o = torch.cat(
            [t["action_labels"][J] for t, (_, J) in zip(targets, indices)])
        occlusion_target_classes_o = torch.cat(
            [t["occlusion_labels"][J] for t, (_, J) in zip(targets, indices)])

        human_target_classes = torch.full(human_src_logits.shape[:2],
                                          num_humans,
                                          dtype=torch.int64,
                                          device=human_src_logits.device)
        human_target_classes[idx] = human_target_classes_o

        object_target_classes = torch.full(object_src_logits.shape[:2],
                                           self.num_classes,
                                           dtype=torch.int64,
                                           device=object_src_logits.device)
        object_target_classes[idx] = object_target_classes_o

        action_target_classes = torch.full(action_src_logits.shape[:2],
                                           self.num_actions,
                                           dtype=torch.int64,
                                           device=action_src_logits.device)
        action_target_classes[idx] = action_target_classes_o

        occlusion_target_classes = torch.full(occlusion_src_logits.shape[:2],
                                              self.num_actions,
                                              dtype=torch.int64,
                                              device=occlusion_src_logits.device)
        occlusion_target_classes[idx] = occlusion_target_classes_o

        human_loss_ce = F.cross_entropy(human_src_logits.transpose(1, 2),
                                        human_target_classes,
                                        self.human_empty_weight)
        object_loss_ce = F.cross_entropy(object_src_logits.transpose(1, 2),
                                         object_target_classes,
                                         self.object_empty_weight)
        action_loss_ce = F.cross_entropy(action_src_logits.transpose(1, 2),
                                         action_target_classes,
                                         self.action_empty_weight)
        occlusion_loss_ce = F.cross_entropy(
            occlusion_src_logits.transpose(1, 2),
            occlusion_target_classes, self.occlusion_empty_weight)

        loss_ce = human_loss_ce + object_loss_ce + 2 * action_loss_ce + 2 * occlusion_loss_ce
        losses = {
            'loss_ce': loss_ce,
            'human_loss_ce': human_loss_ce,
            'object_loss_ce': object_loss_ce,
            'action_loss_ce': action_loss_ce,
            'occlusion_loss_ce': occlusion_loss_ce
        }

        if log:
            losses['class_error_action'] = 100 - \
                                           accuracy(action_src_logits[idx],
                                                    action_target_classes_o)[0]
            losses['class_error_occlusion'] = 100 - accuracy(
                occlusion_src_logits[idx], occlusion_target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the
            number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only.
            It doesn't propagate gradients
        """
        pred_logits_action = outputs['action_pred_logits']
        device_action = pred_logits_action.device
        tgt_lengths_action = torch.as_tensor(
            [len(v["action_labels"]) for v in targets], device=device_action)
        # Count the number of predictions that are NOT "no-object"
        # (which is the last class)
        card_pred_action = (
                    pred_logits_action.argmax(-1) != pred_logits_action.shape[
                -1] - 1).sum(1)
        card_err_action = F.l1_loss(card_pred_action.float(),
                                    tgt_lengths_action.float())

        pred_logits_occlusion = outputs['occlusion_pred_logits']
        device_occlusion = pred_logits_occlusion.device
        tgt_lengths_occlusion = torch.as_tensor(
            [len(v["occlusion_labels"]) for v in targets],
            device=device_occlusion)
        # Count the number of predictions that are NOT "no-object"
        # (which is the last class)
        card_pred_occlusion = (pred_logits_occlusion.argmax(-1) !=
                               pred_logits_occlusion.shape[-1] - 1).sum(1)
        card_err_occlusion = F.l1_loss(card_pred_occlusion.float(),
                                       tgt_lengths_occlusion.float())

        losses = {'cardinality_error_action': card_err_action,
                  'cardinality_error_occlusion': card_err_occlusion}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression
            loss and the GIoU loss targets dicts must contain the key "boxes"
            containing a tensor of dim [nb_target_boxes, 4] The target boxes
            are expected in format (center_x, center_y, w, h),
            normalized by the image size.
        """
        assert 'human_pred_boxes' in outputs
        assert 'object_pred_boxes' in outputs

        idx = self._get_src_permutation_idx(indices)

        human_src_boxes = outputs['human_pred_boxes'][idx]
        human_target_boxes = torch.cat(
            [t['human_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        object_src_boxes = outputs['object_pred_boxes'][idx]
        object_target_boxes = torch.cat(
            [t['object_boxes'][i] for t, (_, i) in zip(targets, indices)],
            dim=0)

        human_loss_bbox = F.l1_loss(human_src_boxes, human_target_boxes,
                                    reduction='none')
        object_loss_bbox = F.l1_loss(object_src_boxes, object_target_boxes,
                                     reduction='none')

        losses = dict()
        losses['human_loss_bbox'] = human_loss_bbox.sum() / num_boxes
        losses['object_loss_bbox'] = object_loss_bbox.sum() / num_boxes
        losses['loss_bbox'] = losses['human_loss_bbox'] + losses[
            'object_loss_bbox']

        human_loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(human_src_boxes),
            box_ops.box_cxcywh_to_xyxy(human_target_boxes)))
        object_loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(object_src_boxes),
            box_ops.box_cxcywh_to_xyxy(object_target_boxes)))
        losses['human_loss_giou'] = human_loss_giou.sum() / num_boxes
        losses['object_loss_giou'] = object_loss_giou.sum() / num_boxes

        losses['loss_giou'] = losses['human_loss_giou'] + losses[
            'object_loss_giou']
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the
                    model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses
                      applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if
                               k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the
        # last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes,
        # for normalization purposes
        num_boxes = sum(len(t["human_labels"]) for t in targets)

        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float,
                                    device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(
                self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output
        # of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices,
                                           num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    assert args.dataset_file in ['hico', 'vcoco', 'hoia',
                                 'two_point_five_vrd'], args.dataset_file
    if args.dataset_file in ['hico']:
        num_classes = 91
        num_actions = 118
    elif args.dataset_file in ['vcoco']:
        num_classes = 91
        num_actions = 30
    elif args.dataset_file in ['two_point_five_vrd']:
        num_classes = 602
        num_actions = 4
        global num_humans
        num_humans = num_classes
    else:
        num_classes = 12
        num_actions = 11

    device = torch.device(args.device)

    if args.backbone == 'swin':
        from .backbone_swin import build_backbone_swin
        backbone = build_backbone_swin(args)
    else:
        backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = HoiTR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_actions=num_actions,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )

    matcher = build_hoi_matcher(args)

    weight_dict = dict(loss_ce=1, loss_bbox=args.bbox_loss_coef,
                       loss_giou=args.giou_loss_coef)

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update(
                {k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']

    criterion = SetCriterion(num_classes=num_classes, num_actions=num_actions,
                             matcher=matcher,
                             weight_dict=weight_dict, eos_coef=args.eos_coef,
                             losses=losses)
    criterion.to(device)

    return model, criterion


# TODO: implement optimal transport.
#  1. (Done) cost matrix.
#  2. (Done) supplying vector.
#  3. (Done) demanding vector.
#  4. dynamic k estimation.
#  5. (optional) center prior.
#  6. Loss computation
class OptimalTransport(nn.Module):

    def __init__(self, args, alpha=1, num_queries=100, k=1, eps=0.1, max_iter=50):
        super().__init__()
        self.alpha = alpha
        self.num_queries = num_queries
        self.k = OT_k
        self.sinkhorn = SinkhornDistance(eps=eps, max_iter=max_iter)

        weight_dict = dict(loss_ce=1, loss_bbox=args.bbox_loss_coef,
                           loss_giou=args.giou_loss_coef)
        if args.aux_loss:
            aux_weight_dict = {}
            for i in range(args.dec_layers - 1):
                aux_weight_dict.update(
                    {k + f'_{i}': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        self.weight_dict = weight_dict

        device = torch.device(args.device)
        self.to(device)

    def loss_cls(self, outputs, targets, training=True, log=True):

        human_src_logits = outputs['human_pred_logits']
        object_src_logits = outputs['object_pred_logits']
        action_src_logits = outputs['action_pred_logits']
        occlusion_src_logits = outputs['occlusion_pred_logits']

        def store_to_list(name):
            result = list()
            for i in range(len(targets)):
                result.append(targets[i][name])
            return torch.vstack(result)

        human_target_classes = store_to_list("human_labels")
        object_target_classes = store_to_list('object_labels')
        action_target_classes = store_to_list("action_labels")
        occlusion_target_classes = store_to_list("occlusion_labels")

        # During training, the number of relations in each image is fixed
        num_queries = self.num_queries
        if training:
            # the first target
            human_target_classes_1 = human_target_classes[:, 0].unsqueeze(
                0).T.expand(-1, num_queries)
            object_target_classes_1 = object_target_classes[:, 0].unsqueeze(
                0).T.expand(-1, num_queries)
            action_target_classes_1 = action_target_classes[:, 0].unsqueeze(
                0).T.expand(-1, num_queries)
            occlusion_target_classes_1 = occlusion_target_classes[:,
                                         0].unsqueeze(0).T.expand(-1,
                                                                  num_queries)
            loss_human_classes_1 = F.cross_entropy(
                human_src_logits.permute(0, 2, 1), human_target_classes_1,
                reduction='none').unsqueeze(1)
            loss_object_classes_1 = F.cross_entropy(
                object_src_logits.permute(0, 2, 1), object_target_classes_1,
                reduction='none').unsqueeze(1)
            loss_action_classes_1 = F.cross_entropy(
                action_src_logits.permute(0, 2, 1), action_target_classes_1,
                reduction='none').unsqueeze(1)
            loss_occlusion_classes_1 = F.cross_entropy(
                occlusion_src_logits.permute(0, 2, 1),
                occlusion_target_classes_1, reduction='none').unsqueeze(1)

            # the second target
            human_target_classes_2 = human_target_classes[:, 1].unsqueeze(
                0).T.expand(-1, num_queries)
            object_target_classes_2 = object_target_classes[:, 1].unsqueeze(
                0).T.expand(-1, num_queries)
            action_target_classes_2 = action_target_classes[:, 1].unsqueeze(
                0).T.expand(-1, num_queries)
            occlusion_target_classes_2 = occlusion_target_classes[:,
                                         1].unsqueeze(0).T.expand(-1,
                                                                  num_queries)
            loss_human_classes_2 = F.cross_entropy(
                human_src_logits.permute(0, 2, 1), human_target_classes_2,
                reduction='none').unsqueeze(1)
            loss_object_classes_2 = F.cross_entropy(
                object_src_logits.permute(0, 2, 1), object_target_classes_2,
                reduction='none').unsqueeze(1)
            loss_action_classes_2 = F.cross_entropy(
                action_src_logits.permute(0, 2, 1), action_target_classes_2,
                reduction='none').unsqueeze(1)
            loss_occlusion_classes_2 = F.cross_entropy(
                occlusion_src_logits.permute(0, 2, 1),
                occlusion_target_classes_2, reduction='none').unsqueeze(1)

            # negative labels: background class
            bg_human_classes = torch.ones_like(human_target_classes_2) * (
                        human_src_logits.shape[-1] - 1)
            bg_object_classes = torch.ones_like(object_target_classes_2) * (
                        object_src_logits.shape[-1] - 1)
            bg_action_classes = torch.ones_like(action_target_classes_2) * (
                        action_src_logits.shape[-1] - 1)
            bg_occlusion_classes = torch.ones_like(
                occlusion_target_classes_2) * (occlusion_src_logits.shape[
                                                   -1] - 1)

            loss_human_classes_bg = F.cross_entropy(
                human_src_logits.permute(0, 2, 1), bg_human_classes,
                reduction='none').unsqueeze(1)
            loss_object_classes_bg = F.cross_entropy(
                object_src_logits.permute(0, 2, 1), bg_object_classes,
                reduction='none').unsqueeze(1)
            loss_action_classes_bg = F.cross_entropy(
                action_src_logits.permute(0, 2, 1), bg_action_classes,
                reduction='none').unsqueeze(1)
            loss_occlusion_classes_bg = F.cross_entropy(
                occlusion_src_logits.permute(0, 2, 1), bg_occlusion_classes,
                reduction='none').unsqueeze(1)

            # combine them
            human_loss_cls = torch.cat(
                [loss_human_classes_1, loss_human_classes_2,
                 loss_human_classes_bg], dim=1)
            object_loss_cls = torch.cat(
                [loss_object_classes_1, loss_object_classes_2,
                 loss_object_classes_bg], dim=1)
            action_loss_cls = torch.cat(
                [loss_action_classes_1, loss_action_classes_2,
                 loss_action_classes_bg], dim=1)
            occlusion_loss_cls = torch.cat(
                [loss_occlusion_classes_1, loss_occlusion_classes_2,
                 loss_occlusion_classes_bg], dim=1)

        else:
            raise NotImplementedError

        loss_cls = human_loss_cls + \
                   object_loss_cls + \
                   2 * action_loss_cls + \
                   2 * occlusion_loss_cls

        return loss_cls, human_target_classes, object_target_classes, action_target_classes, occlusion_target_classes

    def loss_reg(self, outputs, targets, training=True, log=True):

        num_queries = self.num_queries

        def store_to_list(name):
            result = list()
            for i in range(len(targets)):
                result.append(targets[i][name].reshape(-1))
            return torch.vstack(result)

        human_src_boxes = outputs['human_pred_boxes']
        human_target_boxes = store_to_list('human_boxes')

        object_src_boxes = outputs['object_pred_boxes']
        object_target_boxes = store_to_list('object_boxes')

        if training:
            # the first target
            human_target_boxes_1 = human_target_boxes[:, 0:4].unsqueeze(
                1).expand(-1, 100, -1)
            object_target_boxes_1 = object_target_boxes[:, 0:4].unsqueeze(
                1).expand(-1, 100, -1)
            loss_human_boxes_1 = F.l1_loss(human_src_boxes, human_target_boxes_1, reduction='none').sum(dim=2).unsqueeze(1)
            loss_object_boxes_1 = F.l1_loss(object_src_boxes, object_target_boxes_1, reduction='none').sum(dim=2).unsqueeze(1)
            human_loss_giou_1 = (1 - torch.diag(box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(human_src_boxes.reshape(-1, 4)),
                box_ops.box_cxcywh_to_xyxy(
                    human_target_boxes_1.reshape(-1, 4)),
                handle_degenerate_boxes=True)))
            human_loss_giou_1 = \
                human_loss_giou_1.reshape(-1, num_queries).unsqueeze(1)

            object_loss_giou_1 = (1 - torch.diag(box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(object_src_boxes.reshape(-1, 4)),
                box_ops.box_cxcywh_to_xyxy(object_target_boxes_1.reshape(-1, 4)),
                handle_degenerate_boxes=True)))
            object_loss_giou_1 \
                = object_loss_giou_1.reshape(-1, num_queries).unsqueeze(1)

            # the second target
            human_target_boxes_2 = human_target_boxes[:, 4:].unsqueeze(
                1).expand(-1, 100, -1)
            object_target_boxes_2 = object_target_boxes[:, 4:].unsqueeze(
                1).expand(-1, 100, -1)
            loss_human_boxes_2 = F.l1_loss(human_src_boxes, human_target_boxes_2, reduction='none').sum(dim=2).unsqueeze(1)
            loss_object_boxes_2 = F.l1_loss(object_src_boxes, object_target_boxes_2, reduction='none').sum(dim=2).unsqueeze(1)
            human_loss_giou_2 = (1 - torch.diag(box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(human_src_boxes.reshape(-1, 4)),
                box_ops.box_cxcywh_to_xyxy(
                    human_target_boxes_2.reshape(-1, 4))))).reshape(-1,
                                                                    num_queries).unsqueeze(
                1)
            object_loss_giou_2 = (1 - torch.diag(box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(object_src_boxes.reshape(-1, 4)),
                box_ops.box_cxcywh_to_xyxy(
                    object_target_boxes_2.reshape(-1, 4))))).reshape(-1,
                                                                     num_queries).unsqueeze(
                1)

            # combine them
            human_loss_boxes = torch.cat([loss_human_boxes_1, loss_human_boxes_2], dim=1)
            object_loss_boxes = torch.cat([loss_object_boxes_1, loss_object_boxes_2], dim=1)
            human_loss_giou = torch.cat([human_loss_giou_1, human_loss_giou_2],
                                        dim=1)
            object_loss_giou = torch.cat(
                [object_loss_giou_1, object_loss_giou_2], dim=1)

        else:
            raise NotImplementedError()

        loss_reg = (human_loss_giou + object_loss_giou + human_loss_boxes + object_loss_boxes) / 2

        return loss_reg, human_target_boxes, object_target_boxes

    def loss_computation(self, outputs,
                         gt_human_classes, gt_object_classes,
                         gt_action_classes, gt_occlusion_classes,
                         gt_human_boxes, gt_object_boxes, log=True):

        losses = dict()

        # loss_ce
        human_src_logits = outputs['human_pred_logits']
        object_src_logits = outputs['object_pred_logits']
        action_src_logits = outputs['action_pred_logits']
        occlusion_src_logits = outputs['occlusion_pred_logits']

        human_loss_ce = F.cross_entropy(human_src_logits.permute(0, 2, 1),
                                        gt_human_classes)
        object_loss_ce = F.cross_entropy(object_src_logits.permute(0, 2, 1),
                                         gt_object_classes)
        action_loss_ce = F.cross_entropy(action_src_logits.permute(0, 2, 1),
                                         gt_action_classes)
        occlusion_loss_ce = F.cross_entropy(
            occlusion_src_logits.permute(0, 2, 1),
            gt_occlusion_classes)
        loss_ce = human_loss_ce + object_loss_ce + \
                  2 * action_loss_ce + 2 * occlusion_loss_ce
        losses['human_loss_ce'] = human_loss_ce
        losses['object_loss_ce'] = object_loss_ce
        losses['action_loss_ce'] = action_loss_ce
        losses['occlusion_loss_ce'] = occlusion_loss_ce
        losses['loss_ce'] = loss_ce

        # loss_bbox
        human_src_boxes = outputs['human_pred_boxes']
        object_src_boxes = outputs['object_pred_boxes']

        human_loss_bbox = F.l1_loss(human_src_boxes, gt_human_boxes, reduction='mean') * 4
        object_loss_bbox = F.l1_loss(object_src_boxes, gt_object_boxes, reduction='mean') * 4
        loss_bbox = human_loss_bbox + object_loss_bbox
        losses['human_loss_bbox'] = human_loss_bbox
        losses['object_loss_bbox'] = object_loss_bbox
        losses['loss_bbox'] = loss_bbox

        # loss_giou
        human_loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(human_src_boxes.reshape(-1, 4)),
            box_ops.box_cxcywh_to_xyxy(gt_human_boxes.reshape(-1, 4)))).mean()
        object_loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(object_src_boxes.reshape(-1, 4)),
            box_ops.box_cxcywh_to_xyxy(gt_object_boxes.reshape(-1, 4)))).mean()
        loss_giou = human_loss_giou + object_loss_giou
        losses['human_loss_giou'] = human_loss_giou
        losses['object_loss_giou'] = object_loss_giou
        losses['loss_giou'] = loss_giou

        if log:
            # Not implemented
            pass

        return losses

    def test_cost_matrix(self, outputs, targets, cost_matrix):

        differences = torch.ones_like(cost_matrix) * 99999


        for image_index in range(0, cost_matrix.shape[0]):
            for prediction_index in range(0, cost_matrix.shape[2]):
                for target_index in range(0, cost_matrix.shape[1] - 1):
                    cost_from_cost_matrix = cost_matrix[image_index, target_index, prediction_index]

                    human_pred_logits = outputs['human_pred_logits'][image_index][prediction_index].unsqueeze(0)
                    target_human_labels = targets[image_index]['human_labels'][target_index].unsqueeze(0)
                    ce_human_class = F.cross_entropy(human_pred_logits, target_human_labels)

                    object_pred_logits = outputs['object_pred_logits'][image_index][prediction_index].unsqueeze(0)
                    target_object_labels = targets[image_index]['object_labels'][target_index].unsqueeze(0)
                    ce_object_class = F.cross_entropy(object_pred_logits, target_object_labels)

                    action_pred_logits = outputs['action_pred_logits'][image_index][prediction_index].unsqueeze(0)
                    target_action_labels = targets[image_index]['action_labels'][target_index].unsqueeze(0)
                    ce_action_class = F.cross_entropy(action_pred_logits, target_action_labels)

                    occlusion_pred_logits = outputs['occlusion_pred_logits'][image_index][prediction_index].unsqueeze(0)
                    target_occlusion_labels = targets[image_index]['occlusion_labels'][target_index].unsqueeze(0)
                    ce_occlusion_class = F.cross_entropy(occlusion_pred_logits, target_occlusion_labels)

                    class_cost = ce_human_class + ce_object_class + 2 * ce_action_class + 2 * ce_occlusion_class

                    human_pred_boxes = outputs['human_pred_boxes'][image_index][prediction_index].unsqueeze(0)
                    target_huamn_boxes = targets[image_index]['human_boxes'][target_index].unsqueeze(0)
                    l1_human_boxes = F.l1_loss(human_pred_boxes, target_huamn_boxes, reduction='none').sum()
                    box1 = box_ops.box_cxcywh_to_xyxy(human_pred_boxes)
                    box2 = box_ops.box_cxcywh_to_xyxy(target_huamn_boxes)
                    giou_human_boxes = 1 - torch.diag(box_ops.generalized_box_iou(box1, box2))

                    object_pred_boxes = outputs['object_pred_boxes'][image_index][prediction_index].unsqueeze(0)
                    target_object_boxes = targets[image_index]['object_boxes'][target_index].unsqueeze(0)
                    l1_object_boxes = F.l1_loss(object_pred_boxes, target_object_boxes, reduction='none').sum()
                    box1 = box_ops.box_cxcywh_to_xyxy(object_pred_boxes)
                    box2 = box_ops.box_cxcywh_to_xyxy(target_object_boxes)
                    giou_object_boxes = 1 - torch.diag(box_ops.generalized_box_iou(box1, box2))

                    reg_cost = giou_human_boxes + giou_object_boxes

                    cost = class_cost + reg_cost

                    difference = torch.abs(cost_from_cost_matrix - cost)

                    differences[image_index, target_index, prediction_index] = difference


        # Backgrounds
        for image_index in range(0, cost_matrix.shape[0]):
            for prediction_index in range(0, cost_matrix.shape[2]):

                # dummpy target index
                target_index = 0
                actual_target_index = cost_matrix.shape[1] - 1

                cost_from_cost_matrix = cost_matrix[image_index, actual_target_index, prediction_index]

                human_pred_logits = outputs['human_pred_logits'][image_index][prediction_index].unsqueeze(0)
                target_human_labels = targets[image_index]['human_labels'][target_index].unsqueeze(0)
                target_human_labels = torch.ones_like(target_human_labels) * 602
                ce_human_class = F.cross_entropy(human_pred_logits, target_human_labels)

                object_pred_logits = outputs['object_pred_logits'][image_index][prediction_index].unsqueeze(0)
                target_object_labels = targets[image_index]['object_labels'][target_index].unsqueeze(0)
                target_object_labels = torch.ones_like(target_object_labels) * 602
                ce_object_class = F.cross_entropy(object_pred_logits, target_object_labels)

                action_pred_logits = outputs['action_pred_logits'][image_index][prediction_index].unsqueeze(0)
                target_action_labels = targets[image_index]['action_labels'][target_index].unsqueeze(0)
                target_action_labels = torch.ones_like(target_action_labels) * 4
                ce_action_class = F.cross_entropy(action_pred_logits, target_action_labels)

                occlusion_pred_logits = outputs['occlusion_pred_logits'][image_index][prediction_index].unsqueeze(0)
                target_occlusion_labels = targets[image_index]['occlusion_labels'][target_index].unsqueeze(0)
                target_occlusion_labels = torch.ones_like(target_occlusion_labels) * 4
                ce_occlusion_class = F.cross_entropy(occlusion_pred_logits, target_occlusion_labels)

                class_cost = ce_human_class + ce_object_class + 2 * ce_action_class + 2 * ce_occlusion_class


                cost = class_cost

                difference = torch.abs(cost_from_cost_matrix - cost)

                differences[image_index, actual_target_index, prediction_index] = difference

        return differences





    def forward(self, outputs, targets, training=True):

        with torch.no_grad():
            if training:
                loss_cls, human_target_classes, object_target_classes, \
                action_target_classes, occlusion_target_classes \
                    = self.loss_cls(outputs, targets)
                loss_reg, human_target_boxes, object_target_boxes \
                    = self.loss_reg(outputs, targets)
                loss_reg = torch.cat(
                    [loss_reg, torch.zeros_like(loss_reg)[:, 0:1, :]], dim=1)
                cost_matrix = loss_cls + self.alpha * loss_reg

                if TEST_COST_MATRIX:
                    differences = self.test_cost_matrix(outputs,targets,cost_matrix)

                n = self.num_queries
                m = cost_matrix.shape[1] - 1
                k = self.k

                # supplying vector s
                s = torch.ones(m + 1, dtype=int, device=cost_matrix.device) * -1
                s[0:m + 1] = k
                s[-1] = n - m * k

                # demanding vector d
                d = torch.ones(n, dtype=int, device=cost_matrix.device)

                # optimal assigning plan π
                _, pi = self.sinkhorn(s, d, cost_matrix)

                # Rescale pi so that the max pi for each gt equals to 1.
                rescale_factor, _ = pi.max(dim=2)
                pi = pi / rescale_factor.unsqueeze(2)

                # Process targets using pi
                max_assigned_units, matched_gt_inds = torch.max(pi, dim=1)
                fg_mask = matched_gt_inds != m

                batch_size = cost_matrix.shape[0]
                num_human_classes = outputs['human_pred_logits'].shape[-1]
                num_object_classes = outputs['object_pred_logits'].shape[-1]
                num_action_classes = outputs['action_pred_logits'].shape[-1]
                num_occlusion_classes = outputs['occlusion_pred_logits'].shape[
                    -1]

                gt_human_classes = torch.ones([batch_size, self.num_queries],
                                              dtype=int,
                                              device=cost_matrix.device) * (
                                               num_human_classes - 1)
                gt_object_classes = torch.ones([batch_size, self.num_queries],
                                               dtype=int,
                                               device=cost_matrix.device) * (
                                                num_object_classes - 1)
                gt_action_classes = torch.ones([batch_size, self.num_queries],
                                               dtype=int,
                                               device=cost_matrix.device) * (
                                                num_action_classes - 1)
                gt_occlusion_classes = torch.ones(
                    [batch_size, self.num_queries], dtype=int,
                    device=cost_matrix.device) * (num_occlusion_classes - 1)
                gt_human_boxes = torch.zeros([batch_size, self.num_queries, 4],
                                             device=cost_matrix.device)
                gt_object_boxes = torch.zeros([batch_size, self.num_queries, 4],
                                              device=cost_matrix.device)

                for i in range(batch_size):
                    gt_human_classes[i][fg_mask[i]] = human_target_classes[i][
                        matched_gt_inds[i][fg_mask[i]]]
                    gt_object_classes[i][fg_mask[i]] = object_target_classes[i][
                        matched_gt_inds[i][fg_mask[i]]]
                    gt_action_classes[i][fg_mask[i]] = action_target_classes[i][
                        matched_gt_inds[i][fg_mask[i]]]
                    gt_occlusion_classes[i][fg_mask[i]] = \
                    occlusion_target_classes[i][matched_gt_inds[i][fg_mask[i]]]
                    gt_human_boxes[i][fg_mask[i]] = \
                    human_target_boxes[i].reshape(2, 4)[
                        matched_gt_inds[i][fg_mask[i]]]
                    gt_object_boxes[i][fg_mask[i]] = \
                    object_target_boxes[i].reshape(2, 4)[
                        matched_gt_inds[i][fg_mask[i]]]

            else:
                # TODO: In the validation and test sets, number of targets for
                #  each image may vary
                raise NotImplementedError()

        losses = self.loss_computation(outputs, gt_human_classes,
                                       gt_object_classes, gt_action_classes,
                                       gt_occlusion_classes, gt_human_boxes,
                                       gt_object_boxes)

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                l_dict = self.loss_computation(outputs, gt_human_classes,
                                       gt_object_classes, gt_action_classes,
                                       gt_occlusion_classes, gt_human_boxes,
                                       gt_object_boxes, log=False)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class SinkhornDistance(torch.nn.Module):
    r"""
        Given two empirical measures each with
          P_1 locations x in R^{D_1} and
          P_2 locations y in R^{D_2}`,
        outputs an approximation of the regularized OT cost for point clouds.

        Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional):
            Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'.
            'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
              elements in the output,
            'sum': the output will be summed. Default: 'none'

        Shape:
            - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
            - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, eps=1e-3, max_iter=100, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, mu, nu, C):
        '''
        mu: supplying vector s      [m+1] (num_gt_relations + 1)
        nu: demanding vector d      [n] (num_queries)              n = 100
        C: costs matrix             [m+1, n]

        output shape: [m+1, n] (the same as that of cost matrix)

        maximum numbers of gt relations is 454
        maximum numbers of gt objects is 22

        It is expected to have the same number of predictions and ground truth
            relations. However, there might sometimes be fewer relations
            in the ground truth.

        n − m × k > 0
        n cannot be very large, so k must be very small (<1)

        On average, k = 0.1 seems appropriate.
        If possible, increase num_queries so that the model can
        produce enough predictions.

        '''
        u = torch.ones_like(mu)
        v = torch.ones_like(nu)

        # Sinkhorn iterations
        for i in range(self.max_iter):
            v = self.eps * \
                (torch.log(
                    nu + 1e-8) - torch.logsumexp(
                    self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            u = self.eps * \
                (torch.log(
                    mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(
            self.M(C, U, V)).detach()
        # Sinkhorn distance
        cost = torch.sum(
            pi * C, dim=(-2, -1))
        return cost, pi

    def M(self, C, u, v):
        '''
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / epsilon$"
        '''
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps
