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
from scipy.optimize import linear_sum_assignment
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
        if PREDICT_INTERSECTION_BOX:
            self.intersection_box_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, samples: NestedTensor, pos_depth=None, writer=None):
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

        if pos_depth is not None:
            # Make sure shape of encoded depth is the same as that of positional encoding
            assert pos[-1].shape == pos_depth.shape
            # Add pos_depth to positional encoding
            pos[-1] = pos[-1] + pos_depth

        if VISUALIZE_ATTENTION_WEIGHTS:
            device = samples.tensors.device
            mean = torch.tensor([0.38582161319756497, 0.417059363143913, 0.44746641122649666]).unsqueeze(1).unsqueeze(1).to(device)
            std = torch.tensor([0.2928927708221023, 0.28587472243230755, 0.2924566717392719]).unsqueeze(1).unsqueeze(1).to(device)
            writer.add_image("image", samples.tensors[0] * std + mean)

        if CASCADE:
            hs, distance_decoder_out, occlusion_decoder_out = \
                self.transformer(self.input_proj(src), mask, self.query_embed.weight,
                                 pos[-1])[:3]
        else:
            hs = \
                self.transformer(self.input_proj(src), mask, self.query_embed.weight,
                                 pos[-1], writer=writer)[0]

        human_outputs_class = self.human_cls_embed(hs)
        human_outputs_coord = self.human_box_embed(hs).sigmoid()
        object_outputs_class = self.object_cls_embed(hs)
        object_outputs_coord = self.object_box_embed(hs).sigmoid()


        if CASCADE:
            action_outputs_class = self.action_cls_embed(distance_decoder_out)
            occlusion_outputs_class = self.occlusion_cls_embed(occlusion_decoder_out)
            if PREDICT_INTERSECTION_BOX:
                intersection_outputs_coord = self.intersection_box_embed(occlusion_decoder_out).sigmoid()
        else:
            action_outputs_class = self.action_cls_embed(hs)
            occlusion_outputs_class = self.occlusion_cls_embed(hs)
            if PREDICT_INTERSECTION_BOX:
                intersection_outputs_coord = self.intersection_box_embed(hs).sigmoid()

        out = {
            'human_pred_logits': human_outputs_class[-1],
            'human_pred_boxes': human_outputs_coord[-1],
            'object_pred_logits': object_outputs_class[-1],
            'object_pred_boxes': object_outputs_coord[-1],
            'action_pred_logits': action_outputs_class[-1],
            'occlusion_pred_logits': occlusion_outputs_class[-1]
        }

        if PREDICT_INTERSECTION_BOX:
            out['intersection_pred_boxes'] = intersection_outputs_coord[-1]

        if self.aux_loss:
            if PREDICT_INTERSECTION_BOX:
                out['aux_outputs'] = self._set_aux_loss_intersection(
                    human_outputs_class,
                    human_outputs_coord,
                    object_outputs_class,
                    object_outputs_coord,
                    action_outputs_class,
                    occlusion_outputs_class,
                    intersection_outputs_coord
                )
            else:
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
            'occlusion_pred_logits': f
        } for
            a,
            b,
            c,
            d,
            e,
            f
            in zip(
                human_outputs_class[:-1],
                human_outputs_coord[:-1],
                object_outputs_class[:-1],
                object_outputs_coord[:-1],
                action_outputs_class[:-1],
                occlusion_outputs_class[:-1]
            )]

    @torch.jit.unused
    def _set_aux_loss_intersection(self,
                      human_outputs_class,
                      human_outputs_coord,
                      object_outputs_class,
                      object_outputs_coord,
                      action_outputs_class,
                      occlusion_outputs_class,
                      intersection_outputs_coord
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
            'intersection_pred_boxes': g
        } for
            a,
            b,
            c,
            d,
            e,
            f,
            g
            in zip(
                human_outputs_class[:-1],
                human_outputs_coord[:-1],
                object_outputs_class[:-1],
                object_outputs_coord[:-1],
                action_outputs_class[:-1],
                occlusion_outputs_class[:-1],
                intersection_outputs_coord[:-1]
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
        if PREDICT_INTERSECTION_BOX:
            intersection_src_boxes = outputs['intersection_pred_boxes'][idx]
            intersection_target_boxes = torch.cat(
                [t['intersection_boxes'][i] for t, (_, i) in zip(targets, indices)],
                dim=0)
            intersection_loss_bbox = F.l1_loss(intersection_src_boxes, intersection_target_boxes,
                                    reduction='none')

        losses = dict()
        losses['human_loss_bbox'] = human_loss_bbox.sum() / num_boxes
        losses['object_loss_bbox'] = object_loss_bbox.sum() / num_boxes
        if PREDICT_INTERSECTION_BOX:
            losses['intersection_loss_bbox'] = intersection_loss_bbox.sum() / num_boxes
            losses['loss_bbox'] = losses['human_loss_bbox'] + losses[
                'object_loss_bbox'] + losses['intersection_loss_bbox']
        else:
            losses['loss_bbox'] = losses['human_loss_bbox'] + losses[
                'object_loss_bbox']

        human_loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(human_src_boxes),
            box_ops.box_cxcywh_to_xyxy(human_target_boxes)))
        object_loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(object_src_boxes),
            box_ops.box_cxcywh_to_xyxy(object_target_boxes)))
        if PREDICT_INTERSECTION_BOX:
            intersection_loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(intersection_src_boxes),
                box_ops.box_cxcywh_to_xyxy(intersection_target_boxes)))

        losses['human_loss_giou'] = human_loss_giou.sum() / num_boxes
        losses['object_loss_giou'] = object_loss_giou.sum() / num_boxes

        if PREDICT_INTERSECTION_BOX:
            losses['intersection_loss_giou'] = intersection_loss_giou.sum() / num_boxes
            losses['loss_giou'] = losses['human_loss_giou'] + losses[
                'object_loss_giou'] + losses['intersection_loss_giou']
        else:
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

    def forward(self, outputs, targets, optimal_transport=None, training=True):
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
        if USE_OPTIMAL_TRANSPORT and training:
            with torch.no_grad():
                indices = OptimalTransport.forward(optimal_transport, outputs_without_aux, targets, indices_only=True)
        else:
            indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes,
        # for normalization purposes
        if USE_OPTIMAL_TRANSPORT and training:
            num_boxes = np.sum([len(k[0]) for k in indices])
        else:
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

    weight_dict = dict(loss_ce=1,
                       loss_relation=args.relation_loss_coef,
                       loss_bbox=args.bbox_loss_coef,
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
#  6. (Done) Loss computation
class OptimalTransport(nn.Module):
    """
    This class uses optimal transport to assign targets to queries.
        After that, it compute the losses using the assigned targets.
    """

    def __init__(self, args, alpha=1, num_queries=100, k=1, eps=0.1, max_iter=50):
        super().__init__()
        self.alpha = alpha
        self.num_queries = args.num_queries
        # the number of positive anchors for each gt
        self.k = OT_k
        self.eps = eps
        self.sinkhorn = SinkhornDistance(eps=eps, max_iter=max_iter)

        self.cost_class = args.set_cost_class
        self.cost_bbox = args.set_cost_bbox
        self.cost_giou = args.set_cost_giou

        weight_dict = dict(loss_ce=1, loss_bbox=args.bbox_loss_coef,
                           loss_giou=args.giou_loss_coef)
        if args.aux_loss:
            aux_weight_dict = {}
            for i in range(args.dec_layers - 1):
                aux_weight_dict.update(
                    {k + f'_{i}': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        self.weight_dict = weight_dict

        # Weights for foreground classes and background classes: 1 : 0.02
        self.eos_coef = args.eos_coef
        num_actions = 4
        self.num_actions = num_actions

        human_empty_weight = torch.ones(num_humans + 1)
        human_empty_weight[-1] = self.eos_coef
        self.register_buffer('human_empty_weight', human_empty_weight)

        object_empty_weight = torch.ones(num_humans + 1)
        object_empty_weight[-1] = self.eos_coef
        self.register_buffer('object_empty_weight', object_empty_weight)

        action_empty_weight = torch.ones(num_actions + 1)
        action_empty_weight[-1] = self.eos_coef
        self.register_buffer('action_empty_weight', action_empty_weight)

        occlusion_empty_weight = torch.ones(num_actions + 1)
        occlusion_empty_weight[-1] = self.eos_coef
        self.register_buffer('occlusion_empty_weight', occlusion_empty_weight)


        device = torch.device(args.device)
        self.to(device)

    def dynamic_k_estimate(self, outputs, targets, training=True):
        """
        Estimate the value of k from the first 10 closest queries.
        :param outputs: outputs of the model after the forward pass.
        :param targets:
        :param training:
        """
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
                1).expand(-1, self.num_queries, -1)
            object_target_boxes_1 = object_target_boxes[:, 0:4].unsqueeze(
                1).expand(-1, self.num_queries, -1)
            loss_human_boxes_1,_ = box_ops.box_iou(box_ops.box_cxcywh_to_xyxy(human_src_boxes.reshape(-1,4)),
                                                   box_ops.box_cxcywh_to_xyxy(human_target_boxes_1.reshape(-1,4)))
            loss_object_boxes_1,_ = box_ops.box_iou(box_ops.box_cxcywh_to_xyxy(object_src_boxes.reshape(-1,4)),
                                                 box_ops.box_cxcywh_to_xyxy(object_target_boxes_1.reshape(-1,4)))

            loss_boxes_1 = 0.5 * torch.diag(loss_human_boxes_1).reshape(-1, self.num_queries) + 0.5 * torch.diag(loss_object_boxes_1).reshape(-1, self.num_queries)

            values_1, _= torch.topk(loss_boxes_1, 10, dim=1)

            k_1 = values_1.sum(dim=1)

            # the second target
            human_target_boxes_2 = human_target_boxes[:, 4:].unsqueeze(
                1).expand(-1, self.num_queries, -1)
            object_target_boxes_2 = object_target_boxes[:, 4:].unsqueeze(
                1).expand(-1, self.num_queries, -1)
            loss_human_boxes_2, _ = box_ops.box_iou(box_ops.box_cxcywh_to_xyxy(human_src_boxes.reshape(-1, 4)),
                                                    box_ops.box_cxcywh_to_xyxy(
                                                        human_target_boxes_2.reshape(-1, 4)))
            loss_object_boxes_2, _ = box_ops.box_iou(box_ops.box_cxcywh_to_xyxy(object_src_boxes.reshape(-1, 4)),
                                                     box_ops.box_cxcywh_to_xyxy(
                                                         object_target_boxes_2.reshape(-1, 4)))

            loss_boxes_2 = 0.5 * torch.diag(loss_human_boxes_1).reshape(-1, self.num_queries) + 0.5 * torch.diag(loss_object_boxes_1).reshape(-1,
                                                                                                                     self.num_queries)

            values_2, _ = torch.topk(loss_boxes_2, 10, dim=1)

            k_2 = values_2.sum(dim=1)

            return k_1, k_2

    @torch.no_grad()
    def loss_cls(self, outputs, targets, training=True, log=True):
        """
        Compute the classification cost matrix. TODO
        :param outputs: outputs of the model after the forward pass.
        :param targets:
        :param training: Whether is using the train set.
            The training set has exactly 2 annotations for each image.
            Validation and test sets have various numbers of annotations
            for each image.
        :param log: Not used.
        :return: TODO
        """

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

            loss_human_classes_bg = BG_COEF * F.cross_entropy(
                human_src_logits.permute(0, 2, 1), bg_human_classes,
                reduction='none').unsqueeze(1)
            loss_object_classes_bg = BG_COEF * F.cross_entropy(
                object_src_logits.permute(0, 2, 1), bg_object_classes,
                reduction='none').unsqueeze(1)
            loss_action_classes_bg = BG_COEF * F.cross_entropy(
                action_src_logits.permute(0, 2, 1), bg_action_classes,
                reduction='none').unsqueeze(1)
            loss_occlusion_classes_bg = BG_COEF * F.cross_entropy(
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

        beta_1, beta_2 = 1.2, 1
        alpha_h, alpha_o, alpha_r = 1, 1, 2

        l_cls_h = alpha_h * self.cost_class * human_loss_cls
        l_cls_o = alpha_o * self.cost_class * object_loss_cls
        l_cls_r = alpha_r * self.cost_class * action_loss_cls
        l_cls_occlusion = alpha_r * self.cost_class * occlusion_loss_cls

        l_cls_all = (l_cls_h + l_cls_o + l_cls_r + l_cls_occlusion) / (alpha_h + alpha_o + alpha_r + alpha_r)

        loss_cls = beta_1 * l_cls_all

        return loss_cls, human_target_classes, object_target_classes, action_target_classes, occlusion_target_classes

    #@torch.no_grad()
    def loss_reg(self, outputs, targets, training=True, log=True):
        """
        TODO
        :param outputs:
        :param targets:
        :param training:
        :param log:
        :return:
        """
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
            if PREDICT_INTERSECTION_BOX:
                intersection_src_boxes = outputs['intersection_pred_boxes']
                intersection_target_boxes = store_to_list('intersection_boxes')

            # the first target
            human_target_boxes_1 = human_target_boxes[:, 0:4].unsqueeze(
                1).expand(-1, num_queries, -1)
            object_target_boxes_1 = object_target_boxes[:, 0:4].unsqueeze(
                1).expand(-1, num_queries, -1)

            loss_human_boxes_1 = F.l1_loss(human_src_boxes, human_target_boxes_1, reduction='none').sum(dim=2).unsqueeze(1)
            loss_object_boxes_1 = F.l1_loss(object_src_boxes, object_target_boxes_1, reduction='none').sum(dim=2).unsqueeze(1)
            human_loss_giou_1 = (1 - torch.diag(box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(human_src_boxes.reshape(-1, 4)),
                box_ops.box_cxcywh_to_xyxy(
                    human_target_boxes_1.reshape(-1, 4)))))
            human_loss_giou_1 = \
                human_loss_giou_1.reshape(-1, num_queries).unsqueeze(1)

            object_loss_giou_1 = (1 - torch.diag(box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(object_src_boxes.reshape(-1, 4)),
                box_ops.box_cxcywh_to_xyxy(object_target_boxes_1.reshape(-1, 4)))))
            object_loss_giou_1 \
                = object_loss_giou_1.reshape(-1, num_queries).unsqueeze(1)

            if PREDICT_INTERSECTION_BOX:
                intersection_target_boxes_1 = intersection_target_boxes[:, 0:4].unsqueeze(
                    1).expand(-1, num_queries, -1)
                loss_intersection_boxes_1 = F.l1_loss(intersection_src_boxes, intersection_target_boxes_1, reduction='none').sum(dim=2).unsqueeze(1)
                intersection_loss_giou_1 = (1 - torch.diag(box_ops.generalized_box_iou(
                    box_ops.box_cxcywh_to_xyxy(intersection_src_boxes.reshape(-1, 4)),
                    box_ops.box_cxcywh_to_xyxy(
                        intersection_target_boxes_1.reshape(-1, 4)))))
                intersection_loss_giou_1 = \
                    intersection_loss_giou_1.reshape(-1, num_queries).unsqueeze(1)

            # the second target
            human_target_boxes_2 = human_target_boxes[:, 4:].unsqueeze(
                1).expand(-1, self.num_queries, -1)
            object_target_boxes_2 = object_target_boxes[:, 4:].unsqueeze(
                1).expand(-1, self.num_queries, -1)

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

            if PREDICT_INTERSECTION_BOX:
                intersection_target_boxes_2 = intersection_target_boxes[:, 4:].unsqueeze(
                    1).expand(-1, self.num_queries, -1)
                loss_intersection_boxes_2 = F.l1_loss(intersection_src_boxes, intersection_target_boxes_2, reduction='none').sum(dim=2).unsqueeze(1)
                intersection_loss_giou_2 = (1 - torch.diag(box_ops.generalized_box_iou(
                    box_ops.box_cxcywh_to_xyxy(intersection_src_boxes.reshape(-1, 4)),
                    box_ops.box_cxcywh_to_xyxy(
                        intersection_target_boxes_2.reshape(-1, 4)))))
                intersection_loss_giou_2 = \
                    intersection_loss_giou_2.reshape(-1, num_queries).unsqueeze(1)

            # combine them
            human_loss_boxes = torch.cat([loss_human_boxes_1, loss_human_boxes_2], dim=1)
            object_loss_boxes = torch.cat([loss_object_boxes_1, loss_object_boxes_2], dim=1)


            human_loss_giou = torch.cat([human_loss_giou_1, human_loss_giou_2],
                                        dim=1)
            object_loss_giou = torch.cat(
                [object_loss_giou_1, object_loss_giou_2], dim=1)

            if PREDICT_INTERSECTION_BOX:
                intersection_loss_boxes = torch.cat([loss_intersection_boxes_1, loss_intersection_boxes_2], dim=1)
                intersection_loss_giou = torch.cat(
                    [intersection_loss_giou_1, intersection_loss_giou_2], dim=1)

        else:
            raise NotImplementedError()

        beta_1, beta_2 = 1.2, 1
        l_box_h = self.cost_bbox * human_loss_boxes + self.cost_giou * human_loss_giou
        l_box_o = self.cost_bbox * object_loss_boxes + self.cost_giou * object_loss_giou

        if PREDICT_INTERSECTION_BOX:
            l_box_i = self.cost_bbox * intersection_loss_boxes + self.cost_giou * intersection_loss_giou
            l_box_all = (l_box_h + l_box_o + l_box_i) / 3
        else:
            l_box_all = (l_box_h + l_box_o) / 2

        loss_reg = beta_2 * l_box_all

        if PREDICT_INTERSECTION_BOX:
            return loss_reg, human_target_boxes, object_target_boxes, intersection_target_boxes
        else:
            return loss_reg, human_target_boxes, object_target_boxes


    def loss_computation(self, num_foregrounds, outputs,
                         gt_human_classes, gt_object_classes,
                         gt_action_classes, gt_occlusion_classes,
                         gt_human_boxes, gt_object_boxes, log=True):

        """
        TODO
        :param num_foregrounds:
        :param outputs:
        :param gt_human_classes:
        :param gt_object_classes:
        :param gt_action_classes:
        :param gt_occlusion_classes:
        :param gt_human_boxes:
        :param gt_object_boxes:
        :param log:
        :return:
        """

        losses = dict()

        # loss_ce
        human_src_logits = outputs['human_pred_logits']
        object_src_logits = outputs['object_pred_logits']
        action_src_logits = outputs['action_pred_logits']
        occlusion_src_logits = outputs['occlusion_pred_logits']

        mask = gt_human_classes != 602

        human_loss_ce = F.cross_entropy(human_src_logits.permute(0, 2, 1), gt_human_classes, self.human_empty_weight)
        object_loss_ce = F.cross_entropy(object_src_logits.permute(0, 2, 1), gt_object_classes, self.object_empty_weight)
        action_loss_ce = F.cross_entropy(action_src_logits.permute(0, 2, 1), gt_action_classes, self.action_empty_weight)
        occlusion_loss_ce = F.cross_entropy(occlusion_src_logits.permute(0, 2, 1), gt_occlusion_classes, self.occlusion_empty_weight)
        loss_ce = human_loss_ce + object_loss_ce + 2 * action_loss_ce + 2 * occlusion_loss_ce
        losses['human_loss_ce'] = human_loss_ce
        losses['object_loss_ce'] = object_loss_ce
        losses['action_loss_ce'] = action_loss_ce
        losses['occlusion_loss_ce'] = occlusion_loss_ce
        losses['loss_ce'] = loss_ce

        # loss_bbox
        human_src_boxes = outputs['human_pred_boxes']
        object_src_boxes = outputs['object_pred_boxes']

        # L1 Loss. Normalized with respect to the number of foregrounds
        num_boxes = mask.sum()
        human_loss_bbox = F.l1_loss(human_src_boxes[mask], gt_human_boxes[mask], reduction='none')
        object_loss_bbox = F.l1_loss(object_src_boxes[mask], gt_object_boxes[mask], reduction='none')
        losses['human_loss_bbox'] = human_loss_bbox.sum() / num_boxes
        losses['object_loss_bbox'] = object_loss_bbox.sum() / num_boxes
        losses['loss_bbox'] = losses['human_loss_bbox'] + losses['object_loss_bbox']

        # GIoU Loss. Normalize with respect to the number of foregrounds
        human_loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(human_src_boxes[mask]),
            box_ops.box_cxcywh_to_xyxy(gt_human_boxes[mask])))
        object_loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(object_src_boxes[mask]),
            box_ops.box_cxcywh_to_xyxy(gt_object_boxes[mask])))
        losses['human_loss_giou'] = human_loss_giou.sum() / num_boxes
        losses['object_loss_giou'] = object_loss_giou.sum() / num_boxes
        losses['loss_giou'] = losses['human_loss_giou'] + losses['object_loss_giou']

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





    def forward(self, outputs, targets, training=True, indices_only=False):
        """
        TODO
        :param outputs:
        :param targets:
        :param training:
        :return:
        """

         # Use cost matrix produced by sinkhorn for back-prop
        if BACK_PROP_SINKHORN_COST and training:
            cost = None
            loss_cls, human_target_classes, object_target_classes, \
            action_target_classes, occlusion_target_classes \
                = self.loss_cls(outputs, targets)
            if PREDICT_INTERSECTION_BOX:
                loss_reg, human_target_boxes, object_target_boxes, \
                intersection_target_boxes \
                    = self.loss_reg(outputs, targets)
            else:
                loss_reg, human_target_boxes, object_target_boxes \
                    = self.loss_reg(outputs, targets)
            loss_reg = torch.cat(
                    [loss_reg, torch.zeros_like(loss_reg)[:, 0:1, :]], dim=1)
            cost_matrix = loss_cls + self.alpha * loss_reg

            if TEST_COST_MATRIX:
                differences = self.test_cost_matrix(outputs, targets, cost_matrix)
            with torch.no_grad():
                n = self.num_queries
                m = cost_matrix.shape[1] - 1

                if USE_DYNAMIC_K_ESTIMATE:
                    k_1, k_2 = self.dynamic_k_estimate(outputs,targets)
                else:
                    k_1 = k_2 = self.k * torch.ones(cost_matrix.shape[0])

                # supplying vector s
                s = torch.ones((cost_matrix.shape[0], m + 1), dtype=int, device=cost_matrix.device) * -1
                s[:, 0] = k_1
                s[:, 1] = k_2
                s[:, -1] = n - k_1 - k_2

                # demanding vector d
                d = torch.ones((cost_matrix.shape[0],  n), dtype=int, device=cost_matrix.device)

                # optimal assigning plan π
                _, pi = self.sinkhorn(s, d, cost_matrix)

                max_assigned_units, matched_gt_inds = torch.max(pi, dim=1)

                assign_plan = torch.zeros_like(pi, dtype=int, device=cost_matrix.device, requires_grad=False)

                def accelerated_lr(units):
                    if 0 <= units <= 1.0/3.0:
                        return 0
                    elif 1.0/3.0 < units <= 0.5:
                        return 0.5
                    elif 0.5 < units < 1:
                        return 1 + 0.5 * torch.exp(2 - 1/(1 - units))
                    elif units >= 1:
                        return 1

                for i in range(pi.shape[0]):
                    for j, idx in enumerate(matched_gt_inds[i]):
                        if idx != m:
                            assign_plan[i][idx][j] = accelerated_lr(max_assigned_units[i][j])

            opt_cost = assign_plan * cost_matrix

            return opt_cost.sum()

        with torch.no_grad():
            if training or indices_only:
                loss_cls, human_target_classes, object_target_classes, \
                action_target_classes, occlusion_target_classes \
                    = self.loss_cls(outputs, targets, training)

                if PREDICT_INTERSECTION_BOX:
                    loss_reg, human_target_boxes, object_target_boxes, \
                    intersection_target_boxes \
                        = self.loss_reg(outputs, targets, training)
                else:
                    loss_reg, human_target_boxes, object_target_boxes \
                        = self.loss_reg(outputs, targets, training)
                loss_reg = torch.cat(
                        [loss_reg, torch.zeros_like(loss_reg)[:, 0:1, :]], dim=1)
                cost_matrix = loss_cls + self.alpha * loss_reg

                if TEST_COST_MATRIX:
                    differences = self.test_cost_matrix(outputs,targets,cost_matrix)

                if HUNGARIAN_K_ASSIGNMENTS:
                    n = self.num_queries
                    m = cost_matrix.shape[1] - 1
                    k = self.k

                    expanded_cost_matrix = torch.zeros((cost_matrix.shape[0], 2 * k, n), dtype=float)
                    assignment_indices = 2 * k * torch.ones((cost_matrix.shape[0], n), dtype=int,
                                                       device=cost_matrix.device)

                    for i in range(cost_matrix.shape[0]):
                        expanded_cost_matrix[i][:k] = cost_matrix[i][0].unsqueeze(0).repeat(k, 1)
                        expanded_cost_matrix[i][k:2 * k] = cost_matrix[i][1].unsqueeze(0).repeat(k, 1)
                        row_ind, col_ind = linear_sum_assignment(expanded_cost_matrix[i])
                        for j in range(len(row_ind)):
                            assignment_indices[i][col_ind[j]] = row_ind[j]

                    #max_assigned_units, matched_gt_inds = torch.max(assignment_matrix, dim=1)
                    matched_gt_inds = assignment_indices // k
                    fg_mask = matched_gt_inds < 2

                else:

                    if NORMALIZED_MAX:

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

                    else:

                        n = self.num_queries
                        m = cost_matrix.shape[1] - 1
                        k = self.k

                        expanded_cost_matrix = torch.zeros((cost_matrix.shape[0], n, n), dtype=float, device=cost_matrix.device)

                        for i in range(cost_matrix.shape[0]):
                            expanded_cost_matrix[i][:k] = cost_matrix[i][0].unsqueeze(0).repeat(k,1)
                            expanded_cost_matrix[i][k:2*k] = cost_matrix[i][1].unsqueeze(0).repeat(k,1)
                            expanded_cost_matrix[i][2*k:] = cost_matrix[i][2].unsqueeze(0).repeat(n - 2*k, 1)

                        # supplying vector s
                        s = torch.ones(n, dtype=int, device=cost_matrix.device)
                        #s[0:m + 1] = k
                        #s[-1] = n - m * k

                        # demanding vector d
                        d = torch.ones(n, dtype=int, device=cost_matrix.device)

                        # optimal assigning plan π
                        _, pi = self.sinkhorn(s, d, expanded_cost_matrix)

                        # Rescale pi so that the max pi for each gt equals to 1.
                        #rescale_factor, _ = pi.max(dim=2)
                        #pi = pi / rescale_factor.unsqueeze(2)

                        # Process targets using pi
                        max_assigned_units, matched_gt_inds = torch.max(pi, dim=1)
                        matched_gt_inds = matched_gt_inds // k
                        fg_mask = matched_gt_inds < m


                if indices_only:
                    # Produce indices as a list of tuples
                    # in the same format as the results produced by the Hungarian matcher
                    image_indices, query_indices = torch.where(fg_mask)
                    target_indices = matched_gt_inds[fg_mask]

                    num_images = matched_gt_inds.shape[0]
                    query_list = [None] * num_images
                    target_list = [None] * num_images
                    for i in range(num_images):
                        query_list[i] = list()
                        target_list[i] = list()
                    for i in range(len(image_indices)):
                        image_index = image_indices[i]
                        query_list[image_index].append(query_indices[i].item())
                        target_list[image_index].append(target_indices[i].item())

                    result = [(torch.tensor(q), torch.tensor(t)) for q, t in zip(query_list, target_list)]

                    return result

                if not training:
                    raise NotImplementedError()


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

    def __init__(self, eps=SINKHORN_MAX_ITER_eps, max_iter=SINKHORN_MAX_ITER, reduction='none'):
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
