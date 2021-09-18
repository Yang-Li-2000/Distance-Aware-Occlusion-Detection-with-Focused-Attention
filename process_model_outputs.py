from test import *
from magic_numbers import *
from datasets.two_point_five_vrd import *


def generate_hoi_list_using_model_outputs(args, outputs, original_targets):
    assert args.dataset_file in ['hico', 'vcoco', 'hoia', 'two_point_five_vrd'], args.dataset_file
    if args.dataset_file == 'hico':
        num_classes = 91
        num_actions = 118
        top_k = 200
        hoi_interaction_names = hoi_interaction_names_hico
        coco_instance_id_to_name = coco_instance_ID_to_name_hico
    elif args.dataset_file == 'vcoco':
        num_classes = 91
        num_actions = 30
        top_k = 35
        hoi_interaction_names = hoi_interaction_names_vcoco
        coco_instance_id_to_name = coco_instance_ID_to_name_vcoco
    elif args.dataset_file == 'hoia':
        num_classes = 12
        num_actions = 11
        top_k = 35
        hoi_interaction_names = hoi_interaction_names_hoia
        coco_instance_id_to_name = coco_instance_ID_to_name_hoia
    elif args.dataset_file == 'two_point_five_vrd':
        num_classes = 602
        num_actions = 4
        global num_humans
        num_humans = num_classes - 1
        top_k = 200
    else:
        raise NotImplementedError()

    # Store image ids and  original sizes of images
    image_id_list = list()
    org_size_list = list()
    for i in range(len(original_targets)):
        current_target = original_targets[i]
        current_image_id = current_target['image_id']
        image_id_list.append(current_image_id)
        current_org_size = current_target['org_size']
        org_size_list.append(current_org_size)


    final_hoi_result_list = []

    img_id_list = image_id_list
    org_sizes = org_size_list
    action_pred_logits = outputs['action_pred_logits']
    object_pred_logits = outputs['object_pred_logits']
    object_pred_boxes = outputs['object_pred_boxes']
    human_pred_logits = outputs['human_pred_logits']
    human_pred_boxes = outputs['human_pred_boxes']
    assert len(action_pred_logits) == len(img_id_list)

    for idx_img in range(len(action_pred_logits)):
        image_id = img_id_list[idx_img]
        hh, ww = org_sizes[idx_img]

        act_cls = torch.nn.Softmax(dim=1)(
            action_pred_logits[idx_img]).detach().cpu().numpy()[:, :-1]
        human_cls = torch.nn.Softmax(dim=1)(
            human_pred_logits[idx_img]).detach().cpu().numpy()[:, :-1]
        object_cls = torch.nn.Softmax(dim=1)(
            object_pred_logits[idx_img]).detach().cpu().numpy()[:, :-1]
        human_box = human_pred_boxes[idx_img].detach().cpu().numpy()
        object_box = object_pred_boxes[idx_img].detach().cpu().numpy()

        keep = (act_cls.argmax(axis=1) != num_actions)
        keep = keep * (human_cls.argmax(axis=1) != 2)
        keep = keep * (object_cls.argmax(axis=1) != num_classes)
        keep = keep * (act_cls > hoi_th).any(axis=1)
        keep = keep * (human_cls > human_th).any(axis=1)
        keep = keep * (object_cls > object_th).any(axis=1)

        human_idx_max_list = human_cls[keep].argmax(axis=1)
        human_val_max_list = human_cls[keep].max(axis=1)
        human_box_max_list = human_box[keep]
        object_idx_max_list = object_cls[keep].argmax(axis=1)
        object_val_max_list = object_cls[keep].max(axis=1)
        object_box_max_list = object_box[keep]
        keep_act_scores = act_cls[keep]

        keep_act_scores_1d = keep_act_scores.reshape(-1)
        top_k_idx_1d = np.argsort(-keep_act_scores_1d)[:top_k]
        box_action_pairs = [(idx_1d // num_actions, idx_1d % num_actions)
                            for idx_1d in top_k_idx_1d]

        hoi_list = []
        for idx_box, idx_action in box_action_pairs:
            # action
            i_box = (0, 0, 0, 0)
            i_cls = keep_act_scores[idx_box, idx_action]
            if args.dataset_file == 'two_point_five_vrd':
                i_name = distance_id_to_name[int(idx_action)]
            else:
                i_name = hoi_interaction_names[int(idx_action)]
            if i_name in ['__background__']:
                continue
            # human
            cid = human_idx_max_list[idx_box]
            cx, cy, w, h = human_box_max_list[idx_box]
            cx, cy, w, h = cx * ww, cy * hh, w * ww, h * hh
            h_box = list(map(int, [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w,
                                   cy + 0.5 * h]))
            h_cls = human_val_max_list[idx_box]
            if args.dataset_file == 'two_point_five_vrd':
                h_name = index_to_name(int(cid))
            else:
                h_name = coco_instance_id_to_name[int(cid)]
            # object
            cid = object_idx_max_list[idx_box]
            cx, cy, w, h = object_box_max_list[idx_box]
            cx, cy, w, h = cx * ww, cy * hh, w * ww, h * hh
            o_box = list(map(int, [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w,
                                   cy + 0.5 * h]))
            o_cls = object_val_max_list[idx_box]
            if args.dataset_file == 'two_point_five_vrd':
                o_name = index_to_name(int(cid))
            else:
                o_name = coco_instance_id_to_name[int(cid)]
            if i_cls < hoi_th or h_cls < human_th or o_cls < object_th:
                continue
            pp = dict(
                h_box=h_box, o_box=o_box, i_box=i_box, h_cls=float(h_cls),
                o_cls=float(o_cls),
                i_cls=float(i_cls), h_name=h_name, o_name=o_name,
                i_name=i_name,
            )
            hoi_list.append(pp)

        # TODO: implement a new nms. The thresholds should be changed.
        hoi_list = triplet_nms(hoi_list)
        item = dict(image_id=image_id, hoi_list=hoi_list)
        final_hoi_result_list.append(item)

    return final_hoi_result_list
