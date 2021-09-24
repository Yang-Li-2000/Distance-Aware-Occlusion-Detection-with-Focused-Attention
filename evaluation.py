from engine import *
from datasets.two_point_five_vrd import *
from util.box_ops import box_cxcywh_to_xyxy


# Not used
def evaluate_on_training_set(hoi_list, targets):

    batch_size = len(targets)

    tp = 0
    fp = 0
    fn = 0

    for i in range(batch_size):
        current_hoi_list = hoi_list[i]['hoi_list']
        current_target = targets[i]

        target_human_boxes = current_target['human_boxes']
        target_object_boxes = current_target['object_boxes']
        target_human_labels = current_target['human_labels']
        target_object_labels = current_target['object_labels']
        target_action_labels = current_target['action_labels']

        # decide whether to swap
        need_to_swap = False
        pred_human_box = current_hoi_list[0]['h_box']
        iou_1 = IoU(pred_human_box, target_human_boxes[0])
        iou_2 = IoU(pred_human_box, target_human_boxes[1])
        if iou_2 > iou_1:
            need_to_swap = True

        if need_to_swap:
            # swap target_human_boxes
            temp = target_human_boxes[0]
            target_human_boxes[0] = target_human_boxes[1]
            target_human_boxes[1] = temp

            # swap target_object_boxes
            temp = target_object_boxes[0]
            target_object_boxes[0] = target_object_boxes[1]
            target_object_boxes[1] = temp

            # swap target_human_labels
            temp = target_human_labels[0]
            target_human_labels[0] = target_human_labels[1]
            target_human_labels[1] = temp

            # swap target_object_labels
            temp = target_object_labels[0]
            target_object_labels[0] = target_object_labels[1]
            target_object_labels[1] = temp

            # swap target_action_labels
            temp = target_action_labels[0]
            target_action_labels[0] = target_action_labels[1]
            target_action_labels[1] = temp


        for j in range(0, 2):

            hoi = current_hoi_list[j]

            pred_human_box = hoi['h_box']
            pred_object_box = hoi['o_box']
            pred_human_name = hoi['h_name']
            pred_object_name = hoi['o_name']
            pred_interaction_name = hoi['i_name']

            w, h = targets[i]['org_size']
            # transform from normalized cxcywh to xyxy
            target_human_box_transformed = target_human_boxes[j]
            device = target_human_box_transformed.device
            # 1. normalized cxcywh to normalized xyxy
            target_human_box_transformed = box_cxcywh_to_xyxy(target_human_box_transformed)
            # 2. normalized xyxy to xyxy
            target_human_box_transformed = target_human_box_transformed * torch.tensor([w, h, w, h]).to(device)



            # transform from normalized cxcywh to xyxy
            target_object_box_transformed = target_object_boxes[j]
            device = target_object_box_transformed.device
            # 1. normalized cxcywh to normalized xyxy
            target_object_box_transformed = box_cxcywh_to_xyxy(target_object_box_transformed)
            # 2. normalized xyxy to xyxy
            target_object_box_transformed = target_object_box_transformed * torch.tensor([w, h, w, h]).to(device)



            iou_human = IoU(pred_human_box, target_human_box_transformed)
            iou_object = IoU(pred_object_box, target_object_box_transformed)
            correct_human_name = name_to_index(pred_human_name) == target_human_labels[j].item()
            correct_object_name = name_to_index(pred_object_name) == target_object_labels[j].item()

            correct_interaction = distance_name_to_id[pred_interaction_name] == target_action_labels[j].item()

            print('pred human and target human boxes:')
            print(pred_human_box)
            print(target_human_box_transformed)
            print(iou_human, iou_object, correct_interaction)

            if iou_human > 0.5 and iou_object > 0.5 and correct_interaction:
                tp += 1

    return tp





# For training set, the number of output hoi is fixed to 2
def construct_evaluation_output_using_hoi_list(hoi_list, original_targets,
                                               image_id_1_list,
                                               entity_1_list,
                                               xmin_1_list,
                                               xmax_1_list,
                                               ymin_1_list,
                                               ymax_1_list,
                                               image_id_2_list,
                                               entity_2_list,
                                               xmin_2_list,
                                               xmax_2_list,
                                               ymin_2_list,
                                               ymax_2_list,
                                               distance_list,
                                               occlusion_list
                                               ):

    # batch level
    for i in range(len(hoi_list)):
        current_image_id = hoi_list[i]['image_id']
        # hoi level
        num_bounding_boxes_in_ground_truth = original_targets[i]['num_bounding_boxes_in_ground_truth']
        num_hoi_to_produce = num_bounding_boxes_in_ground_truth * (num_bounding_boxes_in_ground_truth - 1)
        org_size = original_targets[i]['org_size']
        hh, ww = org_size

        # Handle index out of bound exception.
        # TODO: Record the numbers that caused the exceptions.
        num_hoi_to_produce = min(num_hoi_to_produce, len(hoi_list[i]['hoi_list']))

        for j in range(num_hoi_to_produce):
            current_hoi = hoi_list[i]['hoi_list'][j]

            image_id_1 = current_image_id
            entity_1 = name_to_entity(current_hoi['h_name'])
            xmin_1, ymin_1, xmax_1, ymax_1 = current_hoi['h_box']
            xmin_1, ymin_1, xmax_1, ymax_1 = xmin_1/ww, ymin_1/hh, xmax_1/ww, ymax_1/hh

            image_id_2 = current_image_id
            entity_2 = name_to_entity(current_hoi['o_name'])
            xmin_2, ymin_2, xmax_2, ymax_2 = current_hoi['o_box']
            xmin_2, ymin_2, xmax_2, ymax_2 = xmin_2/ww, ymin_2/hh, xmax_2/ww, ymax_2/hh

            distance = distance_name_to_id[current_hoi['i_name']]
            occlusion = occlusion_name_to_id[current_hoi['ocl_name']]

            image_id_1_list.append(image_id_1[:-4])
            entity_1_list.append(entity_1)
            xmin_1_list.append(xmin_1.item())
            xmax_1_list.append(xmax_1.item())
            ymin_1_list.append(ymin_1.item())
            ymax_1_list.append(ymax_1.item())
            image_id_2_list.append(image_id_2[:-4])
            entity_2_list.append(entity_2)
            xmin_2_list.append(xmin_2.item())
            xmax_2_list.append(xmax_2.item())
            ymin_2_list.append(ymin_2.item())
            ymax_2_list.append(ymax_2.item())
            distance_list.append(int(distance))
            occlusion_list.append(int(occlusion))

    return