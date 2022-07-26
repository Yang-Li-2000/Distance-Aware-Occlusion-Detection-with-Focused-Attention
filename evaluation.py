# ------------------------------------------------------------------------
# Licensed under the Apache License, Version 2.0 (the "License")
# ------------------------------------------------------------------------
# Copyright (c) Yang Li and Yucheng Tu. All Rights Reserved
# ------------------------------------------------------------------------

from engine import *
from datasets.two_point_five_vrd import *
from util.box_ops import box_cxcywh_to_xyxy



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
                                               occlusion_list,
                                               index_list=None
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

        if VISUALIZE_ATTENTION_WEIGHTS:
            num_hoi_to_produce = len(hoi_list[i]['hoi_list'])

        for j in range(num_hoi_to_produce):
            current_hoi = hoi_list[i]['hoi_list'][j]

            image_id_1 = current_image_id
            image_id_2 = current_image_id

            # object A name
            object_A_name = current_hoi['h_name']
            entity_1 = name_to_entity(object_A_name)

            # object A box
            xmin_1, ymin_1, xmax_1, ymax_1 = current_hoi['h_box']
            xmin_1, ymin_1, xmax_1, ymax_1 = xmin_1/ww, ymin_1/hh, xmax_1/ww, ymax_1/hh

            # object B name
            object_B_name = current_hoi['o_name']
            entity_2 = name_to_entity(object_B_name)

            # object B box
            xmin_2, ymin_2, xmax_2, ymax_2 = current_hoi['o_box']
            xmin_2, ymin_2, xmax_2, ymax_2 = xmin_2/ww, ymin_2/hh, xmax_2/ww, ymax_2/hh

            # distance name
            distance = distance_name_to_id[current_hoi['i_name']]

            # occlusion name
            occlusion = occlusion_name_to_id[current_hoi['ocl_name']]

            # TODO: record confidence level of objects to facilitate debugging

            assert type(image_id_1[:-4]) == str
            assert type(entity_1) == str
            assert type(xmin_1.item()) == float
            assert type(xmax_1.item()) == float
            assert type(ymin_1.item()) == float
            assert type(ymax_1.item()) == float
            assert type(image_id_2[:-4]) == str
            assert type(entity_2) == str
            assert type(xmin_2.item()) == float
            assert type(xmax_2.item()) == float
            assert type(ymin_2.item()) == float
            assert type(ymax_2.item()) == float
            assert type(distance) == int
            assert type(occlusion) == int

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
            distance_list.append(distance)
            occlusion_list.append(occlusion)

            # index
            if index_list is not None:
                index = current_hoi['index']
                assert type(index) == int
                index_list.append(index)

    return