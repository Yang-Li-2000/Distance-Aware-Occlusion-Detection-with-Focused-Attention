# ------------------------------------------------------------------------
# Licensed under the Apache License, Version 2.0 (the "License")
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train functions used in main.py
"""
import math
import sys
from typing import Iterable

import torch

import util.misc as utils

from magic_numbers import *
from process_model_outputs import *
from evaluation import *
import pandas as pd


def progressBar(i, max, text):
    """
    Produce a progress bar during training.
    :param i: index of current iteration/epoch.
    :param max: max number of iterations/epochs.
    :param text: Text to print on the right of the progress bar.
    :return: None
    """
    if TRAIN_ON_ONE_IMAGE:
        return
    bar_size = 60
    j = i / max
    sys.stdout.write('\r')
    sys.stdout.write(
        f"[{'=' * int(bar_size * j):{bar_size}s}] {int(100 * j)}%  {text}")
    sys.stdout.flush()


def train_one_epoch(args, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    iteratoin_count = 0
    max_num_iterations = len(data_loader)

    tp = 0
    denominator = 0

    image_id_1_list = list()
    entity_1_list = list()
    xmin_1_list = list()
    xmax_1_list = list()
    ymin_1_list = list()
    ymax_1_list = list()
    image_id_2_list = list()
    entity_2_list = list()
    xmin_2_list = list()
    xmax_2_list = list()
    ymin_2_list = list()
    ymax_2_list = list()
    distance_list = list()
    occlusion_list = list()


    ############################################################################
    fixed_samples = None
    fixed_targets = None
    fixed_energy_list = list()
    count = 0
    stop_index = index_of_that_image

    if TRAIN_ON_ONE_IMAGE:
        for samples, targets in data_loader:
            if count == stop_index:
                fixed_samples = samples
                fixed_targets = targets
                break
            else:
                count += 1
                continue
    ############################################################################


    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):

        ########################################################################
        if TRAIN_ON_ONE_IMAGE and DEBUG_OUTPUTS:
            samples = fixed_samples
            targets = fixed_targets
            print()
            if iteratoin_count == 0:
                print(targets[0]['image_id'])
            print("targets:",
                  targets[0]['human_labels'].cpu().numpy(),
                  targets[0]['action_labels'].cpu().numpy(),
                  targets[0]['object_labels'].cpu().numpy())
        ########################################################################



        ########################################################################
        # Print targets when USE_SMALL_ANNOTATION_FILE
        if not TRAIN_ON_ONE_IMAGE and USE_SMALL_ANNOTATION_FILE and DEBUG_OUTPUTS:
            print("Iteration: ", iteratoin_count)
            print("targets:")

            human_labels = targets[0]['human_labels'].cpu().numpy()
            action_labels = targets[0]['action_labels'].cpu().numpy()
            object_labels = targets[0]['object_labels'].cpu().numpy()

            human_names = list()
            action_names = list()
            object_names = list()

            for i in range(len(human_labels)):
                human_names.append(index_to_name(human_labels[i]))
                action_names.append(distance_id_to_name[action_labels[i]])
                object_names.append(index_to_name(object_labels[i]))

            print('Object A:', human_names)
            print('relation:', action_names)
            print('Object B:', object_names)

        ########################################################################


        original_targets = targets

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items() if k not in ['image_id', 'num_bounding_boxes_in_ground_truth']} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        ########################################################################
        if (TRAIN_ON_ONE_IMAGE or USE_SMALL_ANNOTATION_FILE) and DEBUG_OUTPUTS:
            hoi_list = generate_hoi_list_using_model_outputs(args, outputs, original_targets)
            if len(hoi_list) == 0:
                print("empty hoi_list")
            else:
                try:
                    for i in range(0, top_k_predictions_to_print):
                        print(hoi_list[0]['hoi_list'][i])
                except:
                    pass
            print()
        ########################################################################

        hoi_list = generate_hoi_list_using_model_outputs(args, outputs, original_targets)
        # tp += evaluate_on_training_set(hoi_list, targets)
        # denominator += len(targets) * 2
        # print('tp:           ', tp)
        # print('denominator:  ', denominator)
        # print('precision:    ', tp/denominator)
        # print()
        # print()
        # print()

        # Construct outputs for training set and write to a csv file.
        #  Coordinates are be in normalized xyxy format

        construct_evaluation_output_using_hoi_list(hoi_list,
                                                   original_targets,
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
                                                                   distance_list, occlusion_list)

        iteratoin_count += 1

    df = pd.DataFrame({'image_id_1': image_id_1_list,
                  'entity_1': entity_1_list,
                  'xmin_1': xmin_1_list,
                  'xmax_1': xmax_1_list,
                  'ymin_1': ymin_1_list,
                  'ymax_1': ymax_1_list,
                  'image_id_2': image_id_2_list,
                  'entity_2': entity_2_list,
                  'xmin_2': xmin_2_list,
                  'xmax_2': xmax_2_list,
                  'ymin_2': ymin_2_list,
                  'ymax_2': ymax_2_list,
                  'occlusion': occlusion_list,
                  'distance': distance_list,
                  })


    file_name = 'temp_df'

    file_name = file_name + '_train_epoch_' + str(epoch) + '.csv'
    print(file_name)

    df.to_csv(file_name, index=False)



    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}






def validate(args, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.eval()
    criterion.eval()

    iteratoin_count = 0
    max_num_iterations = len(data_loader)

    image_id_1_list = list()
    entity_1_list = list()
    xmin_1_list = list()
    xmax_1_list = list()
    ymin_1_list = list()
    ymax_1_list = list()
    image_id_2_list = list()
    entity_2_list = list()
    xmin_2_list = list()
    xmax_2_list = list()
    ymin_2_list = list()
    ymax_2_list = list()
    distance_list = list()
    occlusion_list = list()

    for samples, targets in data_loader:

        original_targets = targets
        samples = samples.to(device)
        targets = [
            {k: v.to(device) for k, v in t.items() if k not in ['image_id', 'num_bounding_boxes_in_ground_truth']} for
            t in targets]

        outputs = model(samples)
        hoi_list = generate_hoi_list_using_model_outputs(args, outputs,
                                                         original_targets)
        construct_evaluation_output_using_hoi_list(hoi_list,
                                                   original_targets,
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
                                                   occlusion_list)

        progressBar(iteratoin_count + 1, max_num_iterations, 'Validate Progress')

        iteratoin_count += 1

    df = pd.DataFrame({'image_id_1': image_id_1_list,
                       'entity_1': entity_1_list,
                       'xmin_1': xmin_1_list,
                       'xmax_1': xmax_1_list,
                       'ymin_1': ymin_1_list,
                       'ymax_1': ymax_1_list,
                       'image_id_2': image_id_2_list,
                       'entity_2': entity_2_list,
                       'xmin_2': xmin_2_list,
                       'xmax_2': xmax_2_list,
                       'ymin_2': ymin_2_list,
                       'ymax_2': ymax_2_list,
                       'occlusion': occlusion_list,
                       'distance': distance_list,
                       })

    file_name = 'temp_df'
    file_name = file_name + '_valid_epoch_' + str(epoch) + '.csv'
    print(file_name)
    df.to_csv(file_name, index=False)




    return






