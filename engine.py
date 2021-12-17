# ------------------------------------------------------------------------
# Licensed under the Apache License, Version 2.0 (the "License")
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train functions used in main.py
"""
import gc
import math
import sys
from typing import Iterable
import torch

import util
import util.misc as utils
from magic_numbers import *
from process_model_outputs import *
from evaluation import *
import pandas as pd
from util.misc import is_main_process
import torch.distributed as dist
from models.position_encoding import build_position_encoding
from torch import nn
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)


def progressBar(i, max, text):
    """
    Print a progress bar during training.
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


def train_one_epoch(args, writer, model: torch.nn.Module, criterion: torch.nn.Module, optimal_transport: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    use_optimal_transport=False, lr_scheduler=None):
    """
    Train the model for one epoch.
    """
    model.train()
    criterion.train()
    iteratoin_count = 0
    max_num_iterations = len(data_loader)


    epoch_loss = 0

    objects_loss_ce_unscaled = 0
    action_loss_ce_unscaled = 0
    occlusion_loss_ce_unscaled = 0
    loss_bbox_unscaled = 0
    loss_giou_unscaled = 0

    ############################################################################
    # (For debugging purpose)
    # Select a specific image from the training set to repeatedly train
    # on that image if TRAIN_ON_ONE_IMAGE is set to True
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

    # Specify the stats to be recorded by the metric_logger
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.10f}'))
    metric_logger.add_meter('class_error_action', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('class_error_occlusion', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, depth, targets in metric_logger.log_every(data_loader, print_freq, header):

        ########################################################################
        # (For debugging purpose)
        # Print the image_id and class labels for that specific image
        # selected earlier above
        if TRAIN_ON_ONE_IMAGE and DEBUG_OUTPUTS:
            samples = fixed_samples
            targets = fixed_targets
            print()
            if iteratoin_count == 0:
                print(targets[0]['image_id'])
            print("targets:",
                  targets[0]['human_labels'].cpu().numpy(),
                  targets[0]['object_labels'].cpu().numpy(),
                  targets[0]['action_labels'].cpu().numpy(),
                  targets[0]['occlusion_labels'].cpu().numpy())
        ########################################################################
        ########################################################################
        # (For debugging purpose)
        # Print targets when training on the small dataset
        # instead of a single image if DEBUG_OUTPUTS is set to true
        if not TRAIN_ON_ONE_IMAGE and USE_SMALL_ANNOTATION_FILE and DEBUG_OUTPUTS:
            print("Iteration: ", iteratoin_count)
            print("targets:")
            human_labels = targets[0]['human_labels'].cpu().numpy()
            action_labels = targets[0]['action_labels'].cpu().numpy()
            object_labels = targets[0]['object_labels'].cpu().numpy()
            occlusion_labels = targets[0]['occlusion_labels'].cpu().numpy()
            human_names = list()
            action_names = list()
            object_names = list()
            occlusion_names = list()
            for i in range(len(human_labels)):
                human_names.append(index_to_name(human_labels[i]))
                action_names.append(distance_id_to_name[action_labels[i]])
                object_names.append(index_to_name(object_labels[i]))
                occlusion_names.append(occlusion_id_to_name[occlusion_labels[i]])
            print('Object A: ', human_names)
            print('Object B: ', object_names)
            print('relation: ', action_names)
            print('occlusion:', occlusion_names)
        ########################################################################

        # save a copy of targets which preserve
        # image_id and num_bounding_boxes_in_ground_truth
        original_targets = targets

        # move tensors in the samples and targets to GPU and abandon objects
        # that cannot be moved to GPU
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items() if k not in ['image_id', 'num_bounding_boxes_in_ground_truth']} for t in targets]

        # Prepare depth or delete depth
        if USE_DEPTH_DURING_TRAINING:
            depth.tensors = depth.tensors[:, 0:1]
            depth = depth.to(device)
            with torch.no_grad():
                if isinstance(depth, (list, torch.Tensor)):
                    depth = nested_tensor_from_tensor_list(depth)
                m = nn.MaxPool2d(32, stride=32, ceil_mode=True)
                depth.tensors = m(depth.tensors)
                depth.mask = (m(depth.mask.type(torch.float))).type(torch.bool)
                PE = build_position_encoding(args)
                pos_depth = PE(depth)
        else:
            pos_depth = None
            del depth.tensors
            del depth.mask
            del depth
            gc.collect()

        # Forward pass
        outputs = model(samples, pos_depth=pos_depth, writer=writer)

        loss_dict = criterion(outputs, targets, optimal_transport=optimal_transport)
        weight_dict = criterion.weight_dict

        # Sum up weighted losses in the loss dictionary
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)

        # Losses that are not weighted by the weight dict
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}

        # Record unscaled losses
        objects_loss_ce_unscaled += loss_dict_reduced_unscaled['human_loss_ce_unscaled'] + loss_dict_reduced_unscaled['object_loss_ce_unscaled']
        action_loss_ce_unscaled += loss_dict_reduced_unscaled['action_loss_ce_unscaled']
        occlusion_loss_ce_unscaled += loss_dict_reduced_unscaled['occlusion_loss_ce_unscaled']
        loss_bbox_unscaled += loss_dict_reduced_unscaled['loss_bbox_unscaled']
        loss_giou_unscaled += loss_dict_reduced_unscaled['loss_giou_unscaled']


        # Losses that are weighted by the weighted dict
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}

        # Compute a single loss value by summing up the reduced and weighted losses
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        # Raise error if the loss is infinite
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # zero_grad the optimizer and back-prop
        optimizer.zero_grad()
        losses.backward()

        # Clip the gradients
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # Step optimizer
        optimizer.step()

        # Record training stats using metric_logger
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        try:
            metric_logger.update(class_error_action=loss_dict_reduced['class_error_action'])
            metric_logger.update(class_error_occlusion=loss_dict_reduced['class_error_occlusion'])
        except:
            metric_logger.update(class_error_action=-1)
            metric_logger.update(class_error_occlusion=-1)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        ########################################################################
        # (For debugging purpose)
        # Print non-empty hoi lists if training on a single image or
        # a small dataset and DEBUG_OUTPUTS is set to True
        if (TRAIN_ON_ONE_IMAGE or USE_SMALL_ANNOTATION_FILE) and DEBUG_OUTPUTS:
            hoi_list = generate_hoi_list_using_model_outputs(args, outputs, original_targets, filter=True)
            if len(hoi_list[0]['hoi_list']) == 0:
                print("empty hoi_list")
            else:
                try:
                    for i in range(0, top_k_predictions_to_print):
                        print(hoi_list[0]['hoi_list'][i])
                except:
                    pass
            print()
        ########################################################################

        iteratoin_count += 1

        if LR_RANGE_TEST:
            writer.add_scalar('Misc_train/lr', optimizer.param_groups[0]["lr"], iteratoin_count)
            writer.add_scalar('Loss_train_unscaled/1_ce_objects', loss_dict_reduced_unscaled['human_loss_ce_unscaled'] + loss_dict_reduced_unscaled['object_loss_ce_unscaled'], iteratoin_count)
            writer.add_scalar('Loss_train_unscaled/2_ce_distance', loss_dict_reduced_unscaled['action_loss_ce_unscaled'], iteratoin_count)
            writer.add_scalar('Loss_train_unscaled/3_ce_occlusion', loss_dict_reduced_unscaled['occlusion_loss_ce_unscaled'], iteratoin_count)
            writer.add_scalar('Loss_train_unscaled/4_reg_bbox', loss_dict_reduced_unscaled['loss_bbox_unscaled'], iteratoin_count)
            writer.add_scalar('Loss_train_unscaled/5_reg_giou', loss_dict_reduced_unscaled['loss_giou_unscaled'], iteratoin_count)
            # writer.add_scalar('Misc_train/error_distance', class_error_action, iteratoin_count)
            # writer.add_scalar('Misc_train/error_occlusion', train_stats['class_error_occlusion'], iteratoin_count)
            lr_scheduler.step()

    # gather stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # Write loss and lr to tensorboard at the end of each epoch
    writer.add_scalar('Loss_train_unscaled/1_ce_objects', objects_loss_ce_unscaled / len(data_loader), epoch)
    writer.add_scalar('Loss_train_unscaled/2_ce_distance', action_loss_ce_unscaled / len(data_loader), epoch)
    writer.add_scalar('Loss_train_unscaled/3_ce_occlusion', action_loss_ce_unscaled / len(data_loader), epoch)
    writer.add_scalar('Loss_train_unscaled/4_reg_bbox', loss_bbox_unscaled / len(data_loader), epoch)
    writer.add_scalar('Loss_train_unscaled/5_reg_giou', loss_giou_unscaled / len(data_loader), epoch)
    writer.add_scalar('Misc_train/lr', train_stats['lr'], epoch)
    writer.add_scalar('Misc_train/error_distance', train_stats['class_error_action'], epoch)
    writer.add_scalar('Misc_train/error_occlusion', train_stats['class_error_occlusion'], epoch)


    torch.cuda.empty_cache()
    return train_stats



def validate(args, writer, valid_or_test, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    """
    Compute losses on the validation or test set.
    :param valid_or_test: 'valid' or 'test'
    :param criterion: Hungarian matcher
    :param data_loader: data loader for the validation or test set
    """
    model.eval()
    criterion.eval()

    iteratoin_count = 0
    max_num_iterations = len(data_loader)

    loss = 0
    loss_ce = 0
    loss_bbox = 0
    loss_giou = 0

    objects_loss_ce_unscaled = 0
    action_loss_ce_unscaled = 0
    occlusion_loss_ce_unscaled = 0
    loss_bbox_unscaled = 0
    loss_giou_unscaled = 0
    error_distance_unscaled = 0
    error_occlusion_unscaled = 0

    for samples, depth, targets in data_loader:

        samples = samples.to(device)
        targets = [
            {k: v.to(device) for k, v in t.items() if k not in ['image_id', 'num_bounding_boxes_in_ground_truth']} for
            t in targets]

        # Prepare depth or delete depth
        if USE_DEPTH_DURING_INFERENCE:
            depth.tensors = depth.tensors[:, 0:1]
            depth = depth.to(device)
            with torch.no_grad():
                if isinstance(depth, (list, torch.Tensor)):
                    depth = nested_tensor_from_tensor_list(depth)
                PE = build_position_encoding(args)
                m = nn.MaxPool2d(32, stride=32, ceil_mode=True)
                depth.tensors = m(depth.tensors)
                depth.mask = (m(depth.mask.type(torch.float))).type(torch.bool)
                pos_depth = PE(depth)
        else:
            pos_depth = None
            del depth.tensors
            del depth.mask
            del depth
            gc.collect()

        # Forward pass
        outputs = model(samples, pos_depth)

        # Compute Losses
        loss_dict = criterion(outputs, targets, training=False)

        # Get the weighted dict to weight different losses
        weight_dict = criterion.weight_dict

        # Reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}

        # Record unscaled losses
        objects_loss_ce_unscaled += loss_dict_reduced_unscaled['human_loss_ce_unscaled'] + loss_dict_reduced_unscaled['object_loss_ce_unscaled']
        action_loss_ce_unscaled += loss_dict_reduced_unscaled['action_loss_ce_unscaled']
        occlusion_loss_ce_unscaled += loss_dict_reduced_unscaled['occlusion_loss_ce_unscaled']
        loss_bbox_unscaled += loss_dict_reduced_unscaled['loss_bbox_unscaled']
        loss_giou_unscaled += loss_dict_reduced_unscaled['loss_giou_unscaled']
        error_distance_unscaled += loss_dict_reduced_unscaled['class_error_action_unscaled']
        error_occlusion_unscaled += loss_dict_reduced_unscaled['class_error_occlusion_unscaled']

        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()
        loss += loss_value
        loss_ce += loss_dict_reduced_scaled['loss_ce']
        loss_bbox += loss_dict_reduced_scaled['loss_bbox']
        loss_giou += loss_dict_reduced_scaled['loss_giou']

        # Print a progress bar to show validation progress
        if utils.get_rank() == 0:
            progressBar(iteratoin_count + 1, max_num_iterations, valid_or_test + ' progress    ')

        iteratoin_count += 1

    # Record Losses to tensorboard
    writer.add_scalar('Loss_valid_unscaled/1_ce_objects', objects_loss_ce_unscaled / len(data_loader), epoch)
    writer.add_scalar('Loss_valid_unscaled/2_ce_distance', action_loss_ce_unscaled / len(data_loader), epoch)
    writer.add_scalar('Loss_valid_unscaled/3_ce_occlusion', action_loss_ce_unscaled / len(data_loader), epoch)
    writer.add_scalar('Loss_valid_unscaled/4_reg_bbox', loss_bbox_unscaled / len(data_loader), epoch)
    writer.add_scalar('Loss_valid_unscaled/5_reg_giou', loss_giou_unscaled / len(data_loader), epoch)
    writer.add_scalar('Misc_valid/error_distance', error_distance_unscaled/ iteratoin_count, epoch)
    writer.add_scalar('Misc_valid/error_occlusion', error_occlusion_unscaled / iteratoin_count, epoch)

    torch.cuda.empty_cache()
    return



def generate_evaluation_outputs(args, valid_or_test, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, folder_name=None):
    """
    Generate predictions on the validation or test set. Results will be saved to
        a file named predictions_[valid_or_test]_[epoch_number].csv.
    :param valid_or_test: Generate predictions for which dataset.
        "valid" for the validation set. "test" for the test set.
    """

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

    for samples, depth, targets in data_loader:

        original_targets = targets
        samples = samples.to(device)

        # Prepare depth or delete depth
        if USE_DEPTH_DURING_INFERENCE:
            depth.tensors = depth.tensors[:, 0:1]
            depth = depth.to(device)
            with torch.no_grad():
                if isinstance(depth, (list, torch.Tensor)):
                    depth = nested_tensor_from_tensor_list(depth)
                PE = build_position_encoding(args)
                m = nn.MaxPool2d(32, stride=32, ceil_mode=True)
                depth.tensors = m(depth.tensors)
                depth.mask = (m(depth.mask.type(torch.float))).type(torch.bool)
                pos_depth = PE(depth)
        else:
            pos_depth = None
            del depth.tensors
            del depth.mask
            del depth
            gc.collect()

        # Forward pass
        outputs = model(samples, pos_depth)

        # Construct Evaluation Outputs
        hoi_list = generate_hoi_list_using_model_outputs(args, outputs, original_targets, filter=True)
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

        # Print a progress bar
        progressBar(iteratoin_count + 1, max_num_iterations, valid_or_test + ' progress    ')
        iteratoin_count += 1
        break

    # Save Evaluation Outputs to a DataFrame
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
    # Make sure the data type for each column is correct
    df.astype({'image_id_1': 'str',
               'entity_1': 'str',
               'xmin_1': 'float',
               'xmax_1': 'float',
               'ymin_1': 'float',
               'ymax_1': 'float',
               'image_id_2': 'str',
               'entity_2': 'str',
               'xmin_2': 'float',
               'xmax_2': 'float',
               'ymin_2': 'float',
               'ymax_2': 'float',
               'occlusion': 'int',
               'distance': 'int'})

    # Write DataFrame to file
    file_name = args.output_name
    if folder_name:
        file_name = folder_name + '/' + file_name
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

    file_name = file_name + '_' + valid_or_test + '_' + str(epoch-1) + '.csv'
    print(file_name)
    df.to_csv(file_name, index=False)



