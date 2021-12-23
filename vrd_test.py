"""
This file implements the methods to produce predictions
on the validation and test sets and write them to files
using model checkpoints.
"""

import argparse
import datetime
import getpass
import json
import random
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import util.misc as utils
from datasets import build_dataset
from engine import *
from models import build_model
from magic_numbers import *


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=999999999999, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')

    # Backbone.
    parser.add_argument('--backbone', choices=['resnet50', 'resnet101', 'swin'], required=True,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # Transformer.
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dec_layers_distance', default=6, type=int)
    parser.add_argument('--dec_layers_occlusion', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Loss.
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # Matcher.
    parser.add_argument('--relation_loss_coef', default=1, type=float)
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # Loss coefficients.
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.02, type=float,
                        help="Relative classification weight of the no-object class")

    # Dataset parameters.
    parser.add_argument('--dataset_file', default='two_point_five_vrd', type=str)
    # Modify to your log path ******************************* !!!
    exp_time = datetime.datetime.now().strftime('%Y%m%d%H%M')
    work_dir = 'checkpoint/p_{}'.format(exp_time)

    parser.add_argument('--output_dir', default=work_dir,
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=0, type=int)

    parser.add_argument('--output_name', default='predictions', type=str)

    return parser


def main(args):

    print()
    print("USE_SMALL_VALID_ANNOTATION_FILE: ", USE_SMALL_VALID_ANNOTATION_FILE)
    print("USE_DEPTH_DURING_INFERENCE:      ", USE_DEPTH_DURING_INFERENCE)
    print("PREDICT_INTERSECTION_BOX:        ", PREDICT_INTERSECTION_BOX)
    print("CASCADE:                         ", CASCADE)
    if CASCADE:
        print("dec_layers | dec_layers_distance | dec_layers_occlusion: ")
        print(args.dec_layers, "         |", args.dec_layers_distance, "                  |", args.dec_layers_occlusion)
    print()

    device = torch.device(args.device)

    model, criterion = build_model(args)
    model.to(device)

    model_without_ddp = model

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_test = build_dataset(image_set='test', args=args, test_scale=800)
    sampler_test = torch.utils.data.RandomSampler(dataset_test)
    batch_sampler_test = torch.utils.data.BatchSampler(sampler_test, args.batch_size, drop_last=False)
    data_loader_test = DataLoader(dataset_test,
                                   batch_sampler=batch_sampler_test,
                                   collate_fn=utils.collate_fn,
                                   num_workers=args.num_workers)

    # Load model from checkpoint
    checkpoint = torch.load(args.resume, map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'])
    if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    print("Start Testing")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):

        # Test set
        # Find out the name of the folder in which the predictions will be saved
        start = args.resume.find('/')
        end = args.resume.rfind('/')
        folder_name = args.resume[start+1:end]
        if len(folder_name) == 0:
            folder_name = None

        with torch.no_grad():
            generate_evaluation_outputs(args, 'test', model, criterion, data_loader_test, optimizer,
                     device, epoch, args.clip_max_norm, folder_name=folder_name)

        break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Test time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('HOI Transformer test script', parents=[get_args_parser()])
    args = parser.parse_args()
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)