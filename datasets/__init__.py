# ------------------------------------------------------------------------
# Licensed under the Apache License, Version 2.0 (the "License")
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from .hico import build as build_hico
from .hoia import build as build_hoia
from .vcoco import build as build_vcoco
from .two_point_five_vrd import build as build_two_point_five_vrd


def build_dataset(image_set, args, test_scale=-1):
    assert args.dataset_file in ['hico', 'vcoco', 'hoia','two_point_five_vrd'], args.dataset_file
    if args.dataset_file == 'hico':
        return build_hico(image_set, test_scale)
    elif args.dataset_file == 'vcoco':
        return build_vcoco(image_set, test_scale)
    elif args.dataset_file == 'two_point_five_vrd':
        return build_two_point_five_vrd(image_set, test_scale)
    else:
        return build_hoia(image_set, test_scale)
