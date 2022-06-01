from torchvision.datasets.vision import VisionDataset
import torchvision
import torch
import numpy as np
import json
import cv2
import random
import PIL
import torchvision.transforms as T
import torchvision.transforms.functional as F
from util.box_ops import box_xyxy_to_cxcywh
from PIL import Image
from magic_numbers import *

import pandas as pd
import gc

# This partially addresses the EOF Error
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy(sharing_strategy)

# Load class_descriptions_boxable
class_descriptions_boxable = pd.read_csv('data/2.5vrd/class-descriptions-boxable.csv', header=None)
# Insert a row at the top
class_descriptions_boxable.loc[-1] = ['','']
class_descriptions_boxable.index = class_descriptions_boxable.index + 1
class_descriptions_boxable.sort_index(inplace=True)

# Load Object Annotations
within_image_objects_train = pd.read_csv('data/2.5vrd/within_image_objects_train.csv')
within_image_objects_validation = pd.read_csv('data/2.5vrd/within_image_objects_validation.csv')
within_image_objects_test = pd.read_csv('data/2.5vrd/within_image_objects_test.csv')
across_images_objects_train = pd.read_csv('data/2.5vrd/across_images_objects_train.csv')
across_images_objects_validation = pd.read_csv('data/2.5vrd/across_images_objects_validation.csv')
across_images_objects_test = pd.read_csv('data/2.5vrd/across_images_objects_test.csv')

# Load VRD Annotations
within_image_vrd_test = pd.read_csv('data/2.5vrd/within_image_vrd_test.csv')
within_image_vrd_train = pd.read_csv('data/2.5vrd/within_image_vrd_train.csv')
within_image_vrd_validation = pd.read_csv('data/2.5vrd/within_image_vrd_validation.csv')
across_images_vrd_test = pd.read_csv('data/2.5vrd/across_images_vrd_test.csv')
across_images_vrd_train = pd.read_csv('data/2.5vrd/across_images_vrd_train.csv')
across_images_vrd_validation = pd.read_csv('data/2.5vrd/across_images_vrd_validation.csv')


def entity_to_name(entity):
    return class_descriptions_boxable[class_descriptions_boxable[0] == entity][1].item()


def name_to_entity(name):
    return class_descriptions_boxable[class_descriptions_boxable[1] == name][0].item()

def entity_to_index(entity):
    """
    find the index of the entity in the the DataFrame class_descriptions_boxable.
    If the entity is not in the DataFrame, return -1
    """
    index = -1
    try:
        index = class_descriptions_boxable[class_descriptions_boxable[0] == entity].index.item()
    except:
        pass
    return index

def name_to_index(name):
    """
    find the index of the name in the the DataFrame class_descriptions_boxable.
    If the name is not in the DataFrame, return -1
    """
    index = -1
    try:
        index = class_descriptions_boxable[class_descriptions_boxable[1] == name].index.item()
    except:
        pass
    return index

def index_to_name(index):
    return class_descriptions_boxable.iloc[index][1]

# TODO: process no majority
distance_id_to_name = {
    #-1: 'no majority',
    0: 'not sure',
    1: 'A is closer',
    2: 'B is closer',
    3: 'same distance'
}

distance_name_to_id = {
    'no majority': int(0),
    'not sure': int(0),
    'A is closer': int(1),
    'B is closer': int(2),
    'same distance': int(3)
}


occlusion_id_to_name = {
    # -1: 'no majority',
    0: 'no occlusion',
    1: 'A occludes B',
    2: 'B occludes A',
    3: 'mutually occluded'
}

occlusion_name_to_id = {
    'no majority': int(0),
    'no occlusion': int(0),
    'A occludes B': int(1),
    'B occludes A': int(2),
    'mutually occluded': int(3)
}

def convert_xywh2x1y1x2y2(box, shape, flip):
    ih, iw = shape[:2]
    x, y, w, h = box
    if flip == 1:
        x1_org = x
        x2_org = x + w - 1
        x2 = iw - 1 - x1_org
        x1 = iw - 1 - x2_org
    else:
        x1 = x
        x2 = x + w - 1
    x1 = max(x1, 0)
    x2 = min(x2, iw-1)
    y1 = max(y, 0)
    y2 = min(y + h - 1, ih-1)
    return [x1, y1, x2, y2]


def get_det_annotation_from_odgt(item, shape, flip, gt_size_min=1):
    total_boxes, gt_boxes, ignored_boxes = [], [], []
    for annot in item['gtboxes']:
        box = convert_xywh2x1y1x2y2(annot['box'], shape, flip)
        x1, y1, x2, y2 = box
        cls_id = name_to_index(annot['tag'])
        total_boxes.append([x1, y1, x2, y2, cls_id, ])
        # if name of object is not in the DataFrame, skip it
        if cls_id == -1:
            continue
        if annot.get('extra', {}).get('ignore', 0) == 1:
            ignored_boxes.append(box)
            continue
        if (x2 - x1 + 1) * (y2 - y1 + 1) < gt_size_min ** 2:
            ignored_boxes.append(box)
            continue
        if x2 <= x1 or y2 <= y1:
            ignored_boxes.append(box)
            continue
        gt_boxes.append([x1, y1, x2, y2, cls_id, ])
    return gt_boxes, ignored_boxes, total_boxes


def get_interaction_box(human_box, object_box, hoi_id, occlusion_id):
    hx1, hy1, hx2, hy2, hid = human_box
    ox1, oy1, ox2, oy2, oid = object_box
    # hcx, hcy = (hx1 + hx2) / 2, (hy1 + hy2) / 2
    # ocx, ocy = (ox1 + ox2) / 2, (oy1 + oy2) / 2
    # dx = (hcx - ocx) / 5
    # dy = (hcy - ocy) / 5
    # xx1, yy1, xx2, yy2 = list(map(int, [ox1 + dx, oy1 + dy, ox2 + dx, oy2 + dy]))
    xx1, yy1, xx2, yy2 = min(hx1, ox1), min(hy1, oy1), max(hx2, ox2), max(hy2, oy2)
    return [xx1, yy1, xx2, yy2, hoi_id, occlusion_id]


def xyxy_to_cxcywh(box):
    x0, y0, x1, y1, cid = box
    return [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0), cid]


def get_hoi_annotation_from_odgt(item, total_boxes, scale):
    human_boxes, object_boxes, action_boxes = [], [], []
    human_labels, object_labels, action_labels, occlusion_labels, raw_distance_labels, raw_occlusion_labels = [], [], [], [], [], []
    img_hh, img_ww = item['height'], item['width']
    for hoi in item.get('hoi', []):
        x1, y1, x2, y2, cls_id = list(map(int, total_boxes[int(hoi['subject_id'])]))
        human_box = x1 // scale, y1 // scale, x2 // scale, y2 // scale, cls_id
        if cls_id == -1 or x1 >= x2 or y1 >= y2:
            continue
        x1, y1, x2, y2, cls_id = list(map(int, total_boxes[int(hoi['object_id'])]))
        object_box = x1 // scale, y1 // scale, x2 // scale, y2 // scale, cls_id
        if cls_id == -1 or x1 >= x2 or y1 >= y2:
            continue
        hoi_id = distance_name_to_id[hoi['distance']]
        occlusion_id = occlusion_name_to_id[hoi['occlusion']]
        hoi_box = get_interaction_box(human_box=human_box, object_box=object_box, hoi_id=hoi_id, occlusion_id=occlusion_id)

        human_boxes.append(human_box[0:4])
        object_boxes.append(object_box[0:4])
        action_boxes.append(hoi_box[0:4])
        human_labels.append(human_box[4])
        object_labels.append(object_box[4])
        action_labels.append(hoi_box[4])
        occlusion_labels.append(hoi_box[5])
        unprocessed_raw_distance_labels = np.array([int(x) for x in hoi['raw_distance'].split(',')])
        unprocessed_raw_occlusion_labels = np.array([int(x) for x in hoi['raw_occlusion'].split(',')])

        # Convert raw distance to probability
        values, counts = np.unique(unprocessed_raw_distance_labels, return_counts=True)
        frequencies = counts / counts.sum()
        expanded_frequency = np.zeros(5, dtype=np.float16)
        for j in range(len(values)):
            current_value = values[j]
            current_frequency = frequencies[j]
            expanded_frequency[current_value] = current_frequency
        raw_distance_labels.append(expanded_frequency)

        # Convert raw occlusion to probability
        values, counts = np.unique(unprocessed_raw_occlusion_labels, return_counts=True)
        frequencies = counts / counts.sum()
        expanded_frequency = np.zeros(5, dtype=np.float16)
        for j in range(len(values)):
            current_value = values[j]
            current_frequency = frequencies[j]
            expanded_frequency[current_value] = current_frequency
        raw_occlusion_labels.append(expanded_frequency)

    return dict(
        human_boxes=torch.from_numpy(np.array(human_boxes).astype(np.float32)),
        human_labels=torch.from_numpy(np.array(human_labels)),
        object_boxes=torch.from_numpy(np.array(object_boxes).astype(np.float32)),
        object_labels=torch.from_numpy(np.array(object_labels)),
        action_boxes=torch.from_numpy(np.array(action_boxes).astype(np.float32)),
        action_labels=torch.from_numpy(np.array(action_labels)),
        occlusion_labels=torch.from_numpy(np.array(occlusion_labels)),
        image_id=item['file_name'],
        org_size=torch.as_tensor([int(img_hh), int(img_ww)]),
        raw_distance_labels=torch.from_numpy(np.array(raw_distance_labels)),
        raw_occlusion_labels=torch.from_numpy(np.array(raw_occlusion_labels))
    )


def parse_one_gt_line(gt_line, scale=1):
    item = json.loads(gt_line)
    img_name = item['file_name']
    img_shape = item['height'], item['width']
    gt_boxes, ignored_boxes, total_boxes = get_det_annotation_from_odgt(item, img_shape, flip=0)
    interaction_boxes = get_hoi_annotation_from_odgt(item, total_boxes, scale)

    num_bounding_boxes_in_ground_truth = len(gt_boxes)
    interaction_boxes['num_bounding_boxes_in_ground_truth'] = num_bounding_boxes_in_ground_truth

    return dict(image_id=img_name, annotations=interaction_boxes)


def hflip(image, depth, target):
    flipped_image = F.hflip(image)
    flipped_depth = F.hflip(depth)
    w, h = image.size
    target = target.copy()
    if "human_boxes" in target:
        boxes = target["human_boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["human_boxes"] = boxes
    if "object_boxes" in target:
        boxes = target["object_boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["object_boxes"] = boxes
    if "action_boxes" in target:
        boxes = target["action_boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["action_boxes"] = boxes
    return flipped_image, flipped_depth, target


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, depth, target):
        if random.random() < self.p:
            return hflip(img, depth, target)
        return img, depth, target


class RandomAdjustImage(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, depth, target):
        if random.random() < self.p:
            img = F.adjust_brightness(img, random.choice([0.8, 0.9, 1.0, 1.1, 1.2]))
            #depth = F.adjust_brightness(depth, random.choice([0.8, 0.9, 1.0, 1.1, 1.2]))
        if random.random() < self.p:
            img = F.adjust_contrast(img, random.choice([0.8, 0.9, 1.0, 1.1, 1.2]))
            #depth = F.adjust_contrast(depth, random.choice([0.8, 0.9, 1.0, 1.1, 1.2]))
        return img, depth, target


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, depth, target):
        if random.random() < self.p:
            return self.transforms1(img, depth, target)
        return self.transforms2(img, depth, target)


def resize(image, depth, target, size, max_size=None):
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))
        if (w <= h and w == size) or (h <= w and h == size):
            return h, w
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        return oh, ow

    rescale_size = get_size_with_aspect_ratio(image_size=image.size, size=size, max_size=max_size)
    rescaled_image = F.resize(image, rescale_size)
    rescaled_depth = F.resize(depth, rescale_size)

    if target is None:
        return rescaled_image, rescaled_depth, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "human_boxes" in target:
        boxes = target["human_boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["human_boxes"] = scaled_boxes
    if "object_boxes" in target:
        boxes = target["object_boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["object_boxes"] = scaled_boxes
    if "action_boxes" in target:
        boxes = target["action_boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["action_boxes"] = scaled_boxes

    return rescaled_image, rescaled_depth, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, depth, target=None):
        size = random.choice(self.sizes)
        return resize(img, depth, target, size, self.max_size)

# TODO: it does not work for depth
def crop(image, depth, org_target, region):
    raise NotImplementedError("crop() does not work for depth")
    cropped_depth = None
    cropped_image = F.crop(image, *region)
    target = org_target.copy()
    i, j, h, w = region
    fields = ["human_labels", "object_labels", "action_labels"]

    if "human_boxes" in target:
        boxes = target["human_boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        target["human_boxes"] = cropped_boxes.reshape(-1, 4)
        fields.append("human_boxes")
    if "object_boxes" in target:
        boxes = target["object_boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        target["object_boxes"] = cropped_boxes.reshape(-1, 4)
        fields.append("object_boxes")
    if "action_boxes" in target:
        boxes = target["action_boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        target["action_boxes"] = cropped_boxes.reshape(-1, 4)
        fields.append("action_boxes")

    # remove elements for which the boxes or masks that have zero area
    if "human_boxes" in target and "object_boxes" in target:
        cropped_boxes = target['human_boxes'].reshape(-1, 2, 2)
        keep1 = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        cropped_boxes = target['object_boxes'].reshape(-1, 2, 2)
        keep2 = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        keep = keep1 * keep2
        if keep.any().sum() == 0:
            return image, org_target
        for field in fields:
            target[field] = target[field][keep]
    return cropped_image, cropped_depth, target


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, depth: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, (h, w))
        raise NotImplementedError("RandomSizeCrop is not implemented")
        return crop(img, depth, target, region)


class ToTensor(object):
    def __call__(self, img, depth, target):
        return torchvision.transforms.functional.to_tensor(img), torchvision.transforms.functional.to_tensor(depth), target


class Normalize(object):
    def __init__(self, mean, std, depth_mean, depth_std):
        self.mean = mean
        self.std = std
        self.depth_mean = depth_mean
        self.depth_std = depth_std


    def __call__(self, image, depth, target):
        image = torchvision.transforms.functional.normalize(image, mean=self.mean, std=self.std)
        depth = torchvision.transforms.functional.normalize(depth, mean=self.depth_mean, std=self.depth_std)
        if target is None:
            return image, depth, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "human_boxes" in target:
            boxes = target["human_boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["human_boxes"] = boxes
        if "object_boxes" in target:
            boxes = target["object_boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["object_boxes"] = boxes
        if "action_boxes" in target:
            boxes = target["action_boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["action_boxes"] = boxes
        return image, depth, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, depth, target):
        for t in self.transforms:
            image, depth, target = t(image, depth, target)
        return image, depth, target

def make_hico_transforms(image_set, test_scale=-1):
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    if GPU_MEMORY_PRESSURE_TEST:
        scales = [800]
    # mean and std for OIDv4 training set
    mean = [0.38582161319756497, 0.417059363143913, 0.44746641122649666]
    std = [0.2928927708221023, 0.28587472243230755, 0.2924566717392719]
    # mean and std for depth of training set
    depth_mean = [0.42352728300018017, 0.42352728300018017, 0.42352728300018017]
    depth_std = [0.29530982498913205, 0.29530982498913205, 0.29530982498913205]
    normalize = Compose([
        ToTensor(),
        Normalize(mean, std, depth_mean, depth_std),
    ])
    if image_set == 'train' and not DEACTIVATE_EXTRA_TRANSFORMS:
        if not GPU_MEMORY_PRESSURE_TEST:
            return Compose([
                RandomHorizontalFlip(),
                RandomAdjustImage(),
                RandomSelect(
                    RandomResize(scales, max_size=1333),
                    Compose([
                        RandomResize([400, 500, 600]),
                        RandomResize(scales, max_size=1333),
                    ])
                ),
                normalize,
            ])
        else:
            return Compose([
                RandomHorizontalFlip(),
                RandomAdjustImage(),
                RandomSelect(
                    RandomResize(scales, max_size=1333),
                    RandomResize(scales, max_size=1333)
                ),
                normalize,
            ])
    if image_set == 'test' or image_set == 'valid' or DEACTIVATE_EXTRA_TRANSFORMS:
        if test_scale == -1:
            return Compose([
                normalize,
            ])
        assert 400 <= test_scale <= 800, test_scale
        return Compose([
            RandomResize([test_scale], max_size=1333),
            normalize,
        ])
    raise ValueError(f'unknown {image_set}')


class two_point_five_VRD(VisionDataset):
    def __init__(self, root, annFile, image_set, transform=None, target_transform=None, transforms=None):
        super(two_point_five_VRD, self).__init__(root, transforms, transform, target_transform)
        with open(annFile, 'r') as f:
            self.annotations = [parse_one_gt_line(l.strip()) for l in f.readlines()]
        self.transforms = transforms
        self.image_set = image_set
        self.image_folder_name = self.image_set
        if self.image_set == 'valid':
            self.image_folder_name = 'validation'

    def __getitem__(self, index):
        ann = self.annotations[index]
        img_name = ann['image_id']
        target = ann['annotations']
        if not CUSTOM_TSET_SET:
            img_path = './data/2.5vrd/images/' + self.image_folder_name + '/' + img_name
        else:
            img_path = './data/2.5vrd/images/' + 'custom' + '/' + img_name

        try:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = Image.fromarray(img[:, :, ::-1]).convert('RGB')
        except:
            print(img_path)
            raise NotImplementedError("Image not found")

        depth_name = img_name[:-3] + 'png'
        depth_path = './data/2.5vrd/depth/' + self.image_folder_name + '/' + depth_name
        try:
            depth = cv2.imread(depth_path, cv2.IMREAD_COLOR)
            depth = Image.fromarray(depth[:, :, ::-1]).convert('RGB')
        except:
            if not USE_DEPTH_DURING_TRAINING and not USE_DEPTH_DURING_INFERENCE:
                depth = img.copy()

        # Save img and depth to temp for visualization and debugging
        if SAVE_IMAGES:
            img.save('temp/' + img_name[:-4] + '_img.png')
            depth.save('temp/' + img_name[:-4] + '_depth.png')

        # before transform, boxes are in xyxy
        # after transform (nomalize), boxes are in cxcywh
        if self.transforms is not None:
            img, depth, target = self.transforms(img, depth, target)

        # Save transformed img and depth
        if SAVE_IMAGES:
            transformed_img = Image.fromarray((img.permute(1,2,0).numpy() * 255).astype(np.uint8))
            transformed_img.save('temp/' + img_name[:-4] + '_img_transformed.png')

            transformed_depth = Image.fromarray((depth.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            transformed_depth.save('temp/' + img_name[:-4] + '_depth_transformed.png')

        assert img.shape == depth.shape

        # Put items in target into arrays to partially address the
        # EOF Error when num_workers > 1
        human_boxes = target['human_boxes']
        human_labels = target['human_labels']
        object_boxes = target['object_boxes']
        object_labels = target['object_labels']
        action_boxes = target['action_boxes']
        action_labels = target['action_labels']
        occlusion_labels = target['occlusion_labels']
        raw_distance_labels = target['raw_distance_labels']
        raw_occlusion_labels = target['raw_occlusion_labels']
        image_id = target['image_id']
        org_size = target['org_size']
        num_bounding_boxes_in_ground_truth = target['num_bounding_boxes_in_ground_truth']

        # Compute interaction boxes
        # (human boxes and object boxes are in the cxcywh format)
        xmin = torch.max(human_boxes[:, 0] - human_boxes[:, 2] / 2, object_boxes[:, 0] - object_boxes[:, 2] / 2)
        ymin = torch.max(human_boxes[:, 1] - human_boxes[:, 3] / 2, object_boxes[:, 1] - object_boxes[:, 3] / 2)
        xmax = torch.min(human_boxes[:, 0] + human_boxes[:, 2] / 2, object_boxes[:, 0] + object_boxes[:, 2] / 2)
        ymax = torch.min(human_boxes[:, 1] + human_boxes[:, 3] / 2, object_boxes[:, 1] + object_boxes[:, 3] / 2)
        # address negative width and height by swapping min and max
        xmin_adjusted = torch.min(xmin, xmax)
        xmax_adjusted = torch.max(xmin, xmax)
        ymin_adjusted = torch.min(ymin, ymax)
        ymax_adjusted = torch.max(ymin, ymax)
        w = xmax_adjusted - xmin_adjusted
        h = ymax_adjusted - ymin_adjusted
        cx = (xmin_adjusted + xmax_adjusted) / 2
        cy = (ymin_adjusted + ymax_adjusted) / 2
        # Randomly shift the location (center) of the ground truth intersection box
        if RANDOMLY_SHIFT_GT_INTERSECTION_BOXES:
            random_noise_scale_x, random_noise_scale_y = np.random.normal(loc=RAND_INTER_LOC, scale=RAND_INTER_SCALE, size=2)
            cx += w * random_noise_scale_x
            cy += h * random_noise_scale_y
        # Randomly adjust the size of the ground truth intersection box
        if RANDOMLY_ADJUST_SIZES_OF_GT_INTERSECTION_BOXES:
            random_size_scale_x, random_size_scale_y = np.random.normal(loc=RAND_INTER_SIZE_LOC, scale=RAND_INTER_SIZE_SCALE, size=2)
            w += w * random_size_scale_x
            h += h * random_size_scale_y
            # make sure the adjusted width and height are greater than zero
            w = torch.max(0.00001 * torch.tensor(w.shape, device=w.device), w)
            h = torch.max(0.00001 * torch.tensor(h.shape, device=h.device), h)
        # If no intersection exists, set w and h to -1.
        # Losses for intersection box will not be back-proped if
        # w or h is -1 (<0).
        if DO_NOT_PREDICT_INTERSECTION_BOX_IF_NO_INTERSECTION:
            if (xmin > xmax).any() or (ymin > ymax).any():
                w = w * 0 - 1
                h = h * 0 - 1
        # this intersection box should also be in the cxcywh format
        intersection_boxes = torch.vstack([cx, cy, w, h]).T


        image_id = np.array(image_id)
        num_bounding_boxes_in_ground_truth = torch.tensor(num_bounding_boxes_in_ground_truth)

        target.clear()
        del target
        gc.collect()

        return img, depth, human_boxes, human_labels, object_boxes, object_labels, action_boxes, action_labels, occlusion_labels, raw_distance_labels, raw_occlusion_labels, image_id, org_size, num_bounding_boxes_in_ground_truth, intersection_boxes

    def __len__(self):
        return len(self.annotations)

# Task: 'distance' + 'occlusion'
def build(image_set, test_scale=-1):

    if image_set == 'train':
        annotation_file = './data/2.5vrd/annotation_train_combined.odgt'
        if USE_SMALL_ANNOTATION_FILE:
            annotation_file = './data/2.5vrd/' + small_annotation_file
    elif image_set == 'valid':
        annotation_file = './data/2.5vrd/annotation_valid_combined.odgt'
        if USE_SMALL_VALID_ANNOTATION_FILE:
            annotation_file = './data/2.5vrd/' + small_valid_annotation_file
    elif image_set == 'test':
        annotation_file = './data/2.5vrd/annotation_test_combined.odgt'
        if USE_SMALL_TEST_ANNOTATION_FILE:
            annotation_file = './data/2.5vrd/' + small_test_annotation_file
    else:
        raise Exception()
    dataset = two_point_five_VRD(root='./data/2.5vrd',
                                 annFile=annotation_file,
                                 image_set = image_set,
                                 transforms=make_hico_transforms(image_set, test_scale))
    return dataset















