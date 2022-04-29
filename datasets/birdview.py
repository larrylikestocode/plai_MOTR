# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
MOT dataset which returns image_id for evaluation.
"""
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.utils.data
import os.path as osp
from PIL import Image, ImageDraw
import copy
import datasets.transforms as T
from models.structures import Instances
import pandas as pd

class Birdview:
    def __init__(self, args, data_txt_path: str, seqs_folder, transforms):
        self.args = args
        self._transforms = transforms
        self.num_frames_per_batch = max(args.sampler_lengths)
        self.sample_mode = args.sample_mode
        self.sample_interval = args.sample_interval # 10
        self.vis = args.vis
        self.video_dict = {}
        # self.sampler_steps = None

        with open(data_txt_path, 'r') as file:
            self.img_files = file.readlines()
            self.img_files = [osp.join(seqs_folder, x.split(',')[0].strip()) for x in self.img_files]
            self.img_files = list(filter(lambda x: len(x) > 0, self.img_files))
        self.label_files = [(x.replace('images', 'labels_with_ids').replace('.png', '.csv').replace('.jpg', '.csv'))
                            for x in self.img_files]
        # The number of images per sample: 1 + (num_frames - 1) * interval.
        # The number of valid samples: num_images - num_image_per_sample + 1.
        self.item_num = len(self.img_files) - (self.num_frames_per_batch - 1) * self.sample_interval
        # print("{} item num".format(self.item_num))
        self._register_videos() # TBD label id_offset each video needs unique object id

        # video sampler.
        self.sampler_steps: list = args.sampler_steps
        self.lengths: list = args.sampler_lengths
        # print("sampler_steps={} lenghts={}".format(self.sampler_steps, self.lengths))
        if self.sampler_steps is not None and len(self.sampler_steps) > 0:
            # Enable sampling length adjustment.
            assert len(self.lengths) > 0
            assert len(self.lengths) == len(self.sampler_steps) + 1
            for i in range(len(self.sampler_steps) - 1):
                assert self.sampler_steps[i] < self.sampler_steps[i + 1]
            self.item_num = len(self.img_files) - (self.lengths[-1] - 1) * self.sample_interval
            self.period_idx = 0
            self.num_frames_per_batch = self.lengths[0]
            self.current_epoch = 0

    def _register_videos(self):
        for label_name in self.label_files:
            video_name = label_name.split('/')[-2]# '/'.join(label_name.split('/')[:-1])
            if video_name not in self.video_dict:
                print("register {}-th video: {} ".format(len(self.video_dict) + 1, video_name))
                self.video_dict[video_name] = len(self.video_dict)
                # assert len(self.video_dict) <= 300 # TBD


    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if self.sampler_steps is None or len(self.sampler_steps) == 0:
            # fixed sampling length.
            return

        for i in range(len(self.sampler_steps)):
            if epoch >= self.sampler_steps[i]:
                self.period_idx = i + 1
        print("set epoch: epoch {} period_idx={}".format(epoch, self.period_idx))
        self.num_frames_per_batch = self.lengths[self.period_idx]

    def step_epoch(self):
        # one epoch finishes.
        print("Dataset: epoch {} finishes".format(self.current_epoch))
        self.set_epoch(self.current_epoch + 1)

    @staticmethod
    def _targets_to_instances(targets: dict, img_shape) -> Instances:
        gt_instances = Instances(tuple(img_shape))
        gt_instances.boxes = targets['boxes']
        gt_instances.labels = targets['labels']
        gt_instances.obj_ids = targets['obj_ids']
        gt_instances.area = targets['area']
        return gt_instances

    def _pre_single_frame(self, idx: int):
        img_path = self.img_files[idx]
        # print("img_path: {}".format(img_path))
        label_path = self.label_files[idx]
        img = Image.open(img_path)
        targets = {}
        w, h = img._size
        # print("w {} h {}".format(w, h))
        assert w > 0 and h > 0, "invalid image {} with shape {} {}".format(img_path, w, h)
        if osp.isfile(label_path):
            # labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)
            try:
                labels0 = pd.read_csv(label_path)
                xyxy = labels0[[ 'frame_id','agent_i','bbox_l_x', 'bbox_l_y', 'bbox_r_x', 'bbox_r_y']].to_numpy()
            except:
                print(label_path)
                raise Exception("Sorry, no numbers below zero")
                # print(xyxy.shape) # 27,4
            labels = xyxy.copy()
            labels[:, 2] = labels[:, 2] + 256.
            labels[:, 3] = labels[:, 3] + 256.
            labels[:, 4] = labels[:, 4] + 256.
            labels[:, 5] = labels[:, 5] + 256.
            """
            output the min max edge of the bb
            """
            # normalized cewh to pixel xyxy format
            '''
            labels = labels0.copy()
            labels[:, 2] = w * (labels0[:, 2] - labels0[:, 4] / 2)
            labels[:, 3] = h * (labels0[:, 3] - labels0[:, 5] / 2)
            labels[:, 4] = w * (labels0[:, 2] + labels0[:, 4] / 2)
            labels[:, 5] = h * (labels0[:, 3] + labels0[:, 5] / 2)
            '''
        else:
            raise ValueError('invalid label path: {}'.format(label_path))
        video_name = label_path.split('/')[-2]
        # print(video_name)
        obj_idx_offset = self.video_dict[video_name] * 100000# *100000#   # 100000 unique ids is enough for a video.
        targets['boxes'] = []
        targets['area'] = []
        targets['iscrowd'] = []
        targets['labels'] = []
        targets['obj_ids'] = []
        targets['image_id'] = torch.as_tensor(idx)
        targets['size'] = torch.as_tensor([h, w])
        targets['orig_size'] = torch.as_tensor([h, w])
        for label in labels:
            targets['boxes'].append(label[2:6].tolist())
            # targets['area'].append(label[4] * label[5])
            targets['area'].append(label[4] * label[5])
            targets['iscrowd'].append(0)
            targets['labels'].append(0)
            obj_id = label[1] + obj_idx_offset if label[1] >= 0 else label[1]
            targets['obj_ids'].append(obj_id)  # relative id

        targets['area'] = torch.as_tensor(targets['area'])
        targets['iscrowd'] = torch.as_tensor(targets['iscrowd'])
        targets['labels'] = torch.as_tensor(targets['labels'])
        targets['obj_ids'] = torch.as_tensor(targets['obj_ids'])
        targets['boxes'] = torch.as_tensor(targets['boxes'], dtype=torch.float32).reshape(-1, 4)
        targets['boxes'][:, 0::2].clamp_(min=0, max=w)
        targets['boxes'][:, 1::2].clamp_(min=0, max=h)
        return img, targets

    def _get_sample_range(self, start_idx):

        # take default sampling method for normal dataset.
        assert self.sample_mode in ['fixed_interval', 'random_interval'], 'invalid sample mode: {}'.format(self.sample_mode)
        if self.sample_mode == 'fixed_interval':
            sample_interval = self.sample_interval
        elif self.sample_mode == 'random_interval':
            sample_interval = np.random.randint(1, self.sample_interval + 1)
        default_range = start_idx, start_idx + (self.num_frames_per_batch - 1) * sample_interval + 1, sample_interval
        return default_range

    def pre_continuous_frames(self, start, end, interval=1):
        targets = []
        images = []
        # print("start {} end {} interval {}".format(start, end, interval))
        for i in range(start, end, interval):

            # print(start)
            # print(end)
            # print(interval)
            img_i, targets_i = self._pre_single_frame(i)
            images.append(img_i)
            targets.append(targets_i)
        return images, targets

    def __getitem__(self, idx):
        ### 30 20gt+10pred
        sample_start, sample_end, sample_interval = self._get_sample_range(idx)

        # print(sample_start)

        images, targets = self.pre_continuous_frames(sample_start, sample_end, sample_interval)
        data = {}
        if self._transforms is not None:
            images, targets = self._transforms(images, targets)
        gt_instances = []
        for img_i, targets_i in zip(images, targets):
            gt_instances_i = self._targets_to_instances(targets_i, img_i.shape[1:3])
            gt_instances.append(gt_instances_i)
        data.update({
            'imgs': images,
            'gt_instances': gt_instances,
        })
        if self.args.vis:
            data['ori_img'] = [target_i['ori_img'] for target_i in targets]
        return data

    def __len__(self):
        return self.item_num


class BirdviewValidation(Birdview):
    def __init__(self, args, seqs_folder, transforms):
        args.data_txt_path = args.val_data_txt_path
        super().__init__(args, seqs_folder, transforms)


def make_detmot_transforms(image_set, args=None):
    normalize = T.MotCompose([
        T.MotToTensor(),
        T.MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if image_set == 'train':
        color_transforms = []
        scale_transforms = [
            normalize,
        ]

        return T.MotCompose(color_transforms + scale_transforms)

    if image_set == 'val':
        return T.MotCompose([
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.mot_path)
    assert root.exists(), f'provided MOT path {root} does not exist'
    transforms = make_detmot_transforms(image_set, args)
    if image_set == 'train':
        data_txt_path = args.data_txt_path_train
        dataset = Birdview(args, data_txt_path=data_txt_path, seqs_folder=root, transforms=transforms)
    if image_set == 'val':
        data_txt_path = args.data_txt_path_val
        dataset = Birdview(args, data_txt_path=data_txt_path, seqs_folder=root, transforms=transforms)
    return dataset
