# ------------------------------------------------------------------------
# SiameseIM
# Copyright (c) SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from MAE (https://github.com/facebookresearch/mae)
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# ------------------------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


class ImagenetWithMask(datasets.ImageFolder):
    def __init__(self,root,dataset=None,
                transform = None,
                with_blockwise_mask=False, ### !!! set to True, enable blockwise masking
                 blockwise_num_masking_patches=75, ### !!! 75 / 196 = 0.38 -> Modify this to increase mask ratio
                 input_size=224, patch_size=16, # no need to change now
                 max_mask_patches_per_block=None, # BEiT default setting, no need to change
                 min_mask_patches_per_block=16, # BEiT default setting, no need to change
                 fixed_num_masking_patches=True, ### set to true, fixed number of masking patch to blockwise_num_masking_patches for sim training 
                 ):
        super().__init__(root, transform)
        if dataset == 'imagenet100':
            cls_list = ['n02869837','n01749939','n02488291','n02107142','n13037406','n02091831','n04517823','n04589890','n03062245','n01773797','n01735189','n07831146','n07753275','n03085013','n04485082','n02105505','n01983481','n02788148','n03530642','n04435653','n02086910','n02859443','n13040303','n03594734','n02085620','n02099849','n01558993','n04493381','n02109047','n04111531','n02877765','n04429376','n02009229','n01978455','n02106550','n01820546','n01692333','n07714571','n02974003','n02114855','n03785016','n03764736','n03775546','n02087046','n07836838','n04099969','n04592741','n03891251','n02701002','n03379051','n02259212','n07715103','n03947888','n04026417','n02326432','n03637318','n01980166','n02113799','n02086240','n03903868','n02483362','n04127249','n02089973','n03017168','n02093428','n02804414','n02396427','n04418357','n02172182','n01729322','n02113978','n03787032','n02089867','n02119022','n03777754','n04238763','n02231487','n03032252','n02138441','n02104029','n03837869','n03494278','n04136333','n03794056','n03492542','n02018207','n04067472','n03930630','n03584829','n02123045','n04229816','n02100583','n03642806','n04336792','n03259280','n02116738','n02108089','n03424325','n01855672','n02090622']
            cls2lbl = {cls: lbl for lbl, cls in enumerate(cls_list)}
            idx = [i for i, lbl in enumerate(self.targets) if self.classes[lbl] in cls2lbl]
            samples = [self.samples[i] for i in idx]
            self.samples = [(fn, cls2lbl[self.classes[lbl]]) for fn, lbl in samples]
            self.targets = [dt[1] for dt in self.samples]
            self.imgs = self.samples
            self.classes = cls_list
            
        self.with_blockwise_mask = with_blockwise_mask
        if with_blockwise_mask:
            from .masking_generator import MaskingGenerator
            window_size = input_size // patch_size
            self.masked_position_generator = MaskingGenerator(
                (window_size, window_size), 
                num_masking_patches=blockwise_num_masking_patches,
                max_num_patches=max_mask_patches_per_block,
                min_num_patches=min_mask_patches_per_block,
                fixed_num_masking_patches=fixed_num_masking_patches
            )
    
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        if self.with_blockwise_mask:
            return sample, target, self.masked_position_generator()
        return sample, target
