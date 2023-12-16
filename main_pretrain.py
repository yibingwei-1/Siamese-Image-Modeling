# ------------------------------------------------------------------------
# SiameseIM
# Copyright (c) SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from MAE (https://github.com/facebookresearch/mae)
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# ------------------------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
# from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm
# assert timm.__version__ == "0.6.12"  # version check 0.9
from timm.optim.optim_factory import param_groups_weight_decay
from timm.optim import create_optimizer

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.augmentation import RandomResizedCrop, GaussianBlur, SingleRandomResizedCrop, RandomHorizontalFlip, Solarize
from util.datasets import ImagenetWithMask
import models_sim
from engine_pretrain import train_one_epoch

import random
import warnings
import torch.multiprocessing as mp
import datasets as myDBs

import wandb

class DataAugmentationForSIM(object):
    def __init__(self, args):
        self.args = args

        self.random_resized_crop = SingleRandomResizedCrop(args.input_size, scale=(args.crop_min, 1.0), interpolation=3)
        self.random_flip = RandomHorizontalFlip()
        # moco color aug
        self.color_transform1 = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
        ])

        self.color_transform2 = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
            transforms.RandomApply([Solarize()], p=0.2),
        ])

        self.format_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, image):
        spatial_image1, flip1 = self.random_flip(image)
        spatial_image2, flip2 = self.random_flip(image)
        spatial_image1, i1, j1, h1, w1, W = self.random_resized_crop(spatial_image1)
        spatial_image2, i2, j2, h2, w2, W = self.random_resized_crop(spatial_image2)
        color_image1 = self.color_transform1(spatial_image1)
        color_image2 = self.color_transform2(spatial_image2)

        relative_flip = (flip1 and not flip2) or (flip2 and not flip1)
        return self.format_transform(color_image1), self.format_transform(color_image2), \
                (i2-i1)/h1, (j2-j1)/w1, h2/h1, w2/w1, relative_flip, (W-j1-j2)/w1

    def __repr__(self):
        repr = "(DataAugmentation,\n"
        repr += "  transform = %s,\n" % str(self.random_resized_crop) + str(self.random_flip) + str(self.color_transform1) + str(self.format_transform)
        repr += ")"
        return repr


def main(args):
    if args.env.seed is not None:
        seed = args.env.seed + misc.get_rank()
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    ngpus_per_node = torch.cuda.device_count()
    args.env.distributed = args.env.world_size > 1 or (args.env.distributed and ngpus_per_node > 1)
    if args.env.distributed:
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(args,))
    else:
        main_worker(0, args)

def main_worker(local_rank, args):
    misc.init_distributed_mode(local_rank, args) # need change to torch.engine

    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
    job_dir = f"{args.output_dir}/{args.job_name}"
    print(f'job dir: {job_dir}')
    print("{}".format(args).replace(', ', ',\n'))
    
    num_tasks = misc.get_world_size()
    num_tasks_per_node = max(1, torch.cuda.device_count())
    global_rank = misc.get_rank()
    args.env.workers = args.env.workers // num_tasks_per_node
    eff_batch_size = args.batch_size * args.accum_iter * num_tasks
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    cudnn.benchmark = True
    
    # disable tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # build augmentation and dataset
    if args.loss_type in ['sim']:
        transform_train = DataAugmentationForSIM(args)
    else:
        transform_train = transforms.Compose([
                transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    if not args.use_tcs_dataset:
        dataset_train = myDBs.load_dataset(args.dataset, args.data_path, transform=transform_train, train=True)
        dataset_train = ImagenetWithMask(os.path.join(args.data_path, 'train'),
                                         transform=transform_train,
                                         with_blockwise_mask=args.with_blockwise_mask,
                                        blockwise_num_masking_patches=args.blockwise_num_masking_patches)
    else: # for internal use only
        from util.tcs_datasets import ImagenetTCSDataset
        dataset_train = ImagenetTCSDataset('train',
                                        's3://imagenet',
                                        use_tcs=True,
                                        transform=transform_train,
                                        with_blockwise_mask=args.with_blockwise_mask,
                                        blockwise_num_masking_patches=args.blockwise_num_masking_patches,
                                        local_rank=int(os.environ['LOCAL_RANK']),
                                        local_size=int(os.environ['LOCAL_SIZE']),
                                        tcs_conf_path='./petreloss.conf')
    print(dataset_train)
    
    eval_img_size = 224
    db_eval = myDBs.load_dataset(
        args.dataset, args.data_path,
        transform=transforms.Compose([
            transforms.Resize(int(eval_img_size / 0.875), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(eval_img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]), train=False)


    if args.env.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        sampler_eval = torch.utils.data.DistributedSampler(
            db_eval, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_eval = torch.utils.data.RandomSampler(db_eval)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.env.workers,
        pin_memory=args.env.pin_mem,
        drop_last=True,
        persistent_workers=True
    )
    data_loader_eval = torch.utils.data.DataLoader(
        db_eval,
        sampler=sampler_eval,
        batch_size=args.batch_size,
        num_workers=args.env.workers,
        pin_memory=True,
        drop_last=True,
    )
    
    # build model
    model = models_sim.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, args=args)
    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    if args.env.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.env.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # build optimizer
    # following timm: set wd as 0 for bias and norm layers
    param_groups = param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, args.beta2))
    print(optimizer)
    loss_scaler = NativeScaler(enabled=(not args.fp32), growth_interval=args.amp_growth_interval)

    # Checkpointing
    modules = {
        'state_dict': model,
        'optimizer': optimizer,
        'loss_scaler': loss_scaler,
    }
    ckpt_manager = misc.CheckpointManager(
        modules=modules,
        ckpt_dir=f"{job_dir}/checkpoints",
        epochs=args.epochs,
        save_freq=args.log.save_freq)
    
    if args.resume:
        args.start_epoch = ckpt_manager.resume()

    if args.log.use_wandb and args.env.rank == 0:
        misc.init_wandb(args, job_dir, entity=args.log.wandb_entity, project=args.log.wandb_project, job_name=args.job_name)
        
    if args.knn_eval_only:
        epoch = args.start_epoch if args.start_epoch else 0
        misc.eval_knn(data_loader_eval, model, epoch, args=args, device=device)
        return

    # start training
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        if args.env.distributed:
            data_loader_train.sampler.set_epoch(epoch)
            
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args=args
        )
        dist.barrier()
        
        # knn eval
        global_step = (epoch + 1) * len(data_loader_train)
        if epoch % args.eval_freq == 0 or epoch == args.epochs-1 or epoch == args.start_epoch:
            nn_acc = misc.eval_knn(data_loader_eval, model, epoch, args=args, device=device)
            if args.log.use_wandb and args.env.rank == 0:
                wandb.log({'NN Acc': nn_acc}, step=global_step)

        # save ckpt
        
        # save checkpoint
        ckpt_manager.checkpoint(epoch+1, {'epoch': epoch+1})
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch,}
        if misc.is_main_process():
            with open(os.path.join(job_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
                
        # if args.output_dir and ((epoch+1) % args.save_freq == 0 or epoch + 1 == args.epochs):
        #     misc.save_model(
        #         args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
        #         loss_scaler=loss_scaler, epoch=epoch)
        # if (epoch+1) % args.save_latest_freq == 0:
        #     misc.save_model(
        #             args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
        #             loss_scaler=loss_scaler, epoch=epoch, latest=True)

        # # log information
        # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
        #                 'epoch': epoch,}

        # if args.output_dir and misc.is_main_process():
        #     with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
        #         f.write(json.dumps(log_stats) + "\n")
                
        if misc.is_main_process():
            epoch_total_time = time.time() - epoch_start_time
            now = datetime.datetime.today()
            eta = now + datetime.timedelta(seconds=(args.epochs-epoch-1)*int(epoch_total_time))
            next_50_ep = ((epoch + 1) // 50 + 1) * 50
            eta_to_next_50 =now + datetime.timedelta(seconds=(next_50_ep - epoch - 1) * int(epoch_total_time))
            print(f"ETA to {args.epochs:4d}ep:\t{str(eta)}")
            print(f"ETA to {next_50_ep:4d}ep:\t{str(eta_to_next_50)}")
        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


# if __name__ == '__main__':
#     args = get_args_parser()
#     args = args.parse_args()
#     if args.output_dir:
#         Path(args.output_dir).mkdir(parents=True, exist_ok=True)
#     main(args)
