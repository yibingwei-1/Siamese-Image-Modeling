# ------------------------------------------------------------------------
# SiameseIM
# Copyright (c) SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from MAE (https://github.com/facebookresearch/mae)
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# References:
# MoCo v3: https://github.com/facebookresearch/moco-v3
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
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

# assert timm.__version__ == "0.6.12" # version check
from timm.models.layers import trunc_normal_

import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.lars import LARS
from util.crop import RandomResizedCrop
import models_vit
from engine_finetune import train_one_epoch, evaluate

import wandb
import random
import warnings
import torch.multiprocessing as mp
import datasets as myDBs

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
    assert (ngpus_per_node>1 and args.env.distributed) or (ngpus_per_node<=1 and not args.env.distributed), f"distributed inconsistency error, ngpus_per_node {ngpus_per_node}, args.env.distributed {args.env.distributed}"
    
    if args.env.distributed:
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(args,))
    else:
        main_worker(0, args)


def main_worker(local_rank, args):
    misc.init_distributed_mode(local_rank,args)

    job_dir = f"{args.output_dir}/{args.job_name}"
    if job_dir and misc.is_main_process():
        Path(job_dir).mkdir(parents=True, exist_ok=True)
    print(f'job dir: {job_dir}')
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')

    num_tasks = misc.get_world_size()
    num_tasks_per_node = max(1, torch.cuda.device_count())
    global_rank = misc.get_rank()
    args.env.workers = args.env.workers // num_tasks_per_node
    eff_batch_size = args.batch_size * args.accum_iter * num_tasks
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)
    
    cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # linear probe: weak augmentation
    dataset_train = myDBs.load_dataset(
        args.dataset, args.data_path,
        transform=transforms.Compose([
          transforms.RandomResizedCrop(size=224, interpolation=transforms.InterpolationMode.BICUBIC),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]), train=True)
    dataset_val = myDBs.load_dataset(
        args.dataset, args.data_path,
        transform=transforms.Compose([
            transforms.Resize(int(224 / 0.875), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]), train=False)
    print(dataset_train)
    print(dataset_val)
    print(dataset_train)
    print(dataset_val)

    # build dataloader
    if args.env.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.env.workers,
        pin_memory=args.env.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.env.workers,
        pin_memory=args.env.pin_mem,
        drop_last=False
    )

    # build model
    args.nb_classes = 100 if args.dataset == "imagenet100" else 1000
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        global_pool=args.global_pool,
        init_values=args.init_values if args.init_values != 1.0 else None,
        drop_path_rate=0.0
    )

    # load ckpt
    if args.pretrain_job_name:
        pretrain_ckpt = os.path.join(args.output_dir,args.pretrain_job_name,"checkpoints",f"checkpoint_{args.pretrain_resume_epoch}.pth") #default: checkpoint_latest.pth
        checkpoint = torch.load(pretrain_ckpt, map_location='cpu') 
        print("Load pre-trained checkpoint from: %s" % args.pretrain_job_name)
        checkpoint_model = checkpoint['model']

        state_dict = model.state_dict()
        
        for k in list(checkpoint_model.keys()):
            if k.startswith('module.'):
                checkpoint_model[k[len("module."):]] = checkpoint_model[k]
                del checkpoint_model[k]
        
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        if args.global_pool:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer: following MoCo v3
        trunc_normal_(model.head.weight, std=0.01)

    # for linear prob only
    # hack: revise model's head with BN
    # model.bn = torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6)
    model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
    # freeze all but the head
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.head.named_parameters():
        p.requires_grad = True

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    if args.env.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        model_without_ddp = model.module

    # build optimizer
    optimizer = LARS(model_without_ddp.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(optimizer)
    loss_scaler = NativeScaler()

    criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    # Checkpointing
    modules = {
        'model': model,
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
        
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)
    
    if args.log.use_wandb and args.env.rank == 0:
        misc.init_wandb(args, job_dir, entity=args.log.wandb_entity, project=args.log.wandb_project, job_name=args.job_name)
        

    # start training
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.env.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            max_norm=None,
            args=args
        )

        if epoch%args.log.eval_freq==0 or epoch==args.epochs-1 or epoch==args.start_epoch:
            test_stats = evaluate(data_loader_val, model, device)
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            max_accuracy = max(max_accuracy, test_stats["acc1"])
            print(f'Max accuracy: {max_accuracy:.2f}%')

            if args.log.use_wandb and args.env.rank == 0:
                global_step = (epoch + 1) * len(data_loader_train)
                wandb.log({'Acc1': test_stats["acc1"]}, step=global_step)
            
        ckpt_manager.checkpoint(epoch+1, {'Acc1': test_stats["acc1"], 'epoch': epoch+1})

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


# if __name__ == '__main__':
#     args = get_args_parser()
#     args = args.parse_args()
#     if args.output_dir:
#         Path(args.output_dir).mkdir(parents=True, exist_ok=True)
#     main(args)
