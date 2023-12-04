#!/usr/bin/env python
import os
import random
import copy
import warnings
import logging
from pathlib import Path

import hydra
import hydra.utils as hydra_utils
import submitit

os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

MAIN_PID = os.getpid()
SIGNAL_RECEIVED = False

log = logging.getLogger(__name__)


def update_pythonpath_relative_hydra():
    """Update PYTHONPATH to only have absolute paths."""
    # NOTE: We do not change sys.path: we want to update paths for future instantiations
    # of python using the current environment (namely, when submitit loads the job
    # pickle).
    try:
        original_cwd = Path(hydra_utils.get_original_cwd()).resolve()
    except (AttributeError, ValueError):
        # Assume hydra is not initialized, we don't need to do anything.
        # In hydra 0.11, this returns AttributeError; later it will return ValueError
        # https://github.com/facebookresearch/hydra/issues/496
        # I don't know how else to reliably check whether Hydra is initialized.
        return
    paths = []
    for orig_path in os.environ["PYTHONPATH"].split(":"):
        path = Path(orig_path)
        if not path.is_absolute():
            path = original_cwd / path
        paths.append(path.resolve())
    os.environ["PYTHONPATH"] = ":".join([str(x) for x in paths])
    log.info('PYTHONPATH: {}'.format(os.environ["PYTHONPATH"]))


class Worker:
    def __call__(self, args):
        import torch.multiprocessing as mp
        import importlib
        import numpy as np

        mp.set_start_method('spawn')
        main_function = getattr(importlib.import_module(args.worker), 'main')
        args = copy.deepcopy(args)

        np.set_printoptions(precision=3)
        socket_name = os.popen(
            "ip r | grep default | awk '{print $5}'").read().strip('\n')
        print("Setting GLOO and NCCL sockets IFNAME to: {}".format(socket_name))
        os.environ["GLOO_SOCKET_IFNAME"] = socket_name

        job_env = submitit.JobEnvironment()
        args.env.node = job_env.hostnames[0]
        args.env.rank = job_env.global_rank

        # Use random port to avoid collision between parallel jobs
        if args.env.world_size == 1:
            args.env.port = np.random.randint(35565, 50000)
            
        args.env.dist_url = f'tcp://{args.env.node}:{args.env.port}'
        # args.env.dist_url = 'tcp://euler22.engr.wisc.edu:40030'
        print('Using url {}. Global rank {}'.format(args.env.dist_url, args.env.rank))

        if args.env.seed == -1:
            args.env.seed = None

        if args.env.gpu is not None:
            warnings.warn(
                'You have chosen a specific GPU. This will completely '
                'disable data parallelism.')

        # Run code
        main_function(args)

    def checkpoint(self, *args,
                   **kwargs) -> submitit.helpers.DelayedSubmission:
        return submitit.helpers.DelayedSubmission(
            Worker(), *args, **kwargs)  # submits to requeuing


def my_jobs():
    return os.popen('squeue -o %j -u $USER').read().split("\n")


@hydra.main(config_path='configs/', config_name='finetune', version_base='1.1')
def main(args):
    update_pythonpath_relative_hydra()
    args.output_dir = hydra_utils.to_absolute_path(args.output_dir)
    os.makedirs(f"{args.output_dir}/{args.job_name}", exist_ok=True)

    # If job is running, ignore
    job_names = my_jobs()
    slurm_job_name = f"{args.job_name}-{args.env.slurm_suffix}" if args.env.slurm_suffix else args.job_name
    if args.env.slurm and slurm_job_name in job_names:
        print(f'Skipping {args.job_name} because already in queue')
        return

    # If model is trained, ignore
    ckpt_fname = os.path.join(args.log.ckpt_dir, 'checkpoint_{:04d}.pth')
    if os.path.exists(ckpt_fname.format(args.epochs - 1)):
        print(f'Skipping {args.job_name} because already finished training')
        return

    # Submit jobs
    executor = submitit.AutoExecutor(
        folder=args.log.submitit_dir,
        slurm_max_num_timeout=100,
        cluster=None if args.env.slurm else "debug",
    )

    # asks SLURM to send USR1 signal 30 seconds before the time limit
    additional_parameters = {}
    if args.env.nodelist != "":
        additional_parameters.update({"nodelist": args.env.nodelist})
    if args.env.exclude != "":
        additional_parameters.update({"exclude": args.env.exclude})
    executor.update_parameters(
        timeout_min=args.env.slurm_timeout,
        slurm_partition=args.env.slurm_partition,
        cpus_per_task=args.env.workers,
        gpus_per_node=args.env.ngpu,
        nodes=args.env.world_size,
        tasks_per_node=1,
        mem_gb=args.env.mem_gb,
        slurm_additional_parameters=additional_parameters,
        slurm_signal_delay_s=120)
    executor.update_parameters(name=slurm_job_name)
    job = executor.submit(Worker(), args)
    if not args.env.slurm:
        job.result()


if __name__ == '__main__':
    main()
