import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def is_dist_run():
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup_distributed(args):
    args["distributed"] = False
    args["rank"] = 0
    args["local_rank"] = 0
    args["world_size"] = 1

    if not is_dist_run():
        return

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=args.get("dist_backend", "nccl"))

    args["distributed"] = True
    args["rank"] = rank
    args["local_rank"] = local_rank
    args["world_size"] = world_size


def cleanup_distributed():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not is_dist_avail_and_initialized() or dist.get_rank() == 0


def barrier():
    if is_dist_avail_and_initialized():
        dist.barrier()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def make_sampler(dataset, shuffle):
    if not is_dist_avail_and_initialized():
        return None
    return DistributedSampler(dataset, shuffle=shuffle)


def wrap_model(model, device, args):
    model = model.to(device)
    if not args.get("distributed", False):
        return model

    local_rank = args["local_rank"]
    return DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=args.get("find_unused_parameters", True),
    )


def unwrap_model(model):
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model
