import pickle
from typing import Callable

import torch.distributed as dist
from loguru import logger


# Load dataset in rank 0, and broadcast to other ranks
def build_dataset_distributed(dataset_load_func: Callable, **kwargs):
    if dist.is_initialized():
        rank = dist.get_rank()
        if rank == 0:
            dataset = dataset_load_func(**kwargs)
            serialized_dataset = pickle.dumps(dataset)
            logger.info(f"Rank {rank} loaded dataset size: {len(dataset)}")
        else:
            serialized_dataset = b""

        dist.barrier()

        objects = [serialized_dataset]
        dist.broadcast_object_list(objects, src=0)

        dist.barrier()

        if rank != 0:
            dataset = pickle.loads(objects[0])
            logger.info(f"Rank {rank} received serialized dataset size: {len(dataset)}")

        # dist.barrier()
    else:
        dataset = dataset_load_func(**kwargs)

    return dataset


def run_on_rank0(func: Callable, **kwargs):
    if dist.is_initialized():
        rank = dist.get_rank()
        if rank == 0:
            res = func(**kwargs)
            logger.info(f"{func.__name__} executed at rank 0")
            serialized = pickle.dumps(res)
        else:
            serialized = b""

        dist.barrier()

        objects = [serialized]
        dist.broadcast_object_list(objects, src=0)

        dist.barrier()

        if rank != 0:
            res = pickle.loads(objects[0])
            logger.info(f"result broadcast to rank {rank}")

        # dist.barrier()
    else:
        res = func(**kwargs)

    return res


def is_main_process(arg: str | None = None) -> bool:
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True
