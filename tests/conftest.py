import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# Placeholder fixtures — actual values are injected by _dist_worker via mp.spawn.
# These exist only so pytest can resolve the parameter names during collection.

@pytest.fixture
def rank():
    return 0


@pytest.fixture
def world_size():
    return 2


@pytest.fixture
def device():
    return torch.device("cuda:0")


def _dist_worker(rank, world_size, test_fn, port, extra_kwargs):
    """Per-GPU worker: initializes NCCL/RCCL and runs the test function."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    try:
        test_fn(rank, world_size, device, **extra_kwargs)
    finally:
        dist.destroy_process_group()


def pytest_pyfunc_call(pyfuncitem):
    """Intercept tests that take (rank, world_size, device) and run them
    across multiple GPUs via torch.multiprocessing.spawn with real RCCL."""
    argnames = pyfuncitem._fixtureinfo.argnames
    if not ("rank" in argnames and "world_size" in argnames and "device" in argnames):
        return None

    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        pytest.skip("Need at least 2 GPUs for distributed tests")

    world_size = min(gpu_count, 2)
    test_fn = pyfuncitem.obj

    extra_kwargs = {}
    for name in argnames:
        if name not in ("rank", "world_size", "device"):
            extra_kwargs[name] = pyfuncitem.funcargs[name]

    port = _find_free_port()
    mp.spawn(
        _dist_worker,
        args=(world_size, test_fn, port, extra_kwargs),
        nprocs=world_size,
        join=True,
    )
    return True
