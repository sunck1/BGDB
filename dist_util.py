import io
import blobfile as bf
import torch as th
import torch.distributed as dist

GPUS_PER_NODE = 8
SETUP_RETRY_COUNT = 3
_GLOBAL_DEVICE = None


def setup_dist():
    global _GLOBAL_DEVICE

    if dist.is_available() and dist.is_initialized():
        _GLOBAL_DEVICE = th.device("cuda" if th.cuda.is_available() else "cpu")
        return

    if th.cuda.is_available():
        _GLOBAL_DEVICE = th.device("cuda:0")
    else:
        _GLOBAL_DEVICE = th.device("cpu")


def dev():
    global _GLOBAL_DEVICE
    if _GLOBAL_DEVICE is not None:
        return _GLOBAL_DEVICE
    return th.device("cuda" if th.cuda.is_available() else "cpu")


def load_state_dict(path, **kwargs):
    map_location = kwargs.pop("map_location", dev())
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    bytestream = io.BytesIO(data)
    return th.load(bytestream, map_location=map_location, **kwargs)


def sync_params(params):
    # dist.broadcast(p, 0) 那套；现在先什么都不做。
    return