"""Compare complete native DCP model and optimizer tensors after resume."""
import argparse
import json
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import TensorStorageMetadata


def load(path):
    # Megatron writes flat DCP keys without planner_data. The generic format
    # converter assumes PyTorch's nested planner metadata and cannot read it.
    reader = dcp.FileSystemReader(path)
    metadata = reader.read_metadata()
    state = {key: torch.empty(item.size, dtype=item.properties.dtype)
             if isinstance(item, TensorStorageMetadata) else None
             for key, item in metadata.state_dict_metadata.items()}
    dcp.load(state, storage_reader=reader,
             planner=DefaultLoadPlanner(flatten_state_dict=False, flatten_sharded_tensors=False))
    return state


def flatten(obj, prefix=''):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from flatten(value, f'{prefix}.{key}')
    elif isinstance(obj, (list, tuple)):
        for key, value in enumerate(obj):
            yield from flatten(value, f'{prefix}.{key}')
    elif torch.is_tensor(obj):
        yield prefix, obj


def compare_common(a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            compare_common(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            compare_common(x, y)
    else:
        assert a == b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('uninterrupted'); parser.add_argument('resumed')
    parser.add_argument('--iteration', type=int, default=3)
    args = parser.parse_args()
    a, b = dict(flatten(load(args.uninterrupted))), dict(flatten(load(args.resumed)))
    assert a and a.keys() == b.keys(), 'Checkpoint tensor structure differs'
    error = 0.
    optimizer_tensors = 0
    for key, tensor in a.items():
        torch.testing.assert_close(tensor, b[key], rtol=1e-5, atol=1e-6, msg=lambda message: f'{key}\n{message}')
        error = max(error, float((tensor.float() - b[key].float()).abs().max()) if tensor.numel() else 0.)
        optimizer_tensors += int('optim' in key)
    assert optimizer_tensors, 'No optimizer tensors compared'
    # Optimizer group step counters and LR scheduler are also common-state fields.
    common_a = torch.load(Path(args.uninterrupted) / 'common.pt', weights_only=False, map_location='cpu')
    common_b = torch.load(Path(args.resumed) / 'common.pt', weights_only=False, map_location='cpu')
    assert common_a['iteration'] == common_b['iteration'] == args.iteration
    compare_common(common_a['opt_param_scheduler'], common_b['opt_param_scheduler'])
    compare_common(common_a['optimizer'], common_b['optimizer'])
    assert common_a['args'].consumed_train_samples == common_b['args'].consumed_train_samples
    print(json.dumps({'native_resume_tensors': len(a), 'optimizer_tensors': optimizer_tensors,
                     'max_absolute_error': error, 'iteration': args.iteration, 'scheduler_equal': True,
                     'consumed_samples_equal': True}), flush=True)


if __name__ == '__main__':
    main()
