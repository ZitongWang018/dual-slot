"""Unmodified upstream GPT pretraining with an optional Dual Slot adapter."""
import atexit
import os
import hashlib
import json
import uuid
import sys
from pathlib import Path

UPSTREAM = Path(__file__).resolve().parent / "vendor" / "Megatron-LM"
if not (UPSTREAM / "pretrain_gpt.py").is_file():
    raise RuntimeError("Initialize Megatron-LM with git submodule update --init")
sys.path.insert(0, str(UPSTREAM))

import pretrain_gpt as upstream
from megatron.training import get_args, get_tokenizer, pretrain, print_rank_0
from megatron.training import global_vars, checkpointing
from megatron.core.enums import ModelType
from dual_slot.native import core_range, enable_dual_slot, prepare_shared_weight_gradients

_NATIVE_CHECKPOINT_MANAGER = None


def install_native_transport():
    """Use Linux abstract IPC sockets for native checkpoint and loader queues.

    A filesystem UNIX socket on NFS cannot be unlinked while the manager's
    server still holds it open. The queue and checkpoint writer stay upstream.
    """
    from multiprocessing import get_context, util, connection
    from multiprocessing.managers import SyncManager
    import megatron.core.dist_checkpointing.strategies.filesystem_async as filesystem
    if not util.abstract_sockets_supported:
        return
    original_address = connection.arbitrary_address

    def ipc_address(family):
        # Tensor storage FD exchange in forked DataLoader workers otherwise
        # creates filesystem sockets under TMPDIR on the same NFS mount.
        if family == 'AF_UNIX':
            return f'\0dual-slot-fd-{os.getuid()}-{os.getpid()}-{uuid.uuid4().hex}'
        return original_address(family)

    connection.arbitrary_address = ipc_address

    def results_queue():
        global _NATIVE_CHECKPOINT_MANAGER
        if filesystem._results_queue is None:
            address = f'\0dual-slot-{os.getuid()}-{os.getpid()}-{uuid.uuid4().hex}'
            manager = SyncManager(address=address, ctx=get_context('spawn'))
            manager.start()
            filesystem._results_queue = manager.Queue()
            _NATIVE_CHECKPOINT_MANAGER = manager
        return filesystem._results_queue

    filesystem._get_write_results_queue = results_queue


def extra_arguments(parser):
    group = parser.add_argument_group("Dual Slot")
    group.add_argument("--dual-slot", action="store_true")
    group.add_argument("--dual-slot-core-start", type=int)
    group.add_argument("--dual-slot-core-end", type=int)
    group.add_argument("--eod-id", type=int)
    group.add_argument("--swanlab", action="store_true")
    group.add_argument("--swanlab-workspace", default=None)
    group.add_argument("--swanlab-project", default="dual-slot")
    group.add_argument("--run-name", default="native-dual-slot")
    group.add_argument("--audit-native", action="store_true")
    return parser


def model_provider(*args, **kwargs):
    model = upstream.model_provider(*args, **kwargs)
    options = get_args()
    # Use identical, explicit main-gradient accumulation semantics in both arms.
    prepare_shared_weight_gradients(model)
    if options.dual_slot:
        start, end = core_range(options.num_layers, options.dual_slot_core_start, options.dual_slot_core_end)
        enable_dual_slot(model, start=start, end=end, eod_id=get_tokenizer().eod,
                         seed=options.seed, iteration_provider=lambda: getattr(get_args(), 'curr_iteration', get_args().iteration))
        model.decoder.dual_slot_audit = options.audit_native
    elif options.dual_slot_core_start is not None or options.dual_slot_core_end is not None:
        raise ValueError("Core bounds require --dual-slot")
    print_rank_0(f"MODEL_PARAMETERS={sum(p.numel() for p in model.parameters())}; DUAL_SLOT={options.dual_slot}; ATTENTION={options.attention_backend}")
    if options.audit_native:
        print_rank_0(json.dumps({"native_model": True, "params": sum(p.numel() for p in model.parameters()),
            "dtype": str(next(model.parameters()).dtype),
            "residual_init_std": model.config.output_layer_init_method.keywords["std"],
            "recompute": model.config.recompute_granularity,
            "disable_parameter_transpose_cache": model.config.disable_parameter_transpose_cache,
            "shared_embeddings": model.share_embeddings_and_output_weights}))
    return model


def install_metadata():
    original = global_vars.build_tokenizer

    def build_tokenizer(options):
        tokenizer = original(options)
        if options.eod_id is not None:
            if options.tokenizer_type != "NullTokenizer" or not 0 <= options.eod_id < tokenizer.vocab_size:
                raise ValueError("EOD override requires NullTokenizer and a valid token ID")
            tokenizer._eod_id = options.eod_id
        return tokenizer

    global_vars.build_tokenizer = build_tokenizer
    original_check = checkpointing.check_checkpoint_args

    def check(saved):
        options = get_args()
        if bool(getattr(saved, "dual_slot", False)) != options.dual_slot:
            raise ValueError("Checkpoint method differs; use --finetune for weight initialization")
        if options.dual_slot:
            previous = core_range(saved.num_layers, getattr(saved, "dual_slot_core_start", None), getattr(saved, "dual_slot_core_end", None))
            current = core_range(options.num_layers, options.dual_slot_core_start, options.dual_slot_core_end)
            if previous != current or getattr(saved, "eod_id", None) != options.eod_id:
                raise ValueError("Checkpoint core/EOD differs")
        if not options.skip_train:
            for key in ('global_batch_size', 'micro_batch_size', 'seq_length', 'train_iters',
                        'lr', 'min_lr', 'lr_warmup_iters', 'lr_decay_iters', 'weight_decay',
                        'adam_beta1', 'adam_beta2', 'adam_eps', 'clip_grad', 'seed',
                        'train_data_path', 'valid_data_path', 'eod_id'):
                if getattr(saved, key, None) != getattr(options, key, None):
                    raise ValueError(f"Checkpoint training protocol differs: {key}")
        original_check(saved)

    checkpointing.check_checkpoint_args = check
    # DataLoader iterator construction draws worker base seeds from global CPU
    # RNG even with zero workers. It must not advance restored model RNG merely
    # because a job restarted. Dataset sampling/indices remain entirely upstream.
    import torch
    import megatron.training.training as training_module
    original_iterators = training_module.build_train_valid_test_data_iterators

    def build_iterators(*args, **kwargs):
        rng = torch.get_rng_state()
        try:
            return original_iterators(*args, **kwargs)
        finally:
            torch.set_rng_state(rng)

    training_module.build_train_valid_test_data_iterators = build_iterators


def install_tracking():
    # Megatron already writes loss, LR, gradient norm, validation and throughput
    # through this writer on the final rank. Mirror those metrics to SwanLab.
    original = global_vars._set_tensorboard_writer

    def set_writer(options):
        original(options)
        writer = global_vars._GLOBAL_TENSORBOARD_WRITER
        if not options.swanlab or writer is None:
            return
        import swanlab
        config_keys = ("num_layers", "hidden_size", "ffn_hidden_size", "num_attention_heads",
            "num_query_groups", "seq_length", "micro_batch_size", "global_batch_size",
            "train_iters", "lr", "min_lr", "lr_warmup_iters", "weight_decay", "seed",
            "dual_slot", "untie_embeddings_and_output_weights", "attention_backend",
            "use_distributed_optimizer", "eod_id")
        run_id_file = Path(options.save) / 'swanlab_run_id'
        run_id_file.parent.mkdir(parents=True, exist_ok=True)
        run_id = run_id_file.read_text().strip() if run_id_file.is_file() else uuid.uuid4().hex[:8]
        run_id_file.write_text(run_id)
        config = {key: getattr(options, key, None) for key in config_keys}
        config['attention_backend'] = str(config['attention_backend'])
        swanlab.init(project=options.swanlab_project, workspace=options.swanlab_workspace,
                     name=options.run_name, config=config, id=run_id, resume='allow',
                     log_dir=os.path.join(options.save, "swanlog"))
        original_scalar = writer.add_scalar

        def add_scalar(tag, value, global_step=None, *args, **kwargs):
            original_scalar(tag, value, global_step, *args, **kwargs)
            if "vs samples" not in tag:
                swanlab.log({tag: float(value)}, step=global_step)

        writer.add_scalar = add_scalar
        atexit.register(swanlab.finish)

    global_vars._set_tensorboard_writer = set_writer


def install_audit():
    # Observe the actual upstream batch and optimizer groups without replacing them.
    original_batch = upstream.get_batch
    seen = set()

    def get_batch(*args, **kwargs):
        batch = tuple(original_batch(*args, **kwargs))
        iteration = getattr(get_args(), 'curr_iteration', get_args().iteration)
        if get_args().audit_native and iteration not in seen:
            seen.add(iteration)
            digest = hashlib.sha256()
            for value in batch:
                if value is not None:
                    digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            print(json.dumps({"native_batch": True, "rank": get_args().rank,
                "iteration": iteration,
                "sha256": digest.hexdigest(), "shape": list(batch[0].shape),
                "attention_mask_is_none": batch[3] is None}), flush=True)
        return batch

    upstream.get_batch = get_batch
    import megatron.core.optimizer as optimizer_module
    original_groups = optimizer_module._get_param_groups

    def groups(*args, **kwargs):
        result = original_groups(*args, **kwargs)
        if get_args().audit_native:
            for group in result:
                for param in group['params']:
                    if param.ndim == 1 and group['wd_mult'] != 0:
                        raise AssertionError("A normalization parameter received weight decay")
            print_rank_0(json.dumps({"native_optimizer_groups": [
                {"wd_mult": g['wd_mult'], "parameters": sum(p.numel() for p in g['params'])}
                for g in result]}))
        return result

    optimizer_module._get_param_groups = groups


def finish_native():
    """Close worker pools and distributed resources before interpreter shutdown."""
    import torch
    from multiprocessing import resource_sharer
    from torch._inductor.async_compile import shutdown_compile_workers
    torch.cuda.synchronize()
    shutdown_compile_workers()
    resource_sharer.stop()
    if _NATIVE_CHECKPOINT_MANAGER is not None:
        _NATIVE_CHECKPOINT_MANAGER.shutdown()
    writer = global_vars._GLOBAL_TENSORBOARD_WRITER
    if writer is not None:
        writer.close()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    install_metadata()
    install_native_transport()
    install_tracking()
    install_audit()
    upstream.train_valid_test_datasets_provider.is_distributed = True
    completed = False
    try:
        pretrain(upstream.train_valid_test_datasets_provider, model_provider,
                 ModelType.encoder_or_decoder, upstream.forward_step,
                 args_defaults={"tokenizer_type": "NullTokenizer"}, extra_args_provider=extra_arguments)
        completed = True
    except SystemExit as error:
        completed = error.code in (None, 0)
        raise
    finally:
        if completed:
            finish_native()
