"""Dual Slot adapter for the unmodified Megatron GPT training engine.

The upstream decoder executes prefix/warm/suffix normally. Only the final
core-layer output is replaced with the prediction states from paired rounds.
No parameter, optimizer, dataset, loss or checkpoint tensor is added.
"""
from __future__ import annotations

import random
import hashlib
import json
import torch


def core_range(num_layers, start=None, end=None):
    if (start is None) != (end is None):
        raise ValueError("Specify both core bounds")
    if start is None:
        if num_layers < 3:
            raise ValueError("Middle-third core requires at least three layers")
        start, end = num_layers // 3 + 1, 2 * num_layers // 3
    if not 1 <= start <= end <= num_layers:
        raise ValueError("Invalid one-based inclusive core bounds")
    return start - 1, end


def shift_previous(states, tokens, eod_id):
    shifted = torch.cat((torch.zeros_like(states[:1]), states[:-1]), dim=0)
    if eod_id >= 0:
        valid = (tokens[:, :-1] != eod_id).T.unsqueeze(-1)
        shifted = torch.cat((shifted[:1], shifted[1:] * valid), dim=0)
    return shifted


def paired_rope(rope, length):
    if isinstance(rope, tuple):
        return tuple(paired_rope(item, length) for item in rope)
    if rope is None:
        return None
    if rope.shape[0] != length:
        raise ValueError("Dual Slot requires full-sequence continuous RoPE")
    return rope.repeat_interleave(2, dim=0)


def prepare_shared_weight_gradients(model):
    """Accumulate every use of a shared TE weight into native main_grad.

    TE treats is_first_microbatch=True as overwrite, which erases gradients
    from later forward uses when the warm pass is visited last in backward.
    Native DDP already zeroes the buffer once per optimizer step. With BF16
    there is no FP8 weight cache to preserve; pass None to TE instead.
    """
    if not getattr(model.config, 'gradient_accumulation_fusion', False):
        return
    model.config.disable_parameter_transpose_cache = True
    for module in model.modules():
        if hasattr(module, 'disable_parameter_transpose_cache'):
            module.disable_parameter_transpose_cache = True


def enable_dual_slot(model, *, start, end, eod_id, seed, iteration_provider):
    config = model.config
    if any(getattr(config, name) != 1 for name in (
        "tensor_model_parallel_size", "pipeline_model_parallel_size", "context_parallel_size"
    )):
        raise ValueError("Dual Slot requires TP=PP=CP=1")
    if config.recompute_granularity is not None or config.fp8 or config.cpu_offloading:
        raise ValueError("Dual Slot requires BF16/FP32 without recomputation or offloading")
    if config.hidden_dropout or config.attention_dropout:
        raise ValueError("Dual Slot requires zero dropout")
    if getattr(config, "num_moe_experts", None) is not None:
        raise ValueError("This adapter supports dense models only")
    if config.sequence_parallel or getattr(config, "enable_cuda_graph", False) or getattr(config, "external_cuda_graph", False):
        raise ValueError("Sequence parallelism and CUDA graphs are unsupported")
    if model.position_embedding_type != "rope":
        raise ValueError("Dual Slot requires RoPE")
    decoder = model.decoder
    if hasattr(decoder, "_dual_slot_handles"):
        raise ValueError("Dual Slot is already enabled")
    if not 0 <= start < end <= len(decoder.layers):
        raise ValueError("Invalid core range")
    prepare_shared_weight_gradients(model)
    state = {"inside": False, "tokens": None, "x": None, "kwargs": None}

    def capture_tokens(_module, inputs, kwargs):
        tokens = kwargs.get("input_ids", inputs[0] if inputs else None)
        if tokens is None or tokens.ndim != 2:
            raise ValueError("Expected [batch, sequence] token IDs")
        # Official forward_step passes optional arguments positionally.
        mask = kwargs.get("attention_mask", inputs[2] if len(inputs) > 2 else None)
        if mask is not None:
            raise ValueError("Use standard causal attention with no explicit document mask")
        for name in ("inference_context", "inference_params", "packed_seq_params"):
            if kwargs.get(name) is not None:
                raise ValueError(f"Native pretraining adapter does not support {name}")
        state["tokens"] = tokens

    def capture_core(_module, _inputs, kwargs):
        if not state["inside"]:
            if kwargs.get("attention_mask") is not None:
                raise ValueError("Core expects implicit causal attention")
            state["x"] = kwargs["hidden_states"]
            state["kwargs"] = {key: value for key, value in kwargs.items() if key != "hidden_states"}

    def refine(_module, _inputs, _kwargs, output):
        if state["inside"]:
            return None
        if state["x"] is None or state["tokens"] is None:
            raise RuntimeError("Core input capture failed")
        k = random.Random(seed + iteration_provider()).choice((2, 3)) if model.training else 3
        decoder.dual_slot_last_k = k
        latent, context = output
        x, tokens = state["x"], state["tokens"]
        audit_key = (iteration_provider(), model.training)
        if getattr(decoder, 'dual_slot_audit', False) and audit_key not in state.setdefault('audit_keys', set()):
            state['audit_keys'].add(audit_key)
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                digest = lambda value: hashlib.sha256(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
                print(json.dumps({'dual_slot_rounds': k, 'iteration': audit_key[0],
                    'training': model.training, 'core_input_length': x.shape[0],
                    'paired_length': 2 * x.shape[0], 'core_input_sha256': digest(x),
                    'warm_output_sha256': digest(latent)}), flush=True)
        kwargs = dict(state["kwargs"])
        for key in ("rotary_pos_emb", "rotary_pos_cos", "rotary_pos_sin"):
            kwargs[key] = paired_rope(kwargs.get(key), x.shape[0])
        state["inside"] = True
        try:
            for _ in range(k):
                prediction_input = x + shift_previous(latent, tokens, eod_id)
                hidden = torch.stack((x, prediction_input), dim=1).flatten(0, 1)
                for layer in decoder.layers[start:end]:
                    hidden, context = layer(hidden_states=hidden, **dict(kwargs, context=context))
                latent, prediction = hidden[0::2], hidden[1::2]
            # Return only T prediction states; suffix, final norm and loss remain upstream.
            return prediction.contiguous(), context
        finally:
            state.update(inside=False, tokens=None, x=None, kwargs=None)

    decoder._dual_slot_handles = [
        model.register_forward_pre_hook(capture_tokens, with_kwargs=True),
        decoder.layers[start].register_forward_pre_hook(capture_core, with_kwargs=True),
        decoder.layers[end - 1].register_forward_hook(refine, with_kwargs=True),
    ]
    decoder.dual_slot_core_range = (start, end)
