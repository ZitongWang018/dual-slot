"""Actual TE/FlashAttention paired-slot outputs, gradients and kernel audit."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor/Megatron-LM"))
from types import MethodType, SimpleNamespace

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.enums import AttnBackend
from megatron.core.optimizer import _get_param_groups
from dual_slot.native import enable_dual_slot


def build_model(fused=False):
    config = TransformerConfig(num_layers=6, hidden_size=128, ffn_hidden_size=512,
        num_attention_heads=2, num_query_groups=1, kv_channels=64,
        normalization="RMSNorm", qk_layernorm=True, layernorm_epsilon=1e-6,
        gated_linear_unit=True, activation_func=torch.nn.functional.silu,
        add_bias_linear=False, add_qkv_bias=False, hidden_dropout=0, attention_dropout=0,
        params_dtype=torch.bfloat16, bf16=True, attention_backend=AttnBackend.flash,
        gradient_accumulation_fusion=fused, disable_parameter_transpose_cache=True, bias_activation_fusion=True,
        bias_dropout_fusion=True, apply_rope_fusion=True, cross_entropy_loss_fusion=True)
    return GPTModel(config, get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True),
        vocab_size=128, max_sequence_length=64, parallel_output=True,
        share_embeddings_and_output_weights=False, position_embedding_type="rope",
        rotary_base=1000000).cuda()


def main():
    torch.cuda.set_device(0)
    dist.init_process_group('nccl', init_method='tcp://127.0.0.1:29879', rank=0, world_size=1)
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    torch.manual_seed(42); model_parallel_cuda_manual_seed(42)
    tokens = torch.randint(1, 128, (2, 16), device='cuda'); tokens[:, 7] = 0
    positions = torch.arange(16, device='cuda').expand(2, -1)
    results = []
    for iteration, k, fused in [(0, 2, False), (2, 3, False), (0, 2, True), (2, 3, True)]:
        model, reference = build_model(fused), build_model()
        if fused:
            for param in model.parameters():
                param.main_grad = torch.zeros_like(param, dtype=torch.float32)
                param.grad_added_to_main_grad = False
        reference.load_state_dict(model.state_dict())
        keys = set(model.state_dict())
        enable_dual_slot(model, start=2, end=4, eod_id=0, seed=42, iteration_provider=lambda: iteration)

        def reference_forward(decoder, hidden_states, attention_mask, **kwargs):
            def layers(x, begin, end, rope):
                for layer in decoder.layers[begin:end]:
                    x, _ = layer(hidden_states=x, **dict(kwargs, attention_mask=None, rotary_pos_emb=rope))
                return x
            rope = kwargs['rotary_pos_emb']
            x = layers(hidden_states, 0, 2, rope)
            h = layers(x, 2, 4, rope)
            # Independent construction, no production shift/pair helper.
            for _ in range(k):
                feedback = torch.zeros_like(h)
                feedback[1:] = h[:-1] * (tokens[:, :-1] != 0).T.unsqueeze(-1)
                slots = torch.stack([item for t in range(len(x)) for item in (x[t], x[t] + feedback[t])])
                paired_rope = torch.stack([rope[t] for t in range(len(rope)) for _ in range(2)])
                slots = layers(slots, 2, 4, paired_rope)
                h, prediction = slots[::2], slots[1::2]
            return decoder.final_layernorm(layers(prediction.contiguous(), 4, 6, rope))

        reference.decoder.forward = MethodType(reference_forward, reference.decoder)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as profile:
            actual = model(tokens, positions, None)
            actual.float().square().mean().backward()
            torch.cuda.synchronize()
        expected = reference(tokens, positions, None)
        expected.float().square().mean().backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        max_grad_error = 0.
        max_core_relative_error = 0.
        for (name, a), (_, b) in zip(model.named_parameters(), reference.named_parameters()):
            gradient = a.main_grad if fused and a.grad_added_to_main_grad else a.grad
            assert gradient is not None and torch.isfinite(gradient).all(), name
            torch.testing.assert_close(gradient.float(), b.grad.float(), rtol=0.01, atol=0.002, msg=name)
            max_grad_error = max(max_grad_error, float((gradient.float() - b.grad.float()).abs().max()))
            if fused and name.startswith(('decoder.layers.2.', 'decoder.layers.3.')) and a.ndim == 2:
                relative_error = float((gradient.float() - b.grad.float()).norm() / b.grad.float().norm().clamp_min(1e-12))
                assert relative_error < 0.02, f'Shared core main_grad lost contributions: {name}: {relative_error}'
                max_core_relative_error = max(max_core_relative_error, relative_error)
        kernels = sorted(set(event.name for event in profile.events() if 'flash' in event.name.lower()))
        assert kernels, 'No actual FlashAttention kernel observed'
        assert set(model.state_dict()) == keys
        model.ddp_config = SimpleNamespace(use_custom_fsdp=False)
        groups = _get_param_groups([model], None, None, 1., 0.0015, 0.00015, None, None)
        for group in groups:
            for param in group['params']:
                assert group['wd_mult'] == (0. if param.ndim == 1 else 1.)
        results.append({'k': k, 'fused_main_grad': fused, 'outputs_equal': True, 'max_gradient_error': max_grad_error,
            'max_core_relative_gradient_error': max_core_relative_error,
            'flash_attention_observed': True, 'flash_kernels': kernels[:4],
            'weight_decay_groups_correct': True,
            'residual_init_std': model.config.output_layer_init_method.keywords['std']})
        del actual, expected, model, reference, groups, profile
    print(json.dumps({'native_gpu_audit': results, 'gpu': torch.cuda.get_device_name()}), flush=True)


if __name__ == '__main__':
    try:
        main()
    finally:
        import gc
        gc.collect()
        torch.cuda.synchronize()
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()
