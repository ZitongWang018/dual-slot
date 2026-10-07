"""Independent slot construction and full-gradient checks without Megatron/CUDA."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from dual_slot.native import core_range, enable_dual_slot


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)
        self.calls = []

    def forward(self, *, hidden_states, context=None, rotary_pos_emb=None, **kwargs):
        self.calls.append(hidden_states.shape[0])
        x = hidden_states + rotary_pos_emb * 0.01
        score = torch.einsum('tbh,sbh->bts', x, x) / 2
        future = torch.ones(x.shape[0], x.shape[0], dtype=torch.bool).triu(1)
        prob = score.masked_fill(future, -torch.inf).softmax(-1)
        y = torch.einsum('bts,sbh->tbh', prob, x)
        return torch.tanh(self.linear(y)) + hidden_states, context


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
            context_parallel_size=1, recompute_granularity=None, fp8=None, cpu_offloading=False,
            hidden_dropout=0, attention_dropout=0, sequence_parallel=False)
        self.position_embedding_type = 'rope'
        self.embedding = nn.Embedding(12, 4)
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList(Layer() for _ in range(6))
        self.head = nn.Linear(4, 12, bias=False)

    def forward(self, input_ids):
        x = self.embedding(input_ids).transpose(0, 1)
        rope = torch.arange(x.shape[0], dtype=x.dtype)[:, None, None]
        for layer in self.decoder.layers:
            x, _ = layer(hidden_states=x, rotary_pos_emb=rope, attention_mask=None)
        return self.head(x)


def independent(model, ids, k):
    rope = torch.arange(ids.shape[1], dtype=model.embedding.weight.dtype)[:, None, None]
    def layers(x, begin, end, positions):
        for layer in model.decoder.layers[begin:end]:
            x, _ = layer(hidden_states=x, rotary_pos_emb=positions)
        return x
    x = layers(model.embedding(ids).transpose(0, 1), 0, 2, rope)
    latent = layers(x, 2, 4, rope)
    for _ in range(k):
        feedback = torch.zeros_like(latent)
        feedback[1:] = latent[:-1] * (ids[:, :-1] != 0).T.unsqueeze(-1)
        slots = []
        for t in range(x.shape[0]):
            slots.extend((x[t], x[t] + feedback[t]))
        y = layers(torch.stack(slots), 2, 4, torch.stack([rope[t] for t in range(len(rope)) for _ in range(2)]))
        latent = torch.stack([y[t] for t in range(0, len(y), 2)])
        prediction = torch.stack([y[t] for t in range(1, len(y), 2)])
    return model.head(layers(prediction, 4, 6, rope))


@pytest.mark.parametrize('iteration,k', [(0, 2), (2, 3)])
def test_outputs_gradients_and_invocations(iteration, k):
    torch.manual_seed(42)
    model = Model().double()
    reference = deepcopy(model)
    keys = set(model.state_dict())
    enable_dual_slot(model, start=2, end=4, eod_id=0, seed=42, iteration_provider=lambda: iteration)
    ids = torch.tensor([[1, 0, 2, 3, 4, 5], [6, 7, 8, 0, 9, 10]])
    actual, expected = model(ids), independent(reference, ids, k)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().mean().backward()
    expected.square().mean().backward()
    for (name, a), (_, b) in zip(model.named_parameters(), reference.named_parameters()):
        assert a.grad is not None and torch.isfinite(a.grad).all(), name
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=1e-12)
    assert keys == set(model.state_dict())
    assert model.decoder.layers[0].calls == [6]
    assert model.decoder.layers[2].calls == [6] + [12] * k
    assert model.decoder.layers[5].calls == [6]
    assert model.decoder.dual_slot_last_k == k


def test_causality_and_eod_is_not_attention_isolation():
    model = Model().double().eval()
    enable_dual_slot(model, start=2, end=4, eod_id=0, seed=42, iteration_provider=lambda: 0)
    ids = torch.tensor([[1, 0, 2, 3, 4, 5]])
    y = model(ids)
    changed = ids.clone(); changed[0, -1] = 10
    torch.testing.assert_close(model(changed)[:-1], y[:-1], rtol=0, atol=0)
    changed = ids.clone(); changed[0, 0] = 10
    assert not torch.equal(model(changed)[2:], y[2:])
    assert core_range(6) == (2, 4)
