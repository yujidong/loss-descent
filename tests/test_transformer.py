"""Transformer 部件测试：有限差分对拍 + 可训性。"""
import numpy as np

from dlbook.data.corpus import char_corpus
from dlbook.transformer import MiniGPT, MultiHeadAttention


def _numeric_vs_auto(model, idx, h=1e-5):
    """对 head.W 的若干元素做数值对拍（穿透整网的裁判）。"""
    model.loss_and_backward(idx)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(5):
        i = int(rng.integers(model.head.W.shape[0]))
        j = int(rng.integers(model.head.W.shape[1]))
        old = model.head.W[i, j]
        model.head.W[i, j] = old + h
        lp = model.loss_on(idx)
        model.head.W[i, j] = old - h
        lm = model.loss_on(idx)
        model.head.W[i, j] = old
        num = (lp - lm) / (2 * h)
        auto = model.head.grad_W[i, j]
        rel = abs(num - auto) / max(abs(num), abs(auto), 1e-8)
        worst = max(worst, rel)
    return worst


def test_mha_forward_shapes():
    mha = MultiHeadAttention(32, 4, seed=0)
    X = np.random.default_rng(0).normal(size=(3, 10, 32))
    out = mha.forward(X, causal=True)
    assert out.shape == (3, 10, 32)
    assert mha.attention_map(X).shape == (10, 10)
    # 因果性：位置 0 的注意力只在自己身上
    A = mha.attention_map(X)
    assert abs(A[0, 0] - 1.0) < 1e-6 and A[0, 1:].sum() < 1e-6


def test_minigpt_backward_finite_differences():
    ids, vocab, _ = char_corpus()
    V = len(vocab)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, V, size=(2, 16))
    model = MiniGPT(V, d_model=16, n_heads=2, n_blocks=2, block_size=16, seed=0)
    worst = _numeric_vs_auto(model, idx)
    assert worst < 0.05, f"worst relative error {worst}"


def test_minigpt_learns_char_lm():
    ids, vocab, _ = char_corpus()
    V = len(vocab)
    rng = np.random.default_rng(0)
    model = MiniGPT(V, d_model=32, n_heads=2, n_blocks=2, block_size=64, seed=0)
    arr = np.array(ids)
    first = None
    for step in range(800):
        starts = rng.integers(0, len(arr) - 65, size=8)
        batch = np.stack([arr[s : s + 65] for s in starts])
        l = model.loss_and_backward(batch)
        if first is None:
            first = l
        model.step(0.15, clip=1.0)
    assert first > l and l < 2.05  # 明显学习且优于 RNN 同预算（2.23）


def test_minigpt_with_moe_block_steps():
    ids, vocab, _ = char_corpus()
    V = len(vocab)
    from dlbook.transformer.layers import MoEMLP

    model = MiniGPT(V, d_model=16, n_heads=2, n_blocks=2, block_size=16, seed=0)
    model.blocks[1].mlp = MoEMLP(16, n_experts=2, seed=1)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, V, size=(2, 17))
    first = model.loss_and_backward(idx)
    for _ in range(30):
        l = model.loss_and_backward(idx)
        model.step(0.05, clip=1.0)
    assert first > l  # 含 MoE 块的整网可训


def test_lora_adapts_frozen_base():
    """LoRA：基座冻结、只训适配器也能微调；参数量远小于全参。"""
    ids, vocab, _ = char_corpus()
    V = len(vocab)
    rng = np.random.default_rng(0)
    base = MiniGPT(V, d_model=32, n_heads=2, n_blocks=2, block_size=64, seed=0)
    # 先训一个基座
    arr = np.array(ids)
    for _ in range(300):
        starts = rng.integers(0, len(arr) - 65, size=8)
        base.loss_and_backward(np.stack([arr[s : s + 65] for s in starts]))
        base.step(0.15, clip=1.0)

    lora = MiniGPT(V, d_model=32, n_heads=2, n_blocks=2, block_size=64, seed=0, lora_r=4)
    lora.tok_emb = base.tok_emb.copy()
    for lb, bb in zip(lora.blocks, base.blocks):
        for la, ba in ((lb.attn.Wq, bb.attn.Wq), (lb.attn.Wk, bb.attn.Wk),
                       (lb.attn.Wv, bb.attn.Wv), (lb.attn.Wo, bb.attn.Wo)):
            la.W = ba.W.copy()
        lb.mlp.fc1.W = bb.mlp.fc1.W.copy()
        lb.mlp.fc2.W = bb.mlp.fc2.W.copy()

    idx = np.random.default_rng(1).integers(0, V, size=(16, 33))
    first = lora.loss_and_backward(idx)
    for _ in range(200):
        l = lora.loss_and_backward(idx)
        lora.step(0.02, clip=1.0)
    assert first > l  # 适配器可训
    assert 0 < lora.n_trainable_params() < base.n_params() * 0.2
