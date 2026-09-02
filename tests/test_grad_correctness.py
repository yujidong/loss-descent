"""回归测试：有限差分校验 v1.2 审稿发现的梯度盲区。

背景：v1.1 的测试只对输出层权重做 FD，漏掉了 4 个 cell/模型级
梯度 bug（SimpleRNN 缺 Whh.T、LSTM t=0 初始状态、AttentionSeq2Seq
缺 1/√H、MiniGPT mlm 模式嵌入冻结）。本文件把每个盲区固化成测试。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dlbook.rnn import LSTM, SimpleRNN  # noqa: E402
from dlbook.rnn.seq2seq import AttentionSeq2Seq  # noqa: E402
from dlbook.transformer import MiniGPT  # noqa: E402


def _fd_dir_deriv(f, x, dx, eps=1e-6):
    """f 在 x 处沿 dx 的方向导数（中心差分）。"""
    return (f(x + eps * dx) - f(x - eps * dx)) / (2 * eps)


def _randn(rng, *shape):
    dx = rng.normal(size=shape)
    return dx / np.linalg.norm(dx)


def test_rnn_dwhh_and_dh0_directional(rng=None):
    """SimpleRNN: dWhh 与 dh0 的方向导数 vs 有限差分（Whh.T 回传）。"""
    rng = np.random.default_rng(0)
    cell = SimpleRNN(vocab=7, hidden=5, seed=0)
    tokens = rng.integers(0, 7, size=9)
    dhs = rng.normal(size=(9, 5))
    dh_init = rng.normal(size=5)  # 序列末端注入的额外梯度

    cell.forward(tokens)
    cell.backward(dhs, dh_init=dh_init)

    def loss_whh(Whh):
        backup = cell.Whh.copy()
        cell.Whh[...] = Whh
        hs = cell.forward(tokens)
        # 与 backward 的口径一致：dhs 逐项 + 末端的 dh_init
        out = float((hs * dhs).sum() + dh_init @ hs[-1])
        cell.Whh[...] = backup
        return out

    dx = _randn(rng, *cell.Whh.shape)
    analytic = float((cell.grads["Whh"] * dx).sum())
    fd = _fd_dir_deriv(loss_whh, cell.Whh.copy(), dx)
    assert analytic == pytest.approx(fd, rel=1e-4, abs=1e-6)


def test_rnn_loss_matches_grad_on_whh():
    """整条 RNN 的标量损失对 Whh 的 FD 校验（端到端，含 softmax 之外的全部路径）。"""
    rng = np.random.default_rng(1)
    cell = SimpleRNN(vocab=6, hidden=4, seed=1)
    tokens = rng.integers(0, 6, size=7)
    target = rng.normal(size=(7, 4))

    def loss_with(Whh):
        backup = cell.Whh.copy()
        cell.Whh[...] = Whh
        hs = cell.forward(tokens)
        val = float(((hs - target) ** 2).sum())
        cell.Whh[...] = backup
        return val

    cell.forward(tokens)
    cell.backward(2 * (cell._cache[1] - target))
    dx = _randn(rng, *cell.Whh.shape)
    analytic = float((cell.grads["Whh"] * dx).sum())
    fd = _fd_dir_deriv(loss_with, cell.Whh.copy(), dx)
    assert analytic == pytest.approx(fd, rel=1e-4, abs=1e-6)


def test_lstm_nonzero_init_state_grads():
    """LSTM: 非零 (h0, c0) 启动时，参数梯度必须包含 t=0 的两条路径。"""
    rng = np.random.default_rng(2)
    cell = LSTM(vocab=5, hidden=4, seed=2)
    tokens = rng.integers(0, 5, size=6)
    dhs = rng.normal(size=(6, 4))
    state = (rng.normal(size=4), rng.normal(size=4))

    cell.forward(tokens, state=state)
    cell.backward(dhs)

    def loss_wx(Wx):
        backup = cell.Wx.copy()
        cell.Wx[...] = Wx
        hs, _ = cell.forward(tokens, state=state)
        val = float((hs * dhs).sum())
        cell.Wx[...] = backup
        return val

    dx = _randn(rng, *cell.Wx.shape)
    analytic = float((cell.grads["Wx"] * dx).sum())
    fd = _fd_dir_deriv(loss_wx, cell.Wx.copy(), dx)
    assert analytic == pytest.approx(fd, rel=1e-4, abs=1e-6)


def test_lstm_zero_init_grads_still_exact():
    """零初态（章节实验的主流用法）不能因补丁而回归。"""
    rng = np.random.default_rng(3)
    cell = LSTM(vocab=5, hidden=4, seed=3)
    tokens = rng.integers(0, 5, size=6)
    dhs = rng.normal(size=(6, 4))
    cell.forward(tokens)
    cell.backward(dhs)

    def loss_wh(Wh):
        backup = cell.Wh.copy()
        cell.Wh[...] = Wh
        hs, _ = cell.forward(tokens)
        val = float((hs * dhs).sum())
        cell.Wh[...] = backup
        return val

    dx = _randn(rng, *cell.Wh.shape)
    analytic = float((cell.grads["Wh"] * dx).sum())
    fd = _fd_dir_deriv(loss_wh, cell.Wh.copy(), dx)
    assert analytic == pytest.approx(fd, rel=1e-4, abs=1e-6)


def test_attention_seq2seq_directional():
    """AttentionSeq2Seq: 编码器梯度含注意力 1/√H 系数（方向导数对拍）。"""
    rng = np.random.default_rng(4)
    model = AttentionSeq2Seq(vocab=6, hidden=4, seed=4)
    inp = list(rng.integers(2, 6, size=5))
    out = [0] + list(rng.integers(2, 6, size=4)) + [1]

    model.loss_and_backward(inp, out)

    def loss_enc(We):
        backup = model.enc.Wh.copy()
        model.enc.Wh[...] = We
        val = model.loss_and_backward(inp, out)
        model.enc.Wh[...] = backup
        return val

    dx = _randn(rng, *model.enc.Wh.shape)
    analytic = float((model.enc.grads["Wh"] * dx).sum())
    fd = _fd_dir_deriv(loss_enc, model.enc.Wh.copy(), dx)
    assert analytic == pytest.approx(fd, rel=1e-3, abs=1e-6)


def test_minigpt_mlm_trains_embeddings():
    """MLM 模式下 token 嵌入必须收到非零梯度（BERT 实验的前提）。"""
    rng = np.random.default_rng(5)
    m = MiniGPT(vocab=11, d_model=16, n_heads=2, n_blocks=1, block_size=8, mode="mlm", seed=5)
    idx = rng.integers(0, 11, size=(2, 6))
    targets = np.full_like(idx, -1)
    targets[:, 2] = idx[:, 3]
    targets[:, 4] = idx[:, 1]
    m.loss_and_backward(idx, targets)
    assert m.grad_tok is not None
    assert np.linalg.norm(m.grad_tok) > 0, "mlm 分支的嵌入梯度为零（冻结嵌入 bug 回归）"


def test_minigpt_causal_still_trains_embeddings():
    """causal 模式嵌入梯度不回归。"""
    rng = np.random.default_rng(6)
    m = MiniGPT(vocab=9, d_model=16, n_heads=2, n_blocks=1, block_size=8, mode="causal", seed=6)
    idx = rng.integers(0, 9, size=(2, 7))
    m.loss_and_backward(idx)
    assert np.linalg.norm(m.grad_tok) > 0


def test_n_params_with_moe_replacement():
    """MoE 替换块内 mlp 后 n_params() 不再崩溃。"""
    from dlbook.transformer.layers import MoEMLP

    m = MiniGPT(vocab=9, d_model=16, n_heads=2, n_blocks=2, block_size=8, seed=7)
    base = m.n_params()
    m.blocks[1].mlp = MoEMLP(16, n_experts=2, seed=7)
    moe = m.n_params()
    assert moe > base, "MoE 参数加倍后总数应更大"


def test_n_params_lora_counts_base_plus_adapters():
    """LoRA 模式下 n_params 计底座+适配器（总物理量），n_trainable 计可训练全集。"""
    full = MiniGPT(vocab=9, d_model=16, n_heads=2, n_blocks=1, block_size=8, seed=8)
    lora = MiniGPT(vocab=9, d_model=16, n_heads=2, n_blocks=1, block_size=8, seed=8, lora_r=4)
    assert lora.n_params() > full.n_params()
    trainable = lora.n_trainable_params()
    total = lora.n_params()
    assert 0 < trainable < total


def test_mlpnumpy_multioutput_loss_grad_consistent():
    """d_out=2 时 loss 与梯度口径一致（FD 比值应为 1 而非 2）。"""
    from dlbook.nn.mlp_numpy import MLPNumpy

    rng = np.random.default_rng(9)
    m = MLPNumpy(3, [8, 2], seed=9)
    X = rng.normal(size=(6, 3))
    Y = rng.normal(size=(6, 2))

    def loss_at(ws0):
        idx = 0
        for w in m.Ws:
            w[...] = ws0[idx : idx + w.size].reshape(w.shape)
            idx += w.size
        m.forward(X, training=True)
        return float(np.mean((m.forward(X, training=False) - Y) ** 2))

    ws0 = np.concatenate([w.ravel() for w in m.Ws]).copy()
    loss = m.backward(X, Y)
    # 沿某个权重方向的中心差分
    flat_grad = np.concatenate([g.ravel() for g in m.grad_Ws])
    dx = _randn(rng, *flat_grad.shape)

    eps = 1e-6
    up, dn = ws0.copy(), ws0.copy()
    up += eps * dx
    dn -= eps * dx
    fd = (loss_at(up) - loss_at(dn)) / (2 * eps)
    analytic = float((flat_grad * dx).sum())
    ratio = analytic / fd
    loss_at(ws0)  # 恢复原权重
    assert ratio == pytest.approx(1.0, rel=1e-3), f"梯度/FD 比值 {ratio}（=2 说明口径不一致回归）"
    assert loss == pytest.approx(loss_at(ws0), rel=1e-9)
