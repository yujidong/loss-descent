"""Part 4 循环网络测试：有限差分对拍 + 可训性。"""
import numpy as np

from dlbook.data.corpus import char_corpus, make_analogy_corpus, make_reversal_task
from dlbook.rnn import LSTM, RNNLM, SimpleRNN


def _numeric_grads(model, tokens, h=1e-6):
    """对 RNNLM 的 Why 做数值梯度（裁判）。"""
    grads = []
    for W in (model.Why,):
        g = np.zeros_like(W)
        it = np.nditer(W, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            old = W[idx]
            W[idx] = old + h
            lp = model.loss_on(tokens)
            W[idx] = old - h
            lm = model.loss_on(tokens)
            W[idx] = old
            g[idx] = (lp - lm) / (2 * h)
            it.iternext()
        grads.append(g)
    return grads


def test_rnnlm_backward_matches_finite_differences():
    ids, vocab, _ = char_corpus()
    model = RNNLM(len(vocab), hidden=12, cell="rnn", seed=0)
    tokens = np.array(ids[:40])
    model.loss_and_backward(tokens)
    (g_num,) = _numeric_grads(model, tokens)
    assert np.allclose(g_num, model.grads["Why"], atol=1e-5)


def test_lstm_backward_matches_finite_differences():
    ids, vocab, _ = char_corpus()
    model = RNNLM(len(vocab), hidden=10, cell="lstm", seed=0)
    tokens = np.array(ids[:30])
    model.loss_and_backward(tokens)
    (g_num,) = _numeric_grads(model, tokens)
    assert np.allclose(g_num, model.grads["Why"], atol=1e-5)


def test_char_lm_learns_something():
    ids, vocab, _ = char_corpus()
    tokens = np.array(ids)
    uniform = float(np.log(len(vocab)))  # 均匀基线
    model = RNNLM(len(vocab), hidden=32, cell="rnn", seed=0)
    first = model.loss_and_backward(tokens)
    for _ in range(100):
        model.loss_and_backward(tokens)
        model.step(0.1)
    final = model.loss_on(tokens)
    assert first > final < uniform * 0.82  # 显著优于均匀猜测


def test_reversal_and_analogy_data():
    inputs, targets = make_reversal_task(n=10, seed=0)
    assert inputs[0][0] == 0 and inputs[0][-1] == 1  # BOS/EOS
    body_in = inputs[0][1:-1]
    body_out = targets[0][1:-1]
    assert list(body_out) == list(body_in)[::-1]
    enc, vocab = make_analogy_corpus(n_sentences=20)
    assert len(enc) == 20 and "kitten" in vocab and "calf" in vocab
