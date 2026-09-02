"""MiniGPT：教学尺度的 decoder-only Transformer（5.2 章）。

token 嵌入 + 正弦位置编码 + N 个 pre-LN 块 + 输出头。
模式：causal（GPT 式下一词预测）与 mlm（BERT 式掩码重建）——
5.4 章用同一个身体跑两条路线。
"""
from __future__ import annotations

import numpy as np

from dlbook.transformer.layers import Block, Linear, softmax_lastdim


def sinusoidal_pos(T: int, D: int) -> np.ndarray:
    assert D % 2 == 0, "正弦位置编码要求 d_model 为偶数"
    pe = np.zeros((T, D))
    pos = np.arange(T)[:, None]
    div = np.exp(np.arange(0, D, 2) * -(np.log(10000.0) / D))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)
    return pe


class MiniGPT:
    def __init__(self, vocab: int, d_model: int = 64, n_heads: int = 4,
                 n_blocks: int = 2, block_size: int = 64, seed: int = 0,
                 mode: str = "causal", lora_r: int = 0):
        rng = np.random.default_rng(seed)
        self.vocab, self.d_model = vocab, d_model
        self.mode, self.block_size = mode, block_size
        self.lora_r = lora_r
        self.tok_emb = rng.normal(0, 0.02, (vocab, d_model))
        self.blocks = [Block(d_model, n_heads, seed + 10 * i, lora_r=lora_r) for i in range(n_blocks)]
        self.head = Linear(d_model, vocab, seed + 999)
        self.pos = sinusoidal_pos(block_size, d_model)
        self.grad_tok = None
        self._vel = {}

    def n_trainable_params(self) -> int:
        """LoRA 模式下计适配器 + 全部 LayerNorm + 输出头（step() 实际更新的全集）；
        非 LoRA 模式即全部参数。"""
        if self.lora_r == 0:
            return self.n_params()
        total = 0
        for blk in self.blocks:
            for ln in (blk.ln1, blk.ln2):
                total += ln.g.size + ln.b.size
            for lin in (blk.attn.Wq, blk.attn.Wk, blk.attn.Wv, blk.attn.Wo,
                        blk.mlp.fc1, blk.mlp.fc2):
                if hasattr(lin, "lora_A"):
                    total += lin.lora_A.size + lin.lora_B.size
            mlp = blk.mlp
            if hasattr(mlp, "experts"):
                for e in mlp.experts:
                    for lin in (e.fc1, e.fc2):
                        if hasattr(lin, "lora_A"):
                            total += lin.lora_A.size + lin.lora_B.size
        total += self.head.W.size + self.head.b.size
        return total

    def n_params(self) -> int:
        def _lin_params(lin):
            n = lin.W.size + lin.b.size
            if hasattr(lin, "lora_A"):  # LoRA 底座仍在，适配器叠加计入
                n += lin.lora_A.size + lin.lora_B.size
            return n

        n = self.tok_emb.size
        for b in self.blocks:
            n += sum(_lin_params(l) for l in
                     (b.attn.Wq, b.attn.Wk, b.attn.Wv, b.attn.Wo))
            n += b.ln1.g.size + b.ln1.b.size
            mlp = b.mlp
            if hasattr(mlp, "experts"):  # MoEMLP：专家 + 门
                n += mlp.gate_w.size
                n += sum(_lin_params(l) for e in mlp.experts
                         for l in (e.fc1, e.fc2))
            else:
                n += _lin_params(mlp.fc1) + _lin_params(mlp.fc2)
            n += b.ln2.g.size + b.ln2.b.size
        return n + self.head.W.size + self.head.b.size

    def forward(self, idx, training_mask=None):
        """idx: (B, T) 词 id。training_mask: (B, T) 布尔，MLM 模式下标记被掩码位。"""
        B, T = idx.shape
        X = self.tok_emb[idx] + self.pos[:T]
        for blk in self.blocks:
            X = blk.forward(X, causal=(self.mode == "causal"))
        return self.head.forward(X)  # (B, T, V)

    def loss_and_backward(self, idx, targets=None) -> float:
        """causal：由 idx 自带下一词目标；mlm：targets 给掩码位目标，其余忽略。"""
        if self.mode == "causal":
            inputs, tg = idx[:, :-1], idx[:, 1:]
        else:
            inputs, tg = idx, targets
        logits = self.forward(inputs)
        B, T, V = logits.shape
        probs = softmax_lastdim(logits)
        if self.mode == "causal":
            loss = float(-np.mean(np.log(probs[np.arange(B)[:, None], np.arange(T)[None, :], tg] + 1e-12)))
        else:
            mask = targets >= 0  # -1 表示忽略位
            loss = float(-np.mean(np.log(probs[mask][np.arange(mask.sum()), tg[mask]] + 1e-12)))
        dlogits = probs.copy()
        if self.mode == "causal":
            dlogits[np.arange(B)[:, None], np.arange(T)[None, :], tg] -= 1.0
            dlogits /= (B * T)
        else:
            m = (mask.astype(float) / mask.sum())[..., None]
            dlogits *= m
            dlogits[np.arange(B)[:, None], np.arange(T)[None, :], np.where(mask, tg, 0)] -= m[..., 0]
        dX = self.head.backward(dlogits)
        for blk in reversed(self.blocks):
            dX = blk.backward(dX)
        # 嵌入梯度（散射累加）。mlm 分支同样要训练嵌入，
        # 否则 BERT 式实验会在随机冻结的嵌入上跑。
        self.grad_tok = np.zeros_like(self.tok_emb)
        np.add.at(self.grad_tok, inputs.reshape(-1), dX.reshape(-1, self.d_model))
        return loss

    def loss_on(self, idx, targets=None) -> float:
        if self.mode == "causal":
            inputs, tg = idx[:, :-1], idx[:, 1:]
            mask = None
        else:
            inputs, tg, mask = idx, targets, targets >= 0
        logits = self.forward(inputs)
        probs = softmax_lastdim(logits)
        B, T, V = logits.shape
        if self.mode == "causal":
            return float(-np.mean(np.log(probs[np.arange(B)[:, None], np.arange(T)[None, :], tg] + 1e-12)))
        sel = probs[mask]
        return float(-np.mean(np.log(sel[np.arange(mask.sum()), tg[mask]] + 1e-12)))

    def _param_grad_entries(self):
        """(参数, 梯度, 键) 三元组列表——全局范数裁剪与集中更新的基础。"""
        entries = []
        if self.grad_tok is not None and self.lora_r == 0:
            entries.append((self.tok_emb, self.grad_tok, "tok"))
        for bi, blk in enumerate(self.blocks):
            a = blk.attn
            for name, lin in (("Wq", a.Wq), ("Wk", a.Wk), ("Wv", a.Wv), ("Wo", a.Wo)):
                if getattr(lin, "lora_r", 0) > 0:
                    entries.append((lin.lora_A, lin.grad_lora_A, f"b{bi}.{name}.A"))
                    entries.append((lin.lora_B, lin.grad_lora_B, f"b{bi}.{name}.B"))
                else:
                    entries.append((lin.W, lin.grad_W, f"b{bi}.{name}.W"))
                    entries.append((lin.b, lin.grad_b, f"b{bi}.{name}.b"))
            for name, ln in (("ln1", blk.ln1), ("ln2", blk.ln2)):
                entries.append((ln.g, ln.grad_g, f"b{bi}.{name}.g"))
                entries.append((ln.b, ln.grad_b, f"b{bi}.{name}.b"))
            mlp = blk.mlp
            if hasattr(mlp, "experts"):  # MoEMLP：各专家 + 门
                for ei, expert in enumerate(mlp.experts):
                    for name, lin in (("fc1", expert.fc1), ("fc2", expert.fc2)):
                        entries.append((lin.W, lin.grad_W, f"b{bi}.e{ei}.{name}.W"))
                        entries.append((lin.b, lin.grad_b, f"b{bi}.e{ei}.{name}.b"))
                entries.append((mlp.gate_w, mlp.grad_gate_w, f"b{bi}.gate"))
            else:
                for name, lin in (("fc1", mlp.fc1), ("fc2", mlp.fc2)):
                    if getattr(lin, "lora_r", 0) > 0:
                        entries.append((lin.lora_A, lin.grad_lora_A, f"b{bi}.{name}.A"))
                        entries.append((lin.lora_B, lin.grad_lora_B, f"b{bi}.{name}.B"))
                    else:
                        entries.append((lin.W, lin.grad_W, f"b{bi}.{name}.W"))
                        entries.append((lin.b, lin.grad_b, f"b{bi}.{name}.b"))
        entries.append((self.head.W, self.head.grad_W, "head.W"))
        entries.append((self.head.b, self.head.grad_b, "head.b"))
        return entries

    def step(self, lr: float, momentum: float = 0.9, clip: float = 1.0) -> None:
        """带全局梯度范数裁剪的动量更新——Transformer 训练的标准配菜。"""
        entries = [(p, g, k) for p, g, k in self._param_grad_entries() if g is not None]
        if clip:
            gnorm = float(np.sqrt(sum(float(np.sum(g**2)) for _, g, _ in entries)))
            scale = min(1.0, clip / max(gnorm, 1e-12))
        else:
            scale = 1.0
        for p, g, k in entries:
            if k not in self._vel:
                self._vel[k] = np.zeros_like(p)
            self._vel[k] = momentum * self._vel[k] - lr * scale * g
            p += self._vel[k]

    def generate(self, prompt_ids, n_new: int = 200, temperature: float = 0.8, seed: int = 0):
        rng = np.random.default_rng(seed)
        out = list(prompt_ids)
        for _ in range(n_new):
            window = np.array(out[-self.block_size:])[None, :]
            logits = self.forward(window)[0, -1]
            p = softmax_lastdim(logits / temperature)
            out.append(int(rng.choice(len(p), p=p)))
        return out
