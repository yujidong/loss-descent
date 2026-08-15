"""教学用小型字符级语料与合成任务。

TINY_STORY：自撰英文小故事（约 2.5K 字符），刻意使用重复句式与
小词汇表，让 n-gram / RNN / LSTM 三代语言模型在同一语料上可比。

make_grammar_corpus：语法模板 + Zipf 词频的大型合成语料（6.1 章的
scaling 实验燃料）——不同尺寸的模型在它身上量出可拟合的幂律。
"""
from __future__ import annotations

import numpy as np

TINY_STORY = """once upon a time there was a little robot named pip. pip lived in a
small house at the edge of a quiet town. every morning pip opened the door and
looked at the sky. the sky was big and blue. pip liked the sky.

one day pip found a cat in the garden. the cat was small and gray. pip said
hello to the cat. the cat said nothing, because cats do not talk. but the cat
followed pip everywhere.

pip and the cat walked to the market. the market was loud and bright. there
were apples and bread and honey on the tables. pip bought an apple and gave
half of it to the cat. the cat ate the apple and slept in the sun.

the next morning the sky was dark. rain fell on the small house. pip and the
cat stayed inside. pip read a book about the sea. the cat slept on the floor.
the rain made a soft sound on the roof.

when the rain stopped, pip and the cat went to the hill. from the hill they
could see the town and the sea. the sea was big and gray and full of waves.
pip said, one day we will cross the sea. the cat said nothing, and looked at
the waves.

in the town there was an old baker. the baker gave pip warm bread every
friday. the bread was soft and sweet. pip shared the bread with the cat and
with the birds. the birds were small and brown. they sang in the morning.

one night the lights of the town went out. pip could not see the road. but
the cat could see in the dark. the cat walked in front, and pip followed.
together they found the way home. after that night, pip was never afraid of
the dark.

summer came. the days were long and warm. pip and the cat sat by the sea.
children played in the sand. the children liked the little robot and the
small gray cat. they gave the cat a fish. the cat was very happy.

autumn came. the leaves fell from the trees. the town was quiet and red and
gold. pip fixed the roof of the small house. the cat watched the birds fly
away to the south. the sky was full of birds.

winter came. snow fell on the town. the sea was cold and the streets were
white. pip and the cat stayed warm inside. pip told the cat stories about
the sea and the ships and the far lands beyond the waves. the cat slept and
dreamed of fish.

years passed. the little robot and the small gray cat grew old together. the
town changed. new houses were built. new children came to play. but every
morning, pip still opened the door and looked at the sky. the sky was still
big and blue. and the cat was still there, at the edge of the garden, in the
sun.
"""


def char_corpus(text: str = TINY_STORY):
    """字符级语料：返回 (ids, vocab, inverse_vocab)。

    统一小写、压缩空白——让统计集中在字符结构本身。
    """
    normalized = " ".join(text.split())
    chars = sorted(set(normalized))
    vocab = {c: i for i, c in enumerate(chars)}
    inv = {i: c for c, i in vocab.items()}
    ids = [vocab[c] for c in normalized]
    return ids, vocab, inv


def make_grammar_corpus(n_chars: int = 60000, seed: int = 0) -> str:
    """语法模板 + Zipf 词频的合成语料（scaling 实验燃料）。

    结构分层：高频虚词（the/and/a）+ 中频名词动词 + 低频修饰词，
    句式取自十几个模板——不同容量的模型在不同层次上"啃得动"，
    正是幂律 loss 曲线需要的多尺度结构。
    """
    rng = np.random.default_rng(seed)

    def zipf(words, s=1.1):
        ranks = np.arange(1, len(words) + 1)
        p = 1.0 / ranks**s
        p /= p.sum()
        return list(words), p

    subjects, p_sub = zipf(["the cat", "the dog", "the bird", "pip", "the baker",
                            "the children", "the old man", "a little fish"])
    verbs, p_verb = zipf(["sat", "ran", "slept", "walked", "looked", "sang",
                          "played", "read", "smiled", "climbed", "danced", "listened"])
    objects, p_obj = zipf(["the garden", "the market", "the hill", "the sea",
                           "the house", "the road", "the field", "the tree",
                           "the bridge", "the shore"])
    adjs, p_adj = zipf(["quiet", "warm", "small", "bright", "gray", "soft",
                        "cold", "sweet", "loud", "calm", "golden", "ancient"])
    temps, p_temp = zipf(["in the morning", "at noon", "in the evening", "at night",
                          "after the rain", "before the storm", "in summer", "in winter"])

    parts = []
    total = 0
    while total < n_chars:
        t = rng.random()
        if t < 0.30:
            s = f"{rng.choice(subjects, p=p_sub)} {rng.choice(verbs, p=p_verb)} {rng.choice(objects, p=p_obj)} ."
        elif t < 0.50:
            s = (f"{rng.choice(subjects, p=p_sub)} was {rng.choice(adjs, p=p_adj)} "
                 f"{rng.choice(temps, p=p_temp)} .")
        elif t < 0.65:
            s = (f"{rng.choice(temps, p=p_temp).capitalize()}, {rng.choice(subjects, p=p_sub)} "
                 f"and {rng.choice(subjects, p=p_sub)} {rng.choice(verbs, p=p_verb)} "
                 f"to {rng.choice(objects, p=p_obj)} .")
        elif t < 0.80:
            s = (f"the {rng.choice(adjs, p=p_adj)} {rng.choice(objects, p=p_obj).split()[-1]} "
                 f"was {rng.choice(adjs, p=p_adj)} and {rng.choice(adjs, p=p_adj)} .")
        else:
            s = (f"every {rng.choice(['morning', 'evening', 'friday', 'night'])}, "
                 f"{rng.choice(subjects, p=p_sub)} {rng.choice(verbs, p=p_verb)} "
                 f"near {rng.choice(objects, p=p_obj)} .")
        parts.append(s)
        total += len(s) + 1
    return " ".join(parts)


def make_analogy_corpus(n_sentences: int = 1200, seed: int = 0):
    """带清晰语义结构的词级语料（word2vec 专用）。

    词对（成体-幼崽）共享上下文模板 → 嵌入空间中"成体→幼崽"方向
    一致，king-man+woman 式类比可复现。
    """
    rng = np.random.default_rng(seed)
    pairs = [("cat", "kitten"), ("dog", "puppy"), ("bird", "chick"), ("cow", "calf")]
    verbs = ["runs", "sits", "sleeps", "plays"]
    places = ["garden", "yard", "field", "hill"]
    weathers = ["warm", "quiet", "sunny", "calm"]
    sentences = []
    for _ in range(n_sentences):
        adult, baby = pairs[rng.integers(len(pairs))]
        template = rng.integers(4)
        if template == 0:
            sentences.append(f"the {adult} is big . the {baby} is small .")
        elif template == 1:
            sentences.append(f"the {adult} {verbs[rng.integers(len(verbs))]} in the {places[rng.integers(len(places))]} .")
        elif template == 2:
            sentences.append(f"a {weathers[rng.integers(len(weathers))]} day for the {baby} and the {adult} .")
        else:
            sentences.append(f"the {baby} stays near the {adult} .")
    words = sorted({w for s in sentences for w in s.split()})
    vocab = {w: i for i, w in enumerate(words)}
    encoded = [[vocab[w] for w in s.split()] for s in sentences]
    return encoded, vocab


def make_reversal_task(n: int = 600, max_len: int = 5, seed: int = 0):
    """序列反转'翻译'任务（seq2seq 专用）：输入 3..max_len 个数字，目标为其反转。

    特殊记号：0=BOS, 1=EOS；数字词表 2..11。返回 (inputs, targets)，
    每条是词 id 列表。
    """
    rng = np.random.default_rng(seed)
    BOS, EOS = 0, 1
    inputs, targets = [], []
    for _ in range(n):
        length = int(rng.integers(3, max_len + 1))
        seq = [int(rng.integers(2, 12)) for _ in range(length)]
        inputs.append([BOS] + seq + [EOS])
        targets.append([BOS] + seq[::-1] + [EOS])
    return inputs, targets
