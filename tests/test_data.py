import numpy as np



def test_grammar_corpus_structure():
    from dlbook.data.corpus import make_grammar_corpus

    text = make_grammar_corpus(n_chars=5000, seed=0)
    text2 = make_grammar_corpus(n_chars=5000, seed=0)
    assert text == text2  # 确定性
    assert len(text) >= 5000
    assert " the " in text and "cat" in text
    # 多尺度结构：高频词与低频词的词频差距显著（Zipf）
    from collections import Counter
    words = Counter(text.split())
    freqs = np.array(sorted(words.values(), reverse=True), dtype=float)
    assert freqs[0] > 20 * np.median(freqs)
