"""统计检验（阶段 5）：模型比"永远预测涨"多出来的几个百分点，是真本事还是运气？

测试集只有约 378 天，所以每个结论都要附上"不确定性有多大"。
"""
import numpy as np
from scipy.stats import binomtest

SEED = 42


def mcnemar_exact(y_true, pred_a, pred_b):
    """配对比较两个模型（在同一批日子上）。

    只看"两个模型一个对、一个错"的日子：
    - b = A 对 B 错 的天数，c = A 错 B 对 的天数
    如果两模型一样好，b 和 c 应该差不多，像抛硬币。用精确二项检验算 p 值（双侧）。
    """
    a_right = pred_a == y_true
    b_right = pred_b == y_true
    b = int(np.sum(a_right & ~b_right))
    c = int(np.sum(~a_right & b_right))
    p = binomtest(b, b + c, 0.5).pvalue if b + c > 0 else 1.0
    return {"A对B错": b, "A错B对": c, "p_value": p}


def block_bootstrap_indices(n, block_len, rng):
    """移动块 bootstrap：每次随机抽一段连续的 block_len 天，拼到 n 天为止。

    为什么不一天一天地抽？股市有"波动聚集"（动荡的日子扎堆出现），
    整段整段地抽能保留这种相邻日子之间的关系，置信区间不会被低估。
    """
    starts = rng.integers(0, n - block_len + 1, size=int(np.ceil(n / block_len)))
    return np.concatenate([np.arange(s, s + block_len) for s in starts])[:n]


def bootstrap_accuracy_diff(y_true, pred_model, pred_base, n_boot=5000, block_len=10, seed=SEED):
    """模型准确率 − 基准准确率 的 95% 置信区间（移动块 bootstrap）。

    返回：观测到的差、区间下限、区间上限、以及 bootstrap 里差 <= 0 的比例（越小越说明模型真的更好）。
    """
    rng = np.random.default_rng(seed)
    y, m, b = map(np.asarray, (y_true, pred_model, pred_base))
    observed = (m == y).mean() - (b == y).mean()
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = block_bootstrap_indices(len(y), block_len, rng)
        diffs[i] = (m[idx] == y[idx]).mean() - (b[idx] == y[idx]).mean()
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {"diff": observed, "ci_low": low, "ci_high": high, "share_le_0": float((diffs <= 0).mean())}


def accuracy_vs_chance(y_true, pred, p0=0.5):
    """准确率是否显著高于 p0（默认 50%）：单侧精确二项检验。"""
    k = int(np.sum(np.asarray(pred) == np.asarray(y_true)))
    return {"correct": k, "n": len(y_true), "p_value": binomtest(k, len(y_true), p0, alternative="greater").pvalue}


def _abs_corr(X, y):
    """X 每一列和 y 的相关系数绝对值（一次算完所有列）。"""
    Xz = (X - X.mean(axis=0)) / X.std(axis=0)
    yz = (y - y.mean()) / y.std()
    return np.abs(Xz.T @ yz) / len(y)


def permutation_max_corr(X, y, n_perm=5000, seed=SEED):
    """置换检验：特征和目标之间"最强的那个相关系数"，是不是比纯随机更强？

    做法：把 y（涨跌）随机打乱 n_perm 次，打乱后涨跌和新闻毫无关系，记下每次"所有特征里最大的 |r|"。
    - p_single：只看最强那个特征、假装事先就只检验它时的 p 值（会偏乐观）
    - p_max：和"打乱后的最大 |r|"比较的 p 值——自动考虑了"同时看了很多特征、挑了最好的"这件事
    """
    rng = np.random.default_rng(seed)
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    real = _abs_corr(X, y)
    best = int(real.argmax())
    null = np.array([_abs_corr(X, rng.permutation(y)) for _ in range(n_perm)])
    return {
        "best_feature": best,
        "abs_r": float(real[best]),
        "p_single": float((null[:, best] >= real[best]).mean()),
        "p_max": float((null.max(axis=1) >= real[best]).mean()),
    }
