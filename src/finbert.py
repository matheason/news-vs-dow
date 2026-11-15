"""FinBERT 情绪打分：批量处理 + 缓存到 data/，避免重复计算。"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_utils import HEADLINE_COLS

MODEL_NAME = "ProsusAI/finbert"
MAX_LENGTH = 128  # 标题都很短，128 个 token 足够；超过的部分会被截断（会统计数量）


def _load_model():
    # 放在函数里 import：有缓存时根本不用加载 torch，notebook 打开更快
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(42)
    # 有 NVIDIA 显卡（比如 Colab 的 GPU）就用 GPU，没有就用 CPU；结果只有极小的浮点差异
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).eval().to(device)
    print("FinBERT 运行设备:", device)
    return tok, model, device


def score_texts(texts, batch_size=64):
    """给一组文本打分，返回每条的 positive / negative / neutral 概率。

    - 先按长度排序再分批：同一批里长度相近，补齐(padding)更少，速度更快。
    - 标签顺序从模型配置里读，不手写，避免把 positive/negative 弄反。
    """
    import torch

    tok, model, device = _load_model()
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    order = np.argsort([len(t) for t in texts])
    probs = np.zeros((len(texts), len(labels)), dtype=np.float32)
    n_truncated = 0

    with torch.no_grad():  # 只做推理，不训练，关掉梯度省内存、更快
        for start in range(0, len(texts), batch_size):
            idx = order[start:start + batch_size]
            batch = [texts[i] for i in idx]
            n_truncated += sum(len(ids) > MAX_LENGTH for ids in tok(batch)["input_ids"])
            enc = tok(batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)
            probs[idx] = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
            done = start + len(batch)
            if done % (batch_size * 50) == 0 or done == len(texts):
                print(f"  已打分 {done}/{len(texts)}", flush=True)

    return pd.DataFrame(probs, columns=labels), n_truncated


def finbert_scores(news, cache_path):
    """返回长表：每行一条标题（Date, rank, positive, negative, neutral）。

    缓存文件存在就直接读；不存在才计算并保存。空标题不打分。
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f"从缓存读取: {cache_path.name}")
        return pd.read_csv(cache_path, parse_dates=["Date"])

    long = news.melt(id_vars="Date", value_vars=HEADLINE_COLS, var_name="rank", value_name="headline")
    long["rank"] = long["rank"].str.replace("Top", "").astype(int)
    long = long[long["headline"] != ""].sort_values(["Date", "rank"]).reset_index(drop=True)

    t0 = time.time()
    probs, n_truncated = score_texts(long["headline"].tolist())
    print(f"打分 {len(long)} 条标题用时 {time.time() - t0:.0f} 秒；被截断的标题 {n_truncated} 条")

    out = pd.concat([long[["Date", "rank"]], probs], axis=1)
    out.to_csv(cache_path, index=False)
    return out


def to_wide(scores_long, values):
    """把长表某一列转回"每天一行、Top1..Top25 一列"的宽表，和 VADER 的形状一样。"""
    wide = scores_long.pivot(index="Date", columns="rank", values=values)
    wide.columns = [f"Top{c}" for c in wide.columns]
    return wide.reindex(columns=HEADLINE_COLS).reset_index(drop=True)
