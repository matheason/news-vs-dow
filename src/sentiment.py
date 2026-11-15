"""情绪打分，并把每天 25 条标题汇总成"一天的特征"。"""
import nltk
import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from src.data_utils import HEADLINE_COLS

# VADER 官方建议：compound <= -0.05 算负面，>= 0.05 算正面
NEG_THRESHOLD, POS_THRESHOLD = -0.05, 0.05


def vader_scores(news):
    """给每条标题打 VADER compound 分（-1 最负面，+1 最正面）。

    返回和 Top1..Top25 同形状的表；空标题记为 NaN，不让它被当成"中性"拉低平均。
    """
    nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()

    def score(text):
        return sia.polarity_scores(text)["compound"] if text else np.nan

    return news[HEADLINE_COLS].apply(lambda col: col.map(score))


def daily_features(scores, prefix, is_neg=None, is_pos=None):
    """把一天 25 个分数汇总成几个数字（一行 = 一天）。

    is_neg / is_pos：和 scores 同形状的 True/False 表，说明哪些标题算负面/正面。
    不传时用 VADER 的阈值（±0.05）；FinBERT 则传"概率最大的类别"。

    - mean：整体情绪
    - min / max：最坏、最好的那条新闻（一条大坏消息可能比平均更重要）
    - std：当天新闻情绪分歧有多大
    - neg_share / pos_share：负面、正面标题占比
    - top1：排名第一（最热门）那条标题的情绪
    """
    if is_neg is None:
        is_neg = scores <= NEG_THRESHOLD
    if is_pos is None:
        is_pos = scores >= POS_THRESHOLD
    valid = scores.notna().sum(axis=1)
    return pd.DataFrame({
        f"{prefix}_mean": scores.mean(axis=1),
        f"{prefix}_min": scores.min(axis=1),
        f"{prefix}_max": scores.max(axis=1),
        f"{prefix}_std": scores.std(axis=1),
        f"{prefix}_neg_share": is_neg.sum(axis=1) / valid,
        f"{prefix}_pos_share": is_pos.sum(axis=1) / valid,
        f"{prefix}_top1": scores["Top1"].fillna(0.0),
    })
