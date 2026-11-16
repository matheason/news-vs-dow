"""数据读取、清理、检查、切分（阶段 1）。

每个函数都很短，方便在评委面前逐个解释。
"""
import re

import numpy as np
import pandas as pd

HEADLINE_COLS = [f"Top{i}" for i in range(1, 26)]

TRAIN_START, TRAIN_END = "2008-08-08", "2014-12-31"
TEST_START, TEST_END = "2015-01-02", "2016-07-01"

# 匹配整条标题被包在 b'...' 或 b"..." 里的情况（Python 字节串被直接写进了 CSV）
_BYTES_PATTERN = re.compile(r"""^b(['"])(.*)\1$""", re.DOTALL)


def clean_headline(text):
    """去掉 b'...' / b"..." 前缀，空值变成空字符串，去掉首尾空格。"""
    if pd.isna(text):
        return ""
    text = str(text).strip()
    match = _BYTES_PATTERN.match(text)
    if match:
        text = match.group(2)
    return text.strip()


def load_news(path):
    """读取 Combined_News_DJIA.csv：日期转类型、按日期排序、清理 25 列标题。"""
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    for col in HEADLINE_COLS:
        df[col] = df[col].map(clean_headline)
    return df


def load_djia(path):
    """读取 DJIA_table.csv：日期转类型、按日期排序（原文件是倒序的）。"""
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def check_labels(news, djia):
    """把 Label 和道指价格算出来的涨跌对比。

    同时检查两种定义：
    - 收盘对前一天收盘（Adj Close / Close 今天 >= 昨天）
    - 收盘对当天开盘（Close >= Open）
    返回合并后的表，以及每种定义的一致率。
    """
    prices = djia.copy()
    for col, price in [("up_adjclose", "Adj Close"), ("up_close", "Close")]:
        change = prices[price].diff()
        # 第一天没有"前一天"，change 是 NaN，保留 NaN，不参与比较
        prices[col] = np.where(change.isna(), np.nan, (change >= 0).astype(float))
    prices["up_open_close"] = (prices["Close"] >= prices["Open"]).astype(float)

    merged = news[["Date", "Label"]].merge(prices, on="Date", how="left")
    rates = {}
    for col in ["up_adjclose", "up_close", "up_open_close"]:
        valid = merged[col].notna()
        rates[col] = {
            "compared_days": int(valid.sum()),
            "agreement": float((merged.loc[valid, col] == merged.loc[valid, "Label"]).mean()),
        }
    return merged, rates


def split_by_date(df):
    """按日期切分训练集 / 测试集，绝不打乱。"""
    train = df[(df["Date"] >= TRAIN_START) & (df["Date"] <= TRAIN_END)].copy()
    test = df[(df["Date"] >= TEST_START) & (df["Date"] <= TEST_END)].copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)


def baseline_always_up(test):
    """基准 1：永远预测"涨"。"""
    return np.ones(len(test), dtype=int)


def baseline_previous_day(train, test):
    """基准 2：预测今天和前一个交易日同方向。

    测试集第一天的"前一天"是训练集最后一天，所以用训练集最后一个 Label 补上。
    """
    prev = test["Label"].shift(1)
    prev.iloc[0] = train["Label"].iloc[-1]
    return prev.astype(int).to_numpy()


# 阶段 3.4 / 4.1 用：判断一条标题是不是在"报道股市行情"，以及说的是涨还是跌。
# (?:...) 是"不捕获"的分组，只用来把几个写法放在一起。
MARKET_WORDS = r"\b(?:stocks?|stock markets?|dow|wall street|s&p|nasdaq|shares)\b"
DOWN_WORDS = (r"\b(?:plunge[sd]?|crash(?:es|ed)?|tumble[sd]?|fall(?:s|en)?|fell|slump(?:s|ed)?"
              r"|sink[s]?|sank|drop(?:s|ped)?|lose[s]?|lost|sell-?off)\b")
UP_WORDS = (r"\b(?:rall(?:y|ies|ied)|surge[sd]?|soar(?:s|ed)?|jump(?:s|ed)?|rise[sn]?|rose"
            r"|gain(?:s|ed)?|climb(?:s|ed)?|rebound(?:s|ed)?)\b")
