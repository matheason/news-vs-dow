"""简单回测（阶段 6）：如果真按模型的信号买卖，赚不赚钱？

规则（尽量简单，方便解释）：
- 预测"涨" → 当天持有道指；预测"跌" → 当天空仓（拿现金，收益 0）。
- 每次从持有变空仓、或从空仓变持有，扣一次交易成本。
- 和"一直持有（买入持有）"比较。
这是原型，不考虑分红、税、滑点细节，也不构成任何投资建议。
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def daily_returns(djia, kind):
    """从价格算每天的收益率。

    kind="close_to_close"：今天收盘 / 昨天收盘 − 1（对应原始 Label）
    kind="intraday"：今天收盘 / 今天开盘 − 1（对应"开盘买、收盘卖"）
    """
    if kind == "close_to_close":
        r = djia["Close"].pct_change()
    elif kind == "intraday":
        r = djia["Close"] / djia["Open"] - 1
    else:
        raise ValueError(kind)
    return pd.DataFrame({"Date": djia["Date"], "ret": r})


def run_strategy(signal, ret, cost=0.0005, intraday=False):
    """signal：每天 1（持有）或 0（空仓）；ret：对应日子的道指收益率。cost：每笔买或卖的成本（默认 0.05%）。

    - intraday=False（收盘到收盘）：只有仓位变化时才交易。第一天从空仓开始，第一天持有也算一次买入。
    - intraday=True（开盘买、收盘卖）：每个持有日都是一买一卖，扣两次成本。
    返回每天的策略收益（已扣成本）。
    """
    signal = np.asarray(signal, dtype=float)
    ret = np.asarray(ret, dtype=float)
    trades = 2 * signal if intraday else np.abs(np.diff(signal, prepend=0.0))
    return signal * ret - trades * cost


def summary(strategy_ret, name):
    """一行汇总：总收益、年化收益、年化波动、夏普比率（无风险利率按 0）、最大回撤。"""
    r = np.asarray(strategy_ret, dtype=float)
    wealth = np.cumprod(1 + r)
    years = len(r) / TRADING_DAYS
    vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    return {
        "strategy": name,
        "total_return": wealth[-1] - 1,
        "annual_return": wealth[-1] ** (1 / years) - 1,
        "annual_vol": vol,
        "sharpe": r.mean() * TRADING_DAYS / vol if vol > 0 else np.nan,
        "max_drawdown": float((wealth / np.maximum.accumulate(wealth) - 1).min()),
    }


def wealth_curve(strategy_ret):
    """每天的累计净值（从 1 开始），用来画图。"""
    return np.cumprod(1 + np.asarray(strategy_ret, dtype=float))
