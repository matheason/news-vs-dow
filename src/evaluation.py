"""评估指标：准确率（主）、混淆矩阵、"跌"类召回率。"""
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score


def evaluate(y_true, y_pred, name):
    """返回一行结果，方便把多个模型拼成一张表。"""
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        # pos_label=0：把"跌"当成要找的那一类，算出真实下跌日里被猜对的比例
        "down_recall": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
    }


def confusion_table(y_true, y_pred):
    """混淆矩阵做成带行列名的表：行 = 真实，列 = 预测。"""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(cm, index=["真实 跌(0)", "真实 涨(1)"], columns=["预测 跌(0)", "预测 涨(1)"])
