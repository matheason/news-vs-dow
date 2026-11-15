"""模型训练：只在训练集内部用时间序列交叉验证调参。"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
CV = TimeSeriesSplit(n_splits=5)

LOGREG_GRID = {
    # C 越小 = 正则化越强 = 模型越"保守"，防止在噪声上过拟合
    "logisticregression__C": [0.001, 0.01, 0.1, 1, 10, 100],
    # balanced：给少数类（跌）更大权重，避免模型只会说"涨"
    "logisticregression__class_weight": [None, "balanced"],
}


def make_logreg():
    """标准化 + 逻辑回归。

    放进 Pipeline 的好处：每个交叉验证折里，标准化的均值/方差只用该折的训练部分算，
    不会把验证部分的信息泄露进来。
    """
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED))


def tune(model, grid, X_train, y_train):
    """在训练集上做 TimeSeriesSplit 网格搜索，返回搜索对象（含最佳模型，已用全部训练集重训）。

    TimeSeriesSplit：每一折都是"用前面的日子训练、用紧接着后面的日子验证"，
    和真实使用场景一样，永远不用未来预测过去。
    """
    search = GridSearchCV(model, grid, cv=CV, scoring="accuracy", refit=True)
    search.fit(X_train, y_train)
    return search


def cv_table(search):
    """把每组参数的交叉验证平均准确率整理成表，按得分排序。"""
    res = pd.DataFrame(search.cv_results_)
    cols = [c for c in res.columns if c.startswith("param_")] + ["mean_test_score", "std_test_score"]
    res = res[cols].sort_values("mean_test_score", ascending=False).reset_index(drop=True)
    # None 在表里会显示成 NaN，转成文字更清楚
    return res.astype({c: str for c in cols if c.startswith("param_")})


def always_up_cv_accuracy(y_train):
    """同样的 5 个验证折上，"永远预测涨"能拿多少准确率——模型在训练集内部也得先赢它。"""
    return [float(y_train[val].mean()) for _, val in CV.split(y_train)]


def run_experiment(data, feat_cols, label_col="Label", name="模型"):
    """一次完整实验：按日期切分 → 训练集内调参 → 测试集只评估一次。

    data：每行一天，含 Date、特征列、目标列。返回一行结果（dict）和训练好的搜索对象。
    阶段 3、4 要用不同特征 / 不同目标重复同一流程，所以包成一个函数，保证每次做法完全一样。
    """
    from src.data_utils import split_by_date
    from src.evaluation import evaluate

    train, test = split_by_date(data)
    X_train, y_train = train[feat_cols].to_numpy(), train[label_col].to_numpy().astype(int)
    X_test, y_test = test[feat_cols].to_numpy(), test[label_col].to_numpy().astype(int)

    search = tune(make_logreg(), LOGREG_GRID, X_train, y_train)
    pred = search.best_estimator_.predict(X_test)
    row = evaluate(y_test, pred, name)
    row.update({
        "cv_accuracy": search.best_score_,
        "cv_always_up": sum(always_up_cv_accuracy(y_train)) / CV.n_splits,
        "test_always_up": y_test.mean(),
        "pred_up_share": pred.mean(),
        "best_C": search.best_params_["logisticregression__C"],
        "class_weight": str(search.best_params_["logisticregression__class_weight"]),
    })
    return row, search
