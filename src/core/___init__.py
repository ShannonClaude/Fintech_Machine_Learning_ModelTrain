# src/core/__init__.py
# 银行信用风险评估 — 核心模型包
# 每个模块均可作为独立脚本运行，也可从此处导入关键函数供 Stacking 复用。

from .XGBoost          import train_model as train_xgboost,          preprocess, detect_label_column, load_data
from .LightGBM         import train_model as train_lightgbm
from .CatBoost         import train_model as train_catboost
from .RandomForest     import train_model as train_random_forest
from .SVM              import train_model as train_svm
from .LogisticRegression import train_model as train_logistic_regression
from .NaiveBayes       import train_model as train_naive_bayes

__all__ = [
    "load_data",
    "detect_label_column",
    "preprocess",
    "train_xgboost",
    "train_lightgbm",
    "train_catboost",
    "train_random_forest",
    "train_svm",
    "train_logistic_regression",
    "train_naive_bayes",
]
