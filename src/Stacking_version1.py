# =============================================================================
# 毕业设计：基于机器学习的银行客户信用风险评估模型研究与实现
# 主实验入口：多方案对比 —— 单模型基线 + Stacking 集成优化
#
# 实验方案：
#   EXP-1  : 7 个单体模型各自独立运行（作为基线）
#             → LogisticRegression / NaiveBayes / SVM /
#               RandomForest / XGBoost / LightGBM / CatBoost
#   EXP-2  : Stacking [XGB+LGBM+RF] → LR
#   EXP-3  : Stacking [XGB+LGBM+RF+SVM+NB] → LR（增加多样性）
#   EXP-4  : Stacking [XGB+LGBM+RF+SVM+NB] → XGB（升级元学习器）
#   EXP-5  : Stacking [XGB+LGBM+RF+SVM+NB] → XGB + passthrough
#   EXP-6  : 最优 Stacking 方案 + PR 曲线最优阈值调优
# =============================================================================

import os
import sys
import importlib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    recall_score, f1_score, precision_score,
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, precision_recall_curve,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from datetime import datetime

warnings.filterwarnings('ignore')

# ── 将 src/ 加入 sys.path，使 core 包可被 import ──────────────────────────────
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# ── GPU 加速检测 ───────────────────────────────────────────────────────────────
def gpu_capability():
    """Return dicts of GPU params for XGBoost and LightGBM."""
    gpu_available = True
    print("[GPU] GPU acceleration is enabled (manually confirmed).")
    xgb_params  = {'tree_method': 'hist', 'device': 'cuda'} if gpu_available else {}
    lgbm_params = {'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0} if gpu_available else {}
    return xgb_params, lgbm_params

XGB_GPU_PARAMS, LGBM_GPU_PARAMS = gpu_capability()

# ── 中文字体支持 ───────────────────────────────────────────────────────────────
_zh_fonts = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Noto Sans CJK SC']
_font_found = False
for _f in _zh_fonts:
    if any(_f.lower() in fp.name.lower() for fp in fm.fontManager.ttflist):
        plt.rcParams['font.sans-serif'] = [_f]
        plt.rcParams['axes.unicode_minus'] = False
        _font_found = True
        break
if not _font_found:
    print("[提示] 未找到中文字体，图表标签将使用英文。")

RANDOM_SEED = 42

# =============================================================================
# 0. 路径辅助
#    本脚本位于 src/，因此：
#      输入目录         → ../data/input
#      所有输出         → ../data/output/
#         - 单模型基线  → ../data/output/<ModelName>/<数据集名>/
#         - Stacking    → ../data/output/stacking/<时间戳>/
# =============================================================================
def get_input_file_and_output_dir(
    base_input_dir=os.path.join("..", "data", "input"),
    base_output_dir=os.path.join("..", "..", "data", "output"),
):
    while True:
        user_input = input("请输入数据文件名或完整路径：").strip()
        if not user_input:
            print("输入不能为空，请重新输入。")
            continue

        if os.path.exists(user_input):
            file_path     = user_input
            basename      = os.path.basename(file_path)
            filename_base = os.path.splitext(basename)[0]
        else:
            basename  = os.path.basename(user_input)
            filename  = basename if os.path.splitext(basename)[1] else basename + ".csv"
            file_path = os.path.join(base_input_dir, filename)
            if not os.path.exists(file_path):
                print(f"错误：文件 {file_path} 不存在，请重新输入。")
                continue
            filename_base = os.path.splitext(filename)[0]

        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Stacking 总输出目录：../data/output/stacking/<timestamp>/
        stacking_output_dir = os.path.join(base_output_dir, "stacking", timestamp)
        return file_path, stacking_output_dir, filename_base   # 多返回 filename_base 供单模型路径用


# =============================================================================
# 1. 数据加载
# =============================================================================
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[错误] 找不到数据文件：{os.path.abspath(path)}")
        sys.exit(1)
    df = pd.read_csv(path)
    print(f"[数据加载] 共 {df.shape[0]} 行 × {df.shape[1]} 列。")
    return df


# =============================================================================
# 2. 标签列识别
# =============================================================================
def detect_label_column(df: pd.DataFrame) -> str:
    candidates = ['Risk', 'risk', 'class', 'Class', 'default', 'Default',
                  'label', 'Label', 'target', 'Target']
    for col in candidates:
        if col in df.columns:
            print(f"[标签识别] 检测到标签列：'{col}'")
            return col
    last_col = df.columns[-1]
    print(f"[标签识别] 使用最后一列：'{last_col}'")
    return last_col


# =============================================================================
# 3. 数据预处理
# =============================================================================
def preprocess(df: pd.DataFrame, label_col: str):
    drop_cols = [c for c in df.columns if c.lower().startswith('unnamed')]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    X     = df.drop(columns=[label_col]).copy()
    y_raw = df[label_col].copy()

    for col in X.select_dtypes(include=[np.number]).columns:
        if X[col].isna().sum() > 0:
            X[col].fillna(X[col].median(), inplace=True)
    for col in X.select_dtypes(include=['object']).columns:
        if X[col].isna().sum() > 0:
            X[col].fillna(X[col].mode()[0], inplace=True)

    le = LabelEncoder()
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = le.fit_transform(X[col].astype(str))

    le_label      = LabelEncoder()
    y_encoded     = le_label.fit_transform(y_raw.astype(str))
    label_classes = le_label.classes_
    if len(label_classes) == 2:
        if label_classes[1].lower() in ['good', '0', 'no', 'safe']:
            y_encoded = 1 - y_encoded

    y = pd.Series(y_encoded, name=label_col)
    print(f"[预处理完成] 特征维度：{X.shape}，正样本（坏客户）占比：{y.mean():.2%}")
    return X, y, X.columns.tolist()


# =============================================================================
# 4. EXP-1 : 单模型基线 —— 调用 core/ 中的 7 个模型逐一运行
#
#    每个模型的图片输出到 ../data/output/<ModelName>/<dataset>/
#    与直接单独运行 core/<Model>.py 产生的目录完全一致，
#    方便后续 reporters/summarize.py 统一汇总。
#    各模型指标收集到 results 字典供最终对比表使用。
# =============================================================================
def run_exp1_all_single_models(
    X_train, X_test, y_train, y_test,
    dataset_name: str,
    base_output_root: str,      # ../data/output  绝对路径
) -> dict:
    """
    逐一训练并评估 core/ 中的 7 个单体模型。
    返回 {'EXP-1 <ModelName>': metrics_dict, ...}
    """

    # (显示名, core 模块名)
    MODEL_LIST = [
        ('LogisticRegression', 'LogisticRegression'),
        ('NaiveBayes',         'NaiveBayes'),
        ('SVM',                'SVM'),
        ('RandomForest',       'RandomForest'),
        ('XGBoost',            'XGBoost'),
        ('LightGBM',           'LightGBM'),
        ('CatBoost',           'CatBoost'),
    ]

    results = {}

    for display_name, module_name in MODEL_LIST:
        print(f"\n  ── {display_name} ──")

        # 输出目录与单独运行该脚本时完全一致
        output_dir = os.path.join(base_output_root, display_name, dataset_name)
        os.makedirs(output_dir, exist_ok=True)

        try:
            # 动态 import core 模块，注入全局 RANDOM_SEED
            mod = importlib.import_module(f'core.{module_name}')
            mod.RANDOM_SEED = RANDOM_SEED

            # ── 训练 ──────────────────────────────────────────────────────
            # SVM / LogisticRegression 返回 (model, scaler)，其余只返回 model
            needs_scaler = display_name in ('SVM', 'LogisticRegression')

            if display_name == 'CatBoost':
                # 预处理已做 LabelEncoding，传空 cat_feature_indices 即可
                model  = mod.train_model(X_train, y_train, [])
                scaler = None
            elif needs_scaler:
                model, scaler = mod.train_model(X_train, y_train)
            else:
                model  = mod.train_model(X_train, y_train)
                scaler = None

            # ── 评估 & 保存图片 ───────────────────────────────────────────
            if needs_scaler:
                auc = mod.evaluate_model(model, scaler, X_test, y_test, output_dir)
            else:
                auc = mod.evaluate_model(model, X_test, y_test, output_dir)

            # ── 特征重要性 ────────────────────────────────────────────────
            try:
                if display_name == 'SVM':
                    # SVM 的签名：(model, scaler, X_train, y_train, feature_names, output_dir)
                    mod.plot_feature_importance(
                        model, scaler, X_train, y_train,
                        X_train.columns.tolist(), output_dir
                    )
                elif display_name in ('LogisticRegression', 'NaiveBayes'):
                    # 签名：(model, feature_names, output_dir)
                    mod.plot_feature_importance(
                        model, X_train.columns.tolist(), output_dir
                    )
                elif display_name == 'CatBoost':
                    # 签名：(model, feature_names, output_dir)
                    mod.plot_feature_importance(
                        model, X_train.columns.tolist(), output_dir
                    )
                elif display_name == 'RandomForest':
                    # 签名：(model, feature_names, output_dir)
                    mod.plot_feature_importance(
                        model, X_train.columns.tolist(), output_dir
                    )
                else:
                    # XGBoost / LightGBM 签名：(model, output_dir)
                    mod.plot_feature_importance(model, output_dir)
            except Exception as e:
                print(f"  [警告] 特征重要性图生成失败：{e}")

            # ── 计算完整指标 ──────────────────────────────────────────────
            # 对需要 scaler 的模型在 scaled 数据上计算 predict_proba
            if needs_scaler:
                X_test_for_pred = scaler.transform(X_test)
                y_proba = model.predict_proba(X_test_for_pred)[:, 1]
            else:
                y_proba = model.predict_proba(X_test)[:, 1]

            y_pred      = (y_proba >= 0.5).astype(int)
            fpr, tpr, _ = roc_curve(y_test, y_proba)

            metrics = {
                'accuracy':  accuracy_score(y_test, y_pred),
                'auc':       roc_auc_score(y_test, y_proba),
                'recall':    recall_score(y_test, y_pred, zero_division=0),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'f1':        f1_score(y_test, y_pred, zero_division=0),
                'threshold': 0.5,
                'fpr':       fpr,
                'tpr':       tpr,
                'y_pred':    y_pred,
                'y_proba':   y_proba,
            }
            results[f'EXP-1 {display_name}'] = metrics
            print(f"  ✅ AUC={metrics['auc']:.4f}  Acc={metrics['accuracy']:.4f}  "
                  f"Recall={metrics['recall']:.4f}  F1={metrics['f1']:.4f}")
            print(f"  📁 → {output_dir}/")

        except Exception as e:
            print(f"  ❌ {display_name} 运行失败：{e}")
            import traceback; traceback.print_exc()

    return results


# =============================================================================
# 5. Stacking 基础学习器工厂
# =============================================================================
def build_base_learners(scale_pos_weight: float) -> dict:

    def make_xgb():
        return XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_SEED, eval_metric='auc',
            use_label_encoder=False, verbosity=0,
            **XGB_GPU_PARAMS,
        )

    def make_lgbm():
        return LGBMClassifier(
            n_estimators=200, max_depth=-1, learning_rate=0.05,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_SEED, verbosity=-1,
            **LGBM_GPU_PARAMS,
        )

    def make_rf():
        return RandomForestClassifier(
            n_estimators=200, max_depth=None, max_features='sqrt',
            min_samples_leaf=5, class_weight='balanced',
            n_jobs=-1, random_state=RANDOM_SEED,
        )

    def make_svm():
        return Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(
                kernel='rbf', C=1.0, gamma='scale',
                probability=True, class_weight='balanced',
                random_state=RANDOM_SEED,
            ))
        ])

    def make_nb():
        return GaussianNB(var_smoothing=1e-9)

    return {
        'xgboost':      make_xgb,
        'lightgbm':     make_lgbm,
        'randomforest': make_rf,
        'svm':          make_svm,
        'naivebayes':   make_nb,
    }


# =============================================================================
# 6. 运行单次 Stacking 实验
# =============================================================================
def run_stacking_experiment(
    exp_name: str, base_names: list, meta_type: str,
    passthrough: bool, X_train, y_train, scale_pos_weight: float,
) -> StackingClassifier:

    factories  = build_base_learners(scale_pos_weight)
    estimators = [(name, factories[name]()) for name in base_names]

    if meta_type == 'lr':
        meta = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(
                C=1.0, penalty='l2', solver='lbfgs',
                max_iter=1000, class_weight='balanced',
                random_state=RANDOM_SEED,
            ))
        ])
        meta_label = 'LogisticRegression'
    elif meta_type == 'xgb':
        meta = XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.8, scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_SEED, eval_metric='auc',
            use_label_encoder=False, verbosity=0,
            **XGB_GPU_PARAMS,
        )
        meta_label = 'XGBoost'
    elif meta_type == 'lgbm':
        meta = LGBMClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            num_leaves=15, scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_SEED, verbosity=-1,
            **LGBM_GPU_PARAMS,
        )
        meta_label = 'LightGBM'
    else:
        raise ValueError(f"Unknown meta_type: {meta_type}")

    model = StackingClassifier(
        estimators=estimators,
        final_estimator=meta,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
        stack_method='predict_proba',
        passthrough=passthrough,
        n_jobs=-1, verbose=0,
    )

    pt_str = ' + 保留原始特征' if passthrough else ''
    print(f"  → 基学习器 ({len(base_names)}): {base_names}")
    print(f"  → 元学习器: {meta_label}  passthrough={passthrough}{pt_str}")
    model.fit(X_train, y_train)
    return model


# =============================================================================
# 7. 评估指标计算
# =============================================================================
def calc_metrics(model, X_test, y_test, threshold: float = 0.5) -> dict:
    y_proba     = model.predict_proba(X_test)[:, 1]
    y_pred      = (y_proba >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    return {
        'accuracy':  accuracy_score(y_test, y_pred),
        'auc':       roc_auc_score(y_test, y_proba),
        'recall':    recall_score(y_test, y_pred, zero_division=0),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'f1':        f1_score(y_test, y_pred, zero_division=0),
        'threshold': threshold,
        'fpr':       fpr,
        'tpr':       tpr,
        'y_pred':    y_pred,
        'y_proba':   y_proba,
    }


# =============================================================================
# 8. 最优阈值搜索
# =============================================================================
def find_best_threshold(y_test, y_proba, strategy: str = 'f1') -> tuple:
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    if strategy == 'f1':
        f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
        best_idx  = np.argmax(f1_scores[:-1])
        best_thr  = thresholds[best_idx]
        print(f"  [最优阈值-F1策略]     threshold={best_thr:.4f}, F1={f1_scores[best_idx]:.4f}")
    else:
        mask     = precisions[:-1] >= 0.5
        best_idx = np.where(mask, recalls[:-1], -1).argmax() if mask.any() else np.argmax(recalls[:-1])
        best_thr = thresholds[best_idx]
        print(f"  [最优阈值-Recall策略] threshold={best_thr:.4f}, Recall={recalls[best_idx]:.4f}")
    return best_thr, precisions, recalls, thresholds


# =============================================================================
# 9. 为单个实验保存输出（混淆矩阵、ROC、分类报告、指标CSV）
# =============================================================================
def save_experiment_outputs(exp_name: str, y_test, y_pred, y_proba, output_dir: str):
    """保存单个实验的评估结果到指定文件夹。"""
    os.makedirs(output_dir, exist_ok=True)

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['好客户 (0)', '坏客户 (1)'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'混淆矩阵 - {exp_name}', fontsize=12, fontweight='bold')
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    fig.tight_layout()
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    # ROC 曲线
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_val = roc_auc_score(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {auc_val:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC 曲线 - {exp_name}')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    roc_path = os.path.join(output_dir, 'roc_curve.png')
    fig.tight_layout()
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)

    # 分类报告
    report = classification_report(y_test, y_pred, target_names=['好客户', '坏客户'], digits=4)
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(f"分类报告 - {exp_name}\n")
        f.write("="*60 + "\n")
        f.write(report)

    # 指标 CSV
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc': auc_val,
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
    }
    pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, 'metrics.csv'), index=False, encoding='utf-8-sig')

    print(f"  📁 实验输出已保存到：{output_dir}")


# =============================================================================
# 10. 绘图函数（综合对比图）
# =============================================================================
# EXP-1 单模型颜色（浅色系，区分 7 个模型）
_EXP1_COLORS = {
    'EXP-1 LogisticRegression': '#aec6cf',
    'EXP-1 NaiveBayes':         '#b5ead7',
    'EXP-1 SVM':                '#c7a6d4',
    'EXP-1 RandomForest':       '#f7c6a0',
    'EXP-1 XGBoost':            '#a0c4ff',
    'EXP-1 LightGBM':           '#caffbf',
    'EXP-1 CatBoost':           '#ffd6a5',
}
# Stacking 方案颜色（鲜艳粗线）
_STACKING_PALETTE = {
    'EXP-2 Stacking[3 base+LR]':               ('#e41a1c', '-',  2.0),
    'EXP-3 Stacking[5 base+LR]':               ('#377eb8', '-',  2.0),
    'EXP-4 Stacking[5 base+XGB]':              ('#4daf4a', '-',  2.0),
    'EXP-5 Stacking[5 base+XGB+passthrough]':  ('#ff7f00', '-',  2.2),
    'EXP-6 Optimal threshold (based on EXP-5)':('#984ea3', ':',  2.0),
}


def plot_roc_all(results: dict, output_dir: str):
    """全部实验 ROC 曲线：EXP-1 单模型浅色细虚线，Stacking 彩色粗实线。"""
    fig, ax = plt.subplots(figsize=(11, 8))

    # EXP-1 单模型（浅色细线）
    for name, m in results.items():
        if name.startswith('EXP-1'):
            color = _EXP1_COLORS.get(name, '#cccccc')
            ax.plot(m['fpr'], m['tpr'], color=color, lw=1.3, linestyle='--',
                    alpha=0.80, label=f"{name}  (AUC={m['auc']:.4f})")

    # Stacking 方案（彩色粗线）
    for name, m in results.items():
        if not name.startswith('EXP-1'):
            color, ls, lw = _STACKING_PALETTE.get(name, ('gray', '-', 1.5))
            ax.plot(m['fpr'], m['tpr'], color=color, linestyle=ls, lw=lw,
                    label=f"{name}  (AUC={m['auc']:.4f})")

    ax.plot([0, 1], [0, 1], 'lightgray', lw=1.0, linestyle=':', label='Random Classifier')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('ROC 曲线对比\n（单模型基线 EXP-1  vs  Stacking 集成 EXP-2~6）',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=7.5, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.25)

    path = os.path.join(output_dir, 'roc_all_experiments.png')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[图表已保存] 总 ROC 对比图   → {path}")


def plot_metrics_bar(results: dict, output_dir: str):
    """
    分两个子图：
      左：EXP-1 全部单模型指标对比
      右：EXP-2~6 Stacking 方案指标对比
    """
    exp1_names     = [n for n in results if n.startswith('EXP-1')]
    stacking_names = [n for n in results if not n.startswith('EXP-1')]
    metrics        = ['auc', 'accuracy', 'recall', 'f1']
    metric_labels  = ['AUC', 'Accuracy', 'Recall', 'F1-Score']
    colors         = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    width          = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(20, 6), sharey=True)

    for ax, group_names, title in [
        (axes[0], exp1_names,     'EXP-1 单模型基线对比'),
        (axes[1], stacking_names, 'EXP-2~6 Stacking 方案对比'),
    ]:
        short = [n.replace('EXP-1 ', '').replace('EXP-', 'E')
                  .replace('Stacking', 'STK').replace('passthrough', 'PT')
                  for n in group_names]
        x = np.arange(len(group_names))
        for i, (met, lab, col) in enumerate(zip(metrics, metric_labels, colors)):
            vals = [results[n][met] for n in group_names]
            bars = ax.bar(x + i * width, vals, width, label=lab, color=col, alpha=0.85)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=90)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(short, rotation=25, ha='right', fontsize=9)
        ax.set_ylim([0.45, 1.08])
        ax.set_ylabel('指标值', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.25, axis='y')

    path = os.path.join(output_dir, 'metrics_bar_chart.png')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[图表已保存] 指标柱状图     → {path}")


def plot_best_confusion_matrix(y_test, y_pred, exp_name: str, output_dir: str):
    cm   = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['好客户 (0)', '坏客户 (1)']
    )
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'混淆矩阵 - {exp_name}', fontsize=12, fontweight='bold')
    path = os.path.join(output_dir, 'confusion_matrix_best.png')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[图表已保存] 最优混淆矩阵   → {path}")


def plot_pr_threshold(y_test, y_proba, best_thr_f1, best_thr_recall,
                      precisions, recalls, thresholds, output_dir: str):
    f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-8)
    fig = plt.figure(figsize=(13, 5))
    gs  = GridSpec(1, 2, figure=fig, wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(recalls, precisions, color='steelblue', lw=2, label='PR 曲线')
    idx_f1 = np.argmin(np.abs(thresholds - best_thr_f1))
    ax1.scatter(recalls[idx_f1], precisions[idx_f1], s=120, color='#d62728', zorder=5,
                label=f'F1 最优阈值 = {best_thr_f1:.3f}')
    idx_rc = np.argmin(np.abs(thresholds - best_thr_recall))
    ax1.scatter(recalls[idx_rc], precisions[idx_rc], s=120, color='#ff7f00',
                marker='^', zorder=5, label=f'Recall 最优阈值 = {best_thr_recall:.3f}')
    ax1.set_xlabel('Recall', fontsize=12); ax1.set_ylabel('Precision', fontsize=12)
    ax1.set_title('Precision-Recall 曲线\n（坏客户类别）', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(thresholds, precisions[:-1], label='Precision', color='#377eb8', lw=2)
    ax2.plot(thresholds, recalls[:-1],    label='Recall',    color='#e41a1c', lw=2)
    ax2.plot(thresholds, f1_scores,       label='F1-Score',  color='#4daf4a', lw=2, linestyle='--')
    ax2.axvline(x=best_thr_f1,     color='#d62728', lw=1.5, linestyle=':',
                label=f'F1 最优={best_thr_f1:.3f}')
    ax2.axvline(x=best_thr_recall, color='#ff7f00', lw=1.5, linestyle='-.',
                label=f'Recall 最优={best_thr_recall:.3f}')
    ax2.axvline(x=0.5,             color='gray',    lw=1.2, linestyle='--', label='默认阈值=0.500')
    ax2.set_xlabel('决策阈值', fontsize=12); ax2.set_ylabel('指标值', fontsize=12)
    ax2.set_title('阈值 vs Precision / Recall / F1\n（阈值权衡分析）',
                  fontsize=12, fontweight='bold')
    ax2.set_xlim([0.1, 0.9]); ax2.set_ylim([0.0, 1.05])
    ax2.legend(fontsize=8.5, loc='center left'); ax2.grid(True, alpha=0.3)

    path = os.path.join(output_dir, 'pr_curve_threshold.png')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[图表已保存] PR 曲线+阈值分析 → {path}")


# =============================================================================
# 11. 汇总打印与 CSV 保存（EXP-1 单模型先打印，再打印 Stacking）
# =============================================================================
def print_and_save_summary(results: dict, output_dir: str):
    print("\n" + "=" * 92)
    print("                              实验结果汇总")
    print("=" * 92)
    print(f"  {'实验名称':<44} {'AUC':>7} {'准确率':>7} {'召回率':>7} {'F1':>7} {'阈值':>7}")
    print("-" * 92)

    best_auc = max(m['auc'] for m in results.values())
    rows = []

    def _print_row(name, m):
        marker = " ★" if abs(m['auc'] - best_auc) < 1e-6 else "  "
        print(f"  {name:<44}{marker} {m['auc']:>6.4f}  {m['accuracy']:>6.4f}  "
              f"{m['recall']:>6.4f}  {m['f1']:>6.4f}  {m['threshold']:>6.3f}")
        rows.append({
            '实验名称': name, 'AUC': round(m['auc'], 4),
            '准确率': round(m['accuracy'], 4), '召回率': round(m['recall'], 4),
            '精确率': round(m['precision'], 4), 'F1-Score': round(m['f1'], 4),
            '阈值': round(m['threshold'], 3),
        })

    # EXP-1 单模型
    for name, m in results.items():
        if name.startswith('EXP-1'):
            _print_row(name, m)

    print("  " + "─ " * 46)   # 分隔线

    # Stacking 方案
    for name, m in results.items():
        if not name.startswith('EXP-1'):
            _print_row(name, m)

    print("=" * 92)
    print("★ 为所有方案中 AUC 最高者")

    csv_path = os.path.join(output_dir, 'experiment_summary.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n[文件已保存] 汇总表 → {csv_path}")

    return rows


# =============================================================================
# 主程序
# =============================================================================
def main():
    print("=" * 72)
    print("  银行客户信用风险评估 —— 多方案实验对比")
    print()
    print("  EXP-1 : 7 个单模型基线")
    print("          LR / NaiveBayes / SVM / RF / XGBoost / LightGBM / CatBoost")
    print("  EXP-2 : Stacking [XGB+LGBM+RF] → LR")
    print("  EXP-3 : Stacking [XGB+LGBM+RF+SVM+NB] → LR")
    print("  EXP-4 : Stacking [XGB+LGBM+RF+SVM+NB] → XGB")
    print("  EXP-5 : Stacking [XGB+LGBM+RF+SVM+NB] → XGB + passthrough")
    print("  EXP-6 : EXP-5 + PR 曲线最优阈值调优")
    print("=" * 72 + "\n")

    if XGB_GPU_PARAMS:
        print("[硬件] XGBoost / LightGBM GPU 加速已启用。")
    else:
        print("[硬件] 使用 CPU 运行（未检测到 GPU）。")

    # ── 获取路径 ──────────────────────────────────────────────────────────────
    data_path, stacking_output_dir, dataset_name = get_input_file_and_output_dir()
    os.makedirs(stacking_output_dir, exist_ok=True)

    # 单模型图片的根目录：统一放到 ../data/output/
    base_output_root = os.path.normpath(
        os.path.join(_SRC_DIR, "..", "data", "output")
    )

    # ── 数据准备（所有实验共享同一份划分）────────────────────────────────────
    df        = load_data(data_path)
    label_col = detect_label_column(df)
    X, y, _   = preprocess(df, label_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
    )
    print(f"\n[数据划分] 训练集：{len(X_train)} 条，测试集：{len(X_test)} 条\n")

    neg_cnt = (y_train == 0).sum()
    pos_cnt = (y_train == 1).sum()
    spw     = neg_cnt / pos_cnt if pos_cnt > 0 else 1.0
    print(f"[样本权重] scale_pos_weight = {spw:.2f}（好客户 {neg_cnt} : 坏客户 {pos_cnt}）\n")

    results = {}
    models  = {}
    base_3  = ['xgboost', 'lightgbm', 'randomforest']
    base_5  = ['xgboost', 'lightgbm', 'randomforest', 'svm', 'naivebayes']

    steps = [
        'EXP-1 全部单模型基线（7个）',
        'EXP-2 Stacking[3]+LR',
        'EXP-3 Stacking[5]+LR',
        'EXP-4 Stacking[5]+XGB',
        'EXP-5 Stacking[5]+XGB+PT',
        'EXP-6 阈值调优',
        '绘图',
        '汇总',
    ]

    with tqdm(total=len(steps), desc="流水线进度", dynamic_ncols=True, mininterval=0.2) as pbar:

        # ── EXP-1 : 7 个单模型基线 ───────────────────────────────────────────
        print(f"\n{'─'*65}")
        print(f"  EXP-1 : 单模型基线（7 个模型逐一运行）")
        print(f"  各模型图片 → data/output/<ModelName>/{dataset_name}/")
        print(f"{'─'*65}")
        exp1_results = run_exp1_all_single_models(
            X_train, X_test, y_train, y_test,
            dataset_name, base_output_root
        )
        results.update(exp1_results)
        pbar.update(1)

        # ── EXP-2 ────────────────────────────────────────────────────────────
        exp_key = 'EXP-2 Stacking[3 base+LR]'
        exp_dir = os.path.join(stacking_output_dir, 'EXP-2')
        print(f"\n{'─'*65}\n  {exp_key}\n{'─'*65}")
        m = run_stacking_experiment(exp_key, base_3, 'lr', False, X_train, y_train, spw)
        results[exp_key] = calc_metrics(m, X_test, y_test)
        models[exp_key]  = m
        print(f"  AUC = {results[exp_key]['auc']:.4f}")
        save_experiment_outputs(exp_key, y_test,
                                results[exp_key]['y_pred'],
                                results[exp_key]['y_proba'],
                                exp_dir)
        pbar.update(1)

        # ── EXP-3 ────────────────────────────────────────────────────────────
        exp_key = 'EXP-3 Stacking[5 base+LR]'
        exp_dir = os.path.join(stacking_output_dir, 'EXP-3')
        print(f"\n{'─'*65}\n  {exp_key}\n{'─'*65}")
        m = run_stacking_experiment(exp_key, base_5, 'lr', False, X_train, y_train, spw)
        results[exp_key] = calc_metrics(m, X_test, y_test)
        models[exp_key]  = m
        print(f"  AUC = {results[exp_key]['auc']:.4f}")
        save_experiment_outputs(exp_key, y_test,
                                results[exp_key]['y_pred'],
                                results[exp_key]['y_proba'],
                                exp_dir)
        pbar.update(1)

        # ── EXP-4 ────────────────────────────────────────────────────────────
        exp_key = 'EXP-4 Stacking[5 base+XGB]'
        exp_dir = os.path.join(stacking_output_dir, 'EXP-4')
        print(f"\n{'─'*65}\n  {exp_key}\n{'─'*65}")
        m = run_stacking_experiment(exp_key, base_5, 'xgb', False, X_train, y_train, spw)
        results[exp_key] = calc_metrics(m, X_test, y_test)
        models[exp_key]  = m
        print(f"  AUC = {results[exp_key]['auc']:.4f}")
        save_experiment_outputs(exp_key, y_test,
                                results[exp_key]['y_pred'],
                                results[exp_key]['y_proba'],
                                exp_dir)
        pbar.update(1)

        # ── EXP-5 ────────────────────────────────────────────────────────────
        exp_key = 'EXP-5 Stacking[5 base+XGB+passthrough]'
        exp_dir = os.path.join(stacking_output_dir, 'EXP-5')
        print(f"\n{'─'*65}\n  {exp_key}\n{'─'*65}")
        m = run_stacking_experiment(exp_key, base_5, 'xgb', True, X_train, y_train, spw)
        results[exp_key] = calc_metrics(m, X_test, y_test)
        models[exp_key]  = m
        print(f"  AUC = {results[exp_key]['auc']:.4f}")
        save_experiment_outputs(exp_key, y_test,
                                results[exp_key]['y_pred'],
                                results[exp_key]['y_proba'],
                                exp_dir)
        pbar.update(1)

        # ── EXP-6 : 阈值调优 ─────────────────────────────────────────────────
        stacking_only = {k: v for k, v in results.items()
                         if k.startswith('EXP-') and 'Stacking' in k}
        best_stk_key  = max(stacking_only, key=lambda k: stacking_only[k]['auc'])
        best_proba    = results[best_stk_key]['y_proba']
        print(f"\n{'─'*65}\n  EXP-6 阈值调优（基于 {best_stk_key}）\n{'─'*65}")

        thr_f1,    prec, rec, thrs = find_best_threshold(y_test, best_proba, 'f1')
        thr_recall, *_             = find_best_threshold(y_test, best_proba, 'recall')

        exp_key = 'EXP-6 Optimal threshold (based on EXP-5)'
        exp_dir = os.path.join(stacking_output_dir, 'EXP-6')
        results[exp_key] = calc_metrics(models[best_stk_key], X_test, y_test, threshold=thr_f1)
        print(f"  AUC={results[exp_key]['auc']:.4f}  "
              f"Recall={results[exp_key]['recall']:.4f}  "
              f"F1={results[exp_key]['f1']:.4f}")
        save_experiment_outputs(exp_key, y_test,
                                results[exp_key]['y_pred'],
                                results[exp_key]['y_proba'],
                                exp_dir)
        pbar.update(1)

        # ── 绘图（综合对比图） ─────────────────────────────────────────────────
        print(f"\n{'─'*65}\n  生成综合对比图表...\n{'─'*65}")
        plot_roc_all(results, stacking_output_dir)
        plot_metrics_bar(results, stacking_output_dir)
        best_key = max(results, key=lambda k: results[k]['auc'])
        plot_best_confusion_matrix(
            y_test, results[best_key]['y_pred'], best_key, stacking_output_dir
        )
        plot_pr_threshold(y_test, best_proba, thr_f1, thr_recall,
                          prec, rec, thrs, stacking_output_dir)
        pbar.update(1)

        # ── 汇总 ─────────────────────────────────────────────────────────────
        print_and_save_summary(results, stacking_output_dir)
        pbar.update(1)

    # ── 最终统计 ──────────────────────────────────────────────────────────────
    best_exp1_auc = max(v['auc'] for k, v in results.items() if k.startswith('EXP-1'))
    best_key      = max(results, key=lambda k: results[k]['auc'])
    best_auc      = results[best_key]['auc']
    delta         = (best_auc - best_exp1_auc) * 100

    print(f"\n{'=' * 72}")
    print(f"  ✅ 全部实验完成！")
    print(f"     EXP-1 单模型最高 AUC  = {best_exp1_auc:.4f}")
    print(f"     全局最优实验           : [{best_key}]")
    print(f"     全局最优 AUC           = {best_auc:.4f}  "
          f"({'↑' if delta >= 0 else '↓'}{abs(delta):.2f}% vs 单模型最优)")
    print(f"\n  📁 Stacking 实验输出：{stacking_output_dir}/")
    print(f"     ├── EXP-2/                      (各实验独立文件夹)")
    print(f"     ├── EXP-3/")
    print(f"     ├── EXP-4/")
    print(f"     ├── EXP-5/")
    print(f"     ├── EXP-6/")
    print(f"     ├── roc_all_experiments.png     总 ROC 对比曲线")
    print(f"     ├── metrics_bar_chart.png       单模型 vs Stacking 指标柱状图")
    print(f"     ├── confusion_matrix_best.png   最优方案混淆矩阵")
    print(f"     ├── pr_curve_threshold.png      PR 曲线 + 阈值分析")
    print(f"     └── experiment_summary.csv      数值汇总表")
    print(f"\n  📁 单模型图片：data/output/<ModelName>/{dataset_name}/")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()