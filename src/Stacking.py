# =============================================================================
# 毕业设计：基于机器学习的银行客户信用风险评估模型研究与实现
# 主实验入口：多方案对比 —— 单模型基线 + Stacking 集成 + 进阶优化
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  实验方案                                                               │
# │  EXP-1  : 8 个单体模型各自独立运行（作为基线）                          │
# │           LR / NaiveBayes / SVM / DecisionTree /                        │
# │           RandomForest / XGBoost / LightGBM / CatBoost                 │
# │  EXP-2  : Stacking [XGB+LGBM+RF] → LR                                 │
# │  EXP-3  : Stacking [XGB+LGBM+RF+SVM+NB] → LR（增加多样性）            │
# │  EXP-4  : Stacking [XGB+LGBM+RF+SVM+NB] → XGB（升级元学习器）         │
# │  EXP-5  : Stacking [XGB+LGBM+RF+SVM+NB] → XGB + passthrough           │
# │  EXP-6  : 最优 Stacking 方案 + PR 曲线最优阈值调优                     │
# │  EXP-7  : 进阶优化：SMOTE + 深度特征工程 + Bayesian 超参数优化（XGB）  │
# │  EXP-8  : 深度学习尝试：DNN (MLPClassifier)                            │
# │  Post   : KS/Lift 业务指标图 + SHAP 可解释性分析（基于最优树模型）      │
# └─────────────────────────────────────────────────────────────────────────┘
#
# 输出结构：
#   E:\BankCreditRisk\data\output\Stacking\{timestamp}\
#   ├── Chinese\
#   │   ├── EXP-1\{ModelName}\   (单模型输出)
#   │   ├── EXP-2\ ~ EXP-8\     (集成/优化实验)
#   │   ├── SHAP_analysis\
#   │   ├── roc_all_experiments.png
#   │   ├── metrics_bar_chart.png
#   │   ├── confusion_matrix_best.png
#   │   ├── pr_curve_threshold.png
#   │   ├── ks_lift_chart.png
#   │   └── experiment_summary.csv
#   └── English\
#       ├── EXP-1\{ModelName}\
#       ├── EXP-2\ ~ EXP-8\
#       └── (same charts in English)
#
# 依赖安装（可选组件）：
#   pip install shap optuna imbalanced-learn
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

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
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

# ── 可选依赖 ──────────────────────────────────────────────────────────────────
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("[提示] 未安装 shap，SHAP 分析将跳过。运行：pip install shap")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("[提示] 未安装 optuna，Bayesian 优化将跳过。运行：pip install optuna")

try:
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print("[提示] 未安装 imbalanced-learn，SMOTE 将跳过。运行：pip install imbalanced-learn")

# ── 将 src/ 加入 sys.path ──────────────────────────────────────────────────────
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# ── 固定输出根目录 ─────────────────────────────────────────────────────────────
_STACKING_BASE_OUTPUT = r'E:\BankCreditRisk\data\output\Stacking'
_BASE_INPUT           = r'E:\BankCreditRisk\data\input'

# ── GPU 加速检测 ───────────────────────────────────────────────────────────────
def gpu_capability():
    gpu_available = True
    print("[GPU] GPU acceleration is enabled (manually confirmed).")
    xgb_params  = {'tree_method': 'hist', 'device': 'cuda'} if gpu_available else {}
    lgbm_params = {'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0} if gpu_available else {}
    return xgb_params, lgbm_params

XGB_GPU_PARAMS, LGBM_GPU_PARAMS = gpu_capability()

# ── 中文字体 ───────────────────────────────────────────────────────────────────
_zh_fonts   = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Noto Sans CJK SC']
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
# 国际化：翻译字典（中英双语）
# =============================================================================
_T = {
    # ── 中文 ──────────────────────────────────────────────────────────────────
    'zh': {
        'good_lbl':        '好客户 (0)',
        'bad_lbl':         '坏客户 (1)',
        'cm_title':        '混淆矩阵',
        'roc_title':       'ROC 曲线对比\n（单模型基线 EXP-1  vs  集成/优化方案 EXP-2~8）',
        'fpr':             'False Positive Rate',
        'tpr':             'True Positive Rate',
        'random_clf':      '随机分类器',
        'exp1_title':      'EXP-1 单模型基线对比（8 模型）',
        'exp28_title':     'EXP-2~8 集成与优化方案对比',
        'metric_val':      '指标值',
        'pr_title':        'Precision-Recall 曲线\n（坏客户类别）',
        'thr_title':       '阈值 vs Precision / Recall / F1\n（阈值权衡分析）',
        'decision_thr':    '决策阈值',
        'f1_best_thr':     'F1 最优阈值',
        'recall_best_thr': 'Recall 最优阈值',
        'default_thr':     '默认阈值=0.500',
        'f1_best_lbl':     'F1 最优',
        'recall_best_lbl': 'Recall 最优',
        'ks_title':        'KS 曲线对比\n（KS 统计量 = max|TPR - FPR|）',
        'lift_title':      '累积提升图（Lift Chart）\n（坏客户识别效率 vs 随机）',
        'sample_pct_ks':   '样本占比（按预测概率降序）',
        'cum_pn':          '累积正/负样本比例',
        'sample_pct2':     '样本占比',
        'lift_factor':     '提升倍数 (Lift)',
        'random_model':    '随机模型',
        'random_baseline': '随机基线 (Lift=1)',
        'dnn_title':       'DNN 训练过程',
        'train_loss':      '训练损失',
        'val_auc':         '验证 AUC',
        'iteration':       '迭代轮次',
        'pr_curve_lbl':    'PR 曲线',
        'cls_report_hdr':  '分类报告',
        'cls_tgt_good':    '好客户',
        'cls_tgt_bad':     '坏客户',
    },
    # ── 英文 ──────────────────────────────────────────────────────────────────
    'en': {
        'good_lbl':        'Good (0)',
        'bad_lbl':         'Bad (1)',
        'cm_title':        'Confusion Matrix',
        'roc_title':       'ROC Curve Comparison\n(Single Model Baseline EXP-1  vs  Ensemble/Optimized EXP-2~8)',
        'fpr':             'False Positive Rate',
        'tpr':             'True Positive Rate',
        'random_clf':      'Random Classifier',
        'exp1_title':      'EXP-1 Single Model Baseline (8 Models)',
        'exp28_title':     'EXP-2~8 Ensemble & Optimization Comparison',
        'metric_val':      'Metric Value',
        'pr_title':        'Precision-Recall Curve\n(Bad Customer Class)',
        'thr_title':       'Threshold vs Precision / Recall / F1\n(Threshold Trade-off Analysis)',
        'decision_thr':    'Decision Threshold',
        'f1_best_thr':     'Best F1 Threshold',
        'recall_best_thr': 'Best Recall Threshold',
        'default_thr':     'Default Threshold=0.500',
        'f1_best_lbl':     'Best F1',
        'recall_best_lbl': 'Best Recall',
        'ks_title':        'KS Curve Comparison\n(KS Statistic = max|TPR - FPR|)',
        'lift_title':      'Cumulative Lift Chart\n(Bad Customer Detection vs Random)',
        'sample_pct_ks':   'Sample Proportion (Sorted by Predicted Prob Desc)',
        'cum_pn':          'Cumulative Positive/Negative Rate',
        'sample_pct2':     'Sample Proportion',
        'lift_factor':     'Lift Factor',
        'random_model':    'Random Model',
        'random_baseline': 'Random Baseline (Lift=1)',
        'dnn_title':       'DNN Training Progress',
        'train_loss':      'Training Loss',
        'val_auc':         'Validation AUC',
        'iteration':       'Iterations',
        'pr_curve_lbl':    'PR Curve',
        'cls_report_hdr':  'Classification Report',
        'cls_tgt_good':    'GoodCustomer',
        'cls_tgt_bad':     'BadCustomer',
    },
}


# =============================================================================
# 0. 路径辅助
# =============================================================================
def get_input_file_and_output_dir():
    """
    询问数据文件路径；固定输出到
      E:\\BankCreditRisk\\data\\output\\Stacking\\{timestamp}
    返回 (file_path, stacking_output_dir, filename_base)
    """
    while True:
        user_input = input("请输入数据文件名或完整路径：").strip()
        if not user_input:
            print("输入不能为空，请重新输入。")
            continue
        if os.path.exists(user_input):
            file_path     = user_input
            filename_base = os.path.splitext(os.path.basename(file_path))[0]
        else:
            basename  = os.path.basename(user_input)
            filename  = basename if os.path.splitext(basename)[1] else basename + ".csv"
            file_path = os.path.join(_BASE_INPUT, filename)
            if not os.path.exists(file_path):
                print(f"错误：文件 {file_path} 不存在，请重新输入。")
                continue
            filename_base = os.path.splitext(filename)[0]

        timestamp           = datetime.now().strftime("%Y%m%d_%H%M%S")
        stacking_output_dir = os.path.join(_STACKING_BASE_OUTPUT, timestamp)
        return file_path, stacking_output_dir, filename_base


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
    candidates = ['Risk','risk','class','Class','default','Default',
                  'label','Label','target','Target']
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
# 4. 深度特征工程（EXP-7 专用，基于训练集统计量，避免数据泄露）
# =============================================================================
def deep_feature_engineering(X_train: pd.DataFrame,
                              X_test: pd.DataFrame,
                              top_n_interact: int = 5):
    """
    三类特征增强：
    ① Log1p 变换    — 针对右偏分布的数值特征（偏度 > 1）
    ② 比率交互特征  — Top-N 重要特征两两之间的比值（基于方差排序）
    ③ 分箱统计特征  — 将数值分为 5 个分位分箱，编码为序数
    所有统计量均 fit 于 X_train，apply 于 X_test，防止数据泄露。
    返回 (X_train_new, X_test_new, new_feature_names)
    """
    X_tr = X_train.copy()
    X_te = X_test.copy()

    numeric_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()

    # ── ① Log1p 变换（偏度 > 1 的列）──────────────────────────────────────
    log_cols = []
    for col in numeric_cols:
        if X_tr[col].min() >= 0 and X_tr[col].skew() > 1.0:
            new_col = f'{col}_log1p'
            X_tr[new_col] = np.log1p(X_tr[col])
            X_te[new_col] = np.log1p(X_te[col])
            log_cols.append(new_col)
    if log_cols:
        print(f"  [FE] Log1p 变换：{len(log_cols)} 列")

    # ── ② Top-N 列两两比率交互 ──────────────────────────────────────────────
    var_series = X_tr[numeric_cols].var().sort_values(ascending=False)
    top_cols   = var_series.index[:top_n_interact].tolist()
    ratio_cols = []
    for i, c1 in enumerate(top_cols):
        for c2 in top_cols[i+1:]:
            new_col  = f'{c1}_div_{c2}'
            denom_min = X_tr[c2].abs().quantile(0.05) + 1e-8
            X_tr[new_col] = X_tr[c1] / (X_tr[c2].abs() + denom_min)
            X_te[new_col] = X_te[c1] / (X_te[c2].abs() + denom_min)
            ratio_cols.append(new_col)
    if ratio_cols:
        print(f"  [FE] 比率交互特征：{len(ratio_cols)} 列（Top-{top_n_interact} × 两两）")

    # ── ③ 分位分箱（5 箱序数编码）──────────────────────────────────────────
    bin_cols = []
    for col in numeric_cols[:10]:
        bin_edges = X_tr[col].quantile([0, .2, .4, .6, .8, 1.0]).values
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 3:
            continue
        new_col = f'{col}_qbin'
        X_tr[new_col] = np.searchsorted(bin_edges[1:-1], X_tr[col].values).astype(float)
        X_te[new_col] = np.searchsorted(bin_edges[1:-1], X_te[col].values).astype(float)
        bin_cols.append(new_col)
    if bin_cols:
        print(f"  [FE] 分位分箱特征：{len(bin_cols)} 列")

    total_new = len(log_cols) + len(ratio_cols) + len(bin_cols)
    print(f"  [FE] 特征维度：{X_train.shape[1]} → {X_tr.shape[1]}（新增 {total_new} 列）")
    return X_tr, X_te, X_tr.columns.tolist()


# =============================================================================
# 5. EXP-1 : 单模型基线（8 个，含决策树）
# =============================================================================
def run_exp1_all_single_models(
    X_train, X_test, y_train, y_test,
    dataset_name: str,
    zh_exp1_root: str,
    en_exp1_root: str,
) -> dict:
    """
    逐一训练并评估 core/ 中的 8 个单体模型。
    Chinese 图表由 core 模块输出到 zh_exp1_root/{ModelName}/
    English 图表由本函数补充输出到 en_exp1_root/{ModelName}/
    返回 {'EXP-1 <ModelName>': metrics_dict, ...}
    """
    MODEL_LIST = [
        ('LogisticRegression', 'LogisticRegression'),
        ('NaiveBayes',         'NaiveBayes'),
        ('SVM',                'SVM'),
        ('DecisionTree',       'DecisionTree'),
        ('RandomForest',       'RandomForest'),
        ('XGBoost',            'XGBoost'),
        ('LightGBM',           'LightGBM'),
        ('CatBoost',           'CatBoost'),
    ]
    results = {}

    for display_name, module_name in MODEL_LIST:
        print(f"\n  ── {display_name} ──")
        # Chinese: external core module writes here
        zh_model_dir = os.path.join(zh_exp1_root, display_name)
        # English: we write bilingual key plots here
        en_model_dir = os.path.join(en_exp1_root, display_name)
        os.makedirs(zh_model_dir, exist_ok=True)
        os.makedirs(en_model_dir, exist_ok=True)

        try:
            mod = importlib.import_module(f'core.{module_name}')
            mod.RANDOM_SEED = RANDOM_SEED

            # ── 训练 ────────────────────────────────────────────────────────
            needs_scaler = display_name in ('SVM', 'LogisticRegression')
            if display_name == 'CatBoost':
                model  = mod.train_model(X_train, y_train, [])
                scaler = None
            elif needs_scaler:
                model, scaler = mod.train_model(X_train, y_train)
            else:
                model  = mod.train_model(X_train, y_train)
                scaler = None

            # ── 评估（core 模块生成 Chinese 图到 zh_model_dir）──────────────
            if needs_scaler:
                auc = mod.evaluate_model(model, scaler, X_test, y_test, zh_model_dir)
            else:
                auc = mod.evaluate_model(model, X_test, y_test, zh_model_dir)

            # ── 特征重要性（Chinese，输出到 zh_model_dir）──────────────────
            try:
                if display_name == 'SVM':
                    mod.plot_feature_importance(
                        model, scaler, X_train, y_train,
                        X_train.columns.tolist(), zh_model_dir)
                elif display_name in ('LogisticRegression', 'NaiveBayes'):
                    mod.plot_feature_importance(
                        model, X_train.columns.tolist(), zh_model_dir)
                elif display_name in ('CatBoost', 'RandomForest', 'DecisionTree'):
                    mod.plot_feature_importance(
                        model, X_train.columns.tolist(), zh_model_dir)
                else:  # XGBoost / LightGBM
                    mod.plot_feature_importance(model, zh_model_dir)
            except Exception as e:
                print(f"  [警告] 特征重要性图生成失败：{e}")

            # ── 计算完整指标 ─────────────────────────────────────────────────
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
                'model':     model,
                'scaler':    scaler,
            }
            results[f'EXP-1 {display_name}'] = metrics
            print(f"  ✅ AUC={metrics['auc']:.4f}  Acc={metrics['accuracy']:.4f}  "
                  f"Recall={metrics['recall']:.4f}  F1={metrics['f1']:.4f}")

            # ── English 版本（本地生成关键图）──────────────────────────────
            exp_key_en = f'EXP-1 {display_name}'
            save_experiment_outputs(exp_key_en, y_test, y_pred, y_proba,
                                    en_model_dir, lang='en')
            print(f"  📁 ZH → {zh_model_dir}/")
            print(f"  📁 EN → {en_model_dir}/")

        except Exception as e:
            print(f"  ❌ {display_name} 运行失败：{e}")
            import traceback; traceback.print_exc()

    return results


# =============================================================================
# 6. Stacking 基础学习器工厂（含决策树）
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

    def make_dt():
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=10, criterion='gini',
            class_weight='balanced', random_state=RANDOM_SEED,
        )

    return {
        'xgboost':      make_xgb,
        'lightgbm':     make_lgbm,
        'randomforest': make_rf,
        'svm':          make_svm,
        'naivebayes':   make_nb,
        'decisiontree': make_dt,
    }


# =============================================================================
# 7. 运行单次 Stacking 实验
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
        estimators=estimators, final_estimator=meta,
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
# 8. 评估指标计算
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
# 9. 最优阈值搜索
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
# 10. 保存单个实验输出（双语：lang='zh' 或 'en'）
# =============================================================================
def save_experiment_outputs(exp_name, y_test, y_pred, y_proba, output_dir,
                            lang: str = 'zh'):
    os.makedirs(output_dir, exist_ok=True)
    T = _T[lang]

    # ── 混淆矩阵 ────────────────────────────────────────────────────────────
    cm  = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[T['good_lbl'], T['bad_lbl']],
    ).plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f"{T['cm_title']} - {exp_name}", fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
    plt.close(fig)

    # ── ROC 曲线 ─────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_val     = roc_auc_score(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC (AUC={auc_val:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label=T['random_clf'])
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel(T['fpr']); ax.set_ylabel(T['tpr'])
    ax.set_title(f"ROC Curve - {exp_name}")
    ax.legend(loc='lower right'); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=150)
    plt.close(fig)

    # ── 分类报告 ────────────────────────────────────────────────────────────
    report = classification_report(
        y_test, y_pred,
        target_names=[T['cls_tgt_good'], T['cls_tgt_bad']],
        digits=4,
    )
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w',
              encoding='utf-8') as f:
        f.write(f"{T['cls_report_hdr']} - {exp_name}\n" + "=" * 60 + "\n" + report)

    # ── 指标 CSV ─────────────────────────────────────────────────────────────
    metrics_dict = {
        'accuracy':  accuracy_score(y_test, y_pred),
        'auc':       auc_val,
        'recall':    recall_score(y_test, y_pred, zero_division=0),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'f1':        f1_score(y_test, y_pred, zero_division=0),
    }
    pd.DataFrame([metrics_dict]).to_csv(
        os.path.join(output_dir, 'metrics.csv'), index=False, encoding='utf-8-sig')
    print(f"  📁 [{lang.upper()}] 实验输出已保存到：{output_dir}")


# =============================================================================
# 11. EXP-7：SMOTE + 深度特征工程 + Bayesian 超参数优化（XGBoost）
# =============================================================================
def run_exp7_smote_bayesian(
    X_train, X_test, y_train, y_test,
    scale_pos_weight: float,
    zh_output_dir: str,
    en_output_dir: str,
) -> dict:
    """
    进阶优化实验：
    Step-1  深度特征工程（log1p / 比率 / 分箱）
    Step-2  SMOTE 过采样（若 imblearn 可用）
    Step-3  Optuna Bayesian 超参数优化（若 optuna 可用）
    Step-4  训练最优 XGBoost 并评估
    """
    print("\n  [EXP-7] Step-1：深度特征工程...")
    X_tr_fe, X_te_fe, feat_names = deep_feature_engineering(X_train, X_test)

    print("  [EXP-7] Step-2：样本不平衡处理...")
    if HAS_IMBLEARN:
        smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=5)
        X_tr_bal, y_tr_bal = smote.fit_resample(X_tr_fe, y_train)
        pos_before = y_train.sum()
        pos_after  = y_tr_bal.sum()
        print(f"  [SMOTE] 重采样前：{len(y_train)} 条（坏客户 {pos_before}），"
              f"重采样后：{len(y_tr_bal)} 条（坏客户 {pos_after}）")
    else:
        print("  [SMOTE] 跳过（未安装 imbalanced-learn），使用原始训练集")
        X_tr_bal, y_tr_bal = X_tr_fe, y_train

    print("  [EXP-7] Step-3：Bayesian 超参数优化（Optuna）...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    if HAS_OPTUNA:
        def objective(trial):
            params = dict(
                n_estimators      = trial.suggest_int('n_estimators',     100, 600),
                max_depth         = trial.suggest_int('max_depth',          3,   9),
                learning_rate     = trial.suggest_float('learning_rate', 0.01, 0.3,  log=True),
                subsample         = trial.suggest_float('subsample',      0.5, 1.0),
                colsample_bytree  = trial.suggest_float('colsample_bytree',0.5, 1.0),
                min_child_weight  = trial.suggest_int('min_child_weight',   1,  10),
                gamma             = trial.suggest_float('gamma',          0.0, 0.5),
                reg_alpha         = trial.suggest_float('reg_alpha',      0.0, 2.0),
                reg_lambda        = trial.suggest_float('reg_lambda',     0.5, 3.0),
            )
            model = XGBClassifier(
                **params,
                scale_pos_weight=1.0,
                random_state=RANDOM_SEED,
                eval_metric='auc', use_label_encoder=False, verbosity=0,
                **XGB_GPU_PARAMS,
            )
            scores = cross_val_score(
                model, X_tr_bal, y_tr_bal,
                cv=cv, scoring='roc_auc', n_jobs=-1,
            )
            return scores.mean()

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        )
        study.optimize(objective, n_trials=50, show_progress_bar=False)
        best_params = study.best_params
        best_cv_auc = study.best_value
        print(f"  [Optuna] 最优参数：{best_params}")
        print(f"  [Optuna] 最优 AUC(CV)={best_cv_auc:.4f}")
    else:
        print("  [Bayesian] 跳过（未安装 optuna），使用默认参数")
        best_params = dict(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
        )

    print("  [EXP-7] Step-4：训练最优模型并评估...")
    final_model = XGBClassifier(
        **best_params,
        scale_pos_weight=1.0,
        random_state=RANDOM_SEED,
        eval_metric='auc', use_label_encoder=False, verbosity=0,
        **XGB_GPU_PARAMS,
    )
    final_model.fit(X_tr_bal, y_tr_bal)

    y_proba     = final_model.predict_proba(X_te_fe)[:, 1]
    y_pred      = (y_proba >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc         = roc_auc_score(y_test, y_proba)

    # ── 保存双语输出 ────────────────────────────────────────────────────────
    save_experiment_outputs('EXP-7', y_test, y_pred, y_proba, zh_output_dir, lang='zh')
    save_experiment_outputs('EXP-7', y_test, y_pred, y_proba, en_output_dir, lang='en')

    metrics = {
        'accuracy':  accuracy_score(y_test, y_pred),
        'auc':       auc,
        'recall':    recall_score(y_test, y_pred, zero_division=0),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'f1':        f1_score(y_test, y_pred, zero_division=0),
        'threshold': 0.5,
        'fpr':       fpr,
        'tpr':       tpr,
        'y_pred':    y_pred,
        'y_proba':   y_proba,
        'model':     final_model,
        'feat_names':feat_names,
    }
    print(f"  ✅ EXP-7  AUC={auc:.4f}  Acc={metrics['accuracy']:.4f}  "
          f"Recall={metrics['recall']:.4f}  F1={metrics['f1']:.4f}")
    return metrics


# =============================================================================
# 12. EXP-8：深度学习尝试（DNN / MLPClassifier）
# =============================================================================
def _plot_dnn_loss_curve(dnn, output_dir: str, lang: str):
    """DNN 训练损失曲线（双语辅助函数）"""
    T = _T[lang]
    if not hasattr(dnn, 'loss_curve_'):
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(dnn.loss_curve_, color='steelblue', lw=2, label=T['train_loss'])
    if hasattr(dnn, 'validation_scores_') and dnn.validation_scores_:
        ax2 = ax.twinx()
        ax2.plot(dnn.validation_scores_, color='darkorange',
                 lw=2, linestyle='--', label=T['val_auc'])
        ax2.set_ylabel(T['val_auc'], fontsize=10)
        ax2.legend(loc='center right')
    ax.set_xlabel(T['iteration'])
    ax.set_ylabel(T['train_loss'], fontsize=10)
    ax.set_title(T['dnn_title'], fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'dnn_loss_curve.png'), dpi=150)
    plt.close(fig)


def run_exp8_dnn(
    X_train, X_test, y_train, y_test,
    zh_output_dir: str,
    en_output_dir: str,
) -> dict:
    """
    使用 sklearn MLPClassifier 构建多层感知机（DNN）：
    · 标准化输入（StandardScaler）
    · 三隐藏层：[256, 128, 64]，激活 relu
    · Adam 优化器，早停
    """
    from sklearn.utils.class_weight import compute_sample_weight

    print("  [EXP-8] 数据标准化...")
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    sample_w = compute_sample_weight('balanced', y_train)

    print("  [EXP-8] 训练 DNN（MLP：256→128→64，relu，adam，早停）...")
    dnn = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        batch_size=256,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=RANDOM_SEED,
        verbose=False,
    )
    dnn.fit(X_tr_sc, y_train, sample_weight=sample_w)

    y_proba     = dnn.predict_proba(X_te_sc)[:, 1]
    y_pred      = (y_proba >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc         = roc_auc_score(y_test, y_proba)

    # ── 保存双语输出 ────────────────────────────────────────────────────────
    save_experiment_outputs('EXP-8 DNN', y_test, y_pred, y_proba, zh_output_dir, lang='zh')
    save_experiment_outputs('EXP-8 DNN', y_test, y_pred, y_proba, en_output_dir, lang='en')
    _plot_dnn_loss_curve(dnn, zh_output_dir, lang='zh')
    _plot_dnn_loss_curve(dnn, en_output_dir, lang='en')

    metrics = {
        'accuracy':  accuracy_score(y_test, y_pred),
        'auc':       auc,
        'recall':    recall_score(y_test, y_pred, zero_division=0),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'f1':        f1_score(y_test, y_pred, zero_division=0),
        'threshold': 0.5,
        'fpr':       fpr,
        'tpr':       tpr,
        'y_pred':    y_pred,
        'y_proba':   y_proba,
    }
    print(f"  ✅ EXP-8  AUC={auc:.4f}  Acc={metrics['accuracy']:.4f}  "
          f"Recall={metrics['recall']:.4f}  F1={metrics['f1']:.4f}")
    return metrics


# =============================================================================
# 13. KS 统计量曲线 + 累积提升图（Lift Chart）—— 双语
# =============================================================================
def plot_ks_lift(results: dict, output_dir: str, lang: str = 'zh'):
    """
    对 AUC 最高的若干实验绘制：
    ① KS 曲线：累积正/负样本分布差，标注 KS 统计量
    ② 累积提升图（Gain / Lift Chart）
    """
    os.makedirs(output_dir, exist_ok=True)
    T = _T[lang]

    sorted_keys = sorted(results, key=lambda k: results[k]['auc'], reverse=True)
    plot_keys   = sorted_keys[:5]
    palette     = ['#e41a1c','#377eb8','#4daf4a','#ff7f00','#984ea3']

    fig = plt.figure(figsize=(16, 6))
    gs  = GridSpec(1, 2, figure=fig, wspace=0.35)
    ax_ks   = fig.add_subplot(gs[0])
    ax_lift = fig.add_subplot(gs[1])

    for idx, key in enumerate(plot_keys):
        y_true  = results[key].get('y_true_ref')
        y_proba = results[key]['y_proba']
        if y_true is None:
            continue

        color   = palette[idx % len(palette)]
        n       = len(y_true)
        pos_tot = y_true.sum()
        neg_tot = n - pos_tot

        order    = np.argsort(y_proba)[::-1]
        y_sorted = np.array(y_true)[order]
        cum_pos  = np.cumsum(y_sorted)      / pos_tot
        cum_neg  = np.cumsum(1 - y_sorted)  / neg_tot
        pct_pop  = np.arange(1, n + 1) / n

        ks_stat = np.max(np.abs(cum_pos - cum_neg))
        ks_idx  = np.argmax(np.abs(cum_pos - cum_neg))
        short   = key.replace('EXP-','E').replace('Stacking','STK').replace(' ','')

        ax_ks.plot(pct_pop, cum_pos, color=color, lw=1.8,
                   label=f'{short} KS={ks_stat:.3f}')
        ax_ks.plot(pct_pop, cum_neg, color=color, lw=1.2,
                   linestyle='--', alpha=0.6)
        ax_ks.axvline(x=pct_pop[ks_idx], color=color, lw=0.8, linestyle=':')

        cum_gain = cum_pos
        ax_lift.plot(pct_pop, cum_gain / pct_pop, color=color, lw=1.8, label=short)

    ax_ks.plot([0, 1], [0, 1], 'gray', lw=1.0, linestyle='--',
               label=T['random_model'])
    ax_ks.set_xlabel(T['sample_pct_ks'], fontsize=11)
    ax_ks.set_ylabel(T['cum_pn'], fontsize=11)
    ax_ks.set_title(T['ks_title'], fontsize=12, fontweight='bold')
    ax_ks.legend(fontsize=8.5, loc='lower right')
    ax_ks.grid(True, alpha=0.25)

    ax_lift.axhline(y=1.0, color='gray', lw=1.2, linestyle='--',
                    label=T['random_baseline'])
    ax_lift.set_xlabel(T['sample_pct2'], fontsize=11)
    ax_lift.set_ylabel(T['lift_factor'], fontsize=11)
    ax_lift.set_title(T['lift_title'], fontsize=12, fontweight='bold')
    ax_lift.legend(fontsize=8.5, loc='upper right')
    ax_lift.grid(True, alpha=0.25)

    path = os.path.join(output_dir, 'ks_lift_chart.png')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Chart/{lang.upper()}] ks_lift_chart.png → {path}")

    # ── KS 汇总打印 ──────────────────────────────────────────────────────────
    print("\n  KS 统计量汇总（工业标准：>0.3 可用，>0.4 良好）：")
    for key in sorted_keys:
        y_true  = results[key].get('y_true_ref')
        y_proba = results[key]['y_proba']
        if y_true is None:
            continue
        n, pos_tot = len(y_true), y_true.sum()
        neg_tot = n - pos_tot
        order   = np.argsort(y_proba)[::-1]
        ys      = np.array(y_true)[order]
        cum_pos = np.cumsum(ys) / pos_tot
        cum_neg = np.cumsum(1 - ys) / neg_tot
        ks      = np.max(np.abs(cum_pos - cum_neg))
        flag    = "✅" if ks >= 0.3 else "⚠️"
        print(f"  {flag}  {key:<50}  KS={ks:.4f}")


# =============================================================================
# 14. SHAP 可解释性分析（基于最优树模型）
# =============================================================================
def run_shap_analysis(
    model, X_train, X_test, feature_names: list,
    exp_name: str, output_dir: str,
    max_display: int = 20,
    n_background: int = 200,
):
    """
    SHAP 可解释性分析，适配树模型（XGB/LGB/RF/CatBoost/DT）。
    生成：
    ① Summary plot（蜂群图）— 全局特征影响力及方向
    ② Bar plot          — 平均 |SHAP| 特征重要性
    ③ Waterfall plot    — 单样本预测路径
    ④ Dependence plot   — Top-3 特征的 SHAP 依赖图
    """
    if not HAS_SHAP:
        print("  [SHAP] 跳过（未安装 shap）")
        return

    os.makedirs(output_dir, exist_ok=True)

    X_test_df  = X_test  if isinstance(X_test,  pd.DataFrame) else \
        pd.DataFrame(X_test,  columns=feature_names)
    X_train_df = X_train if isinstance(X_train, pd.DataFrame) else \
        pd.DataFrame(X_train, columns=feature_names)

    print(f"  [SHAP] 计算 TreeExplainer SHAP 值（{exp_name}）...")
    try:
        explainer   = shap.TreeExplainer(
            model,
            data=shap.maskers.Independent(
                X_train_df.sample(min(n_background, len(X_train_df)),
                                  random_state=RANDOM_SEED),
                max_samples=n_background,
            ),
        )
        shap_values = explainer(X_test_df)
        if len(shap_values.shape) == 3:
            sv = shap.Explanation(
                values=shap_values.values[:, :, 1],
                base_values=shap_values.base_values[:, 1],
                data=shap_values.data,
                feature_names=feature_names,
            )
        else:
            sv = shap_values
    except Exception as e:
        print(f"  [SHAP] TreeExplainer 失败，尝试 Explainer：{e}")
        explainer   = shap.Explainer(model, X_train_df.sample(
            min(n_background, len(X_train_df)), random_state=RANDOM_SEED))
        shap_values = explainer(X_test_df)
        sv          = shap_values

    # ── ① Summary plot（蜂群图）───────────────────────────────────────────
    shap.plots.bar(sv, max_display=max_display, show=False)
    plt.gcf().set_size_inches(9, 6)
    plt.title(f'SHAP Feature Importance (Mean |SHAP|) — {exp_name}',
              fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_bar_importance.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── ② Beeswarm plot ─────────────────────────────────────────────────
    shap.plots.beeswarm(sv, max_display=max_display, show=False)
    plt.gcf().set_size_inches(9, 6)
    plt.title(f'SHAP Beeswarm — {exp_name}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_beeswarm.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── ③ Waterfall（高风险 & 低风险各一例）─────────────────────────────
    y_proba       = model.predict_proba(X_test_df)[:, 1]
    high_risk_idx = int(np.argmax(y_proba))
    low_risk_idx  = int(np.argmin(y_proba))

    for label_zh, label_en, idx in [
        ('高风险客户', 'HighRisk', high_risk_idx),
        ('低风险客户', 'LowRisk',  low_risk_idx),
    ]:
        shap.plots.waterfall(sv[idx], max_display=15, show=False)
        plt.title(f'SHAP Waterfall — {label_en} (prob={y_proba[idx]:.3f})',
                  fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'shap_waterfall_{label_en.lower()}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # ── ④ Dependence plot（Top-3 特征）──────────────────────────────────
    mean_abs_shap = np.abs(sv.values).mean(axis=0)
    top3_idx      = np.argsort(mean_abs_shap)[::-1][:3]
    for rank, fidx in enumerate(top3_idx):
        fname_feat = feature_names[fidx]
        fig, ax = plt.subplots(figsize=(7, 5))
        shap.plots.scatter(sv[:, fidx], color=sv, show=False, ax=ax)
        ax.set_title(f'SHAP Dependence — Top{rank+1}: {fname_feat}',
                     fontsize=11, fontweight='bold')
        fig.tight_layout()
        safe_fname = fname_feat.replace('/', '_').replace(' ', '_')
        fig.savefig(os.path.join(output_dir, f'shap_dependence_{safe_fname}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"  [SHAP] 所有图表已保存 → {output_dir}/shap_*.png")


# =============================================================================
# 15. 综合绘图函数（双语）
# =============================================================================
# EXP-1 单模型颜色（浅色系）
_EXP1_COLORS = {
    'EXP-1 LogisticRegression': '#aec6cf',
    'EXP-1 NaiveBayes':         '#b5ead7',
    'EXP-1 SVM':                '#c7a6d4',
    'EXP-1 DecisionTree':       '#f9c3b8',
    'EXP-1 RandomForest':       '#f7c6a0',
    'EXP-1 XGBoost':            '#a0c4ff',
    'EXP-1 LightGBM':           '#caffbf',
    'EXP-1 CatBoost':           '#ffd6a5',
}
# Stacking + 进阶实验颜色
_STACKING_PALETTE = {
    'EXP-2 Stacking[3 base+LR]':                ('#e41a1c', '-',  2.0),
    'EXP-3 Stacking[5 base+LR]':                ('#377eb8', '-',  2.0),
    'EXP-4 Stacking[5 base+XGB]':               ('#4daf4a', '-',  2.0),
    'EXP-5 Stacking[5 base+XGB+passthrough]':   ('#ff7f00', '-',  2.2),
    'EXP-6 Optimal threshold (based on EXP-5)': ('#984ea3', ':',  2.0),
    'EXP-7 SMOTE+FE+Bayesian(XGB)':             ('#a65628', '-',  2.5),
    'EXP-8 DNN(MLP)':                           ('#f781bf', '--', 2.0),
}


def plot_roc_all(results: dict, output_dir: str, lang: str = 'zh'):
    T = _T[lang]
    fig, ax = plt.subplots(figsize=(12, 8))
    for name, m in results.items():
        if name.startswith('EXP-1'):
            color = _EXP1_COLORS.get(name, '#cccccc')
            ax.plot(m['fpr'], m['tpr'], color=color, lw=1.3, linestyle='--',
                    alpha=0.80, label=f"{name}  (AUC={m['auc']:.4f})")
    for name, m in results.items():
        if not name.startswith('EXP-1'):
            color, ls, lw = _STACKING_PALETTE.get(name, ('gray', '-', 1.5))
            ax.plot(m['fpr'], m['tpr'], color=color, linestyle=ls, lw=lw,
                    label=f"{name}  (AUC={m['auc']:.4f})")
    ax.plot([0, 1], [0, 1], 'lightgray', lw=1.0, linestyle=':',
            label=T['random_clf'])
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel(T['fpr'], fontsize=13)
    ax.set_ylabel(T['tpr'], fontsize=13)
    ax.set_title(T['roc_title'], fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=7.5, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.25)
    path = os.path.join(output_dir, 'roc_all_experiments.png')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Chart/{lang.upper()}] roc_all_experiments.png → {path}")


def plot_metrics_bar(results: dict, output_dir: str, lang: str = 'zh'):
    """
    修复版：
    · ylim 从 0 开始，动态上限，确保所有指标值（含接近 0 的召回率/F1）完整显示
    · bbox_inches='tight' 防止标签被裁剪
    · 图幅加高至 9 英寸，bar 标签字号适当缩小
    """
    T = _T[lang]
    exp1_names     = [n for n in results if n.startswith('EXP-1')]
    stacking_names = [n for n in results if not n.startswith('EXP-1')]
    metrics        = ['auc', 'accuracy', 'recall', 'f1']
    metric_labels  = ['AUC', 'Accuracy', 'Recall', 'F1-Score']
    colors         = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    width          = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(28, 9))

    for ax, group_names, title in [
        (axes[0], exp1_names,     T['exp1_title']),
        (axes[1], stacking_names, T['exp28_title']),
    ]:
        if not group_names:
            ax.set_visible(False)
            continue

        short = [
            n.replace('EXP-1 ', '').replace('EXP-', 'E')
             .replace('Stacking', 'STK').replace('passthrough', 'PT')
             .replace('SMOTE+FE+Bayesian', 'Opt').replace('DNN(MLP)', 'DNN')
            for n in group_names
        ]

        x = np.arange(len(group_names))

        # 计算动态 ylim
        all_vals = [results[n][m] for n in group_names for m in metrics
                    if results[n].get(m) is not None]
        y_min_data = min(all_vals) if all_vals else 0.0
        y_min = max(0.0, y_min_data - 0.05)   # 从 0 起，留出 5% 余量
        y_max = 1.20                            # 上方留足空间给旋转标签

        for i, (met, lab, col) in enumerate(zip(metrics, metric_labels, colors)):
            vals = [results[n][met] for n in group_names]
            bars = ax.bar(x + i * width, vals, width, label=lab,
                          color=col, alpha=0.85)
            for bar, val in zip(bars, vals):
                # 标签基线取 max(bar顶+偏移, y_min+0.01) 保证始终可见
                y_txt = max(bar.get_height() + 0.010, y_min + 0.012)
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y_txt,
                    f'{val:.3f}',
                    ha='center', va='bottom',
                    fontsize=6.2, rotation=90,
                    clip_on=False,           # ← 关键：不裁剪超出 axes 的文字
                )

        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(short, rotation=35, ha='right', fontsize=8.5)
        ax.set_ylim([y_min, y_max])
        ax.set_ylabel(T['metric_val'], fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.25, axis='y')

    path = os.path.join(output_dir, 'metrics_bar_chart.png')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')   # ← bbox_inches='tight'
    plt.close(fig)
    print(f"[Chart/{lang.upper()}] metrics_bar_chart.png → {path}")


def plot_best_confusion_matrix(y_test, y_pred, exp_name, output_dir,
                               lang: str = 'zh'):
    T  = _T[lang]
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[T['good_lbl'], T['bad_lbl']],
    ).plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f"{T['cm_title']} - {exp_name}", fontsize=12, fontweight='bold')
    path = os.path.join(output_dir, 'confusion_matrix_best.png')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Chart/{lang.upper()}] confusion_matrix_best.png → {path}")


def plot_pr_threshold(y_test, y_proba, best_thr_f1, best_thr_recall,
                      precisions, recalls, thresholds, output_dir,
                      lang: str = 'zh'):
    T         = _T[lang]
    f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-8)
    fig = plt.figure(figsize=(13, 5))
    gs  = GridSpec(1, 2, figure=fig, wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(recalls, precisions, color='steelblue', lw=2, label=T['pr_curve_lbl'])
    idx_f1 = np.argmin(np.abs(thresholds - best_thr_f1))
    ax1.scatter(recalls[idx_f1], precisions[idx_f1], s=120, color='#d62728', zorder=5,
                label=f"{T['f1_best_thr']} = {best_thr_f1:.3f}")
    idx_rc = np.argmin(np.abs(thresholds - best_thr_recall))
    ax1.scatter(recalls[idx_rc], precisions[idx_rc], s=120, color='#ff7f00',
                marker='^', zorder=5,
                label=f"{T['recall_best_thr']} = {best_thr_recall:.3f}")
    ax1.set_xlabel('Recall', fontsize=12)
    ax1.set_ylabel('Precision', fontsize=12)
    ax1.set_title(T['pr_title'], fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(thresholds, precisions[:-1], label='Precision',   color='#377eb8', lw=2)
    ax2.plot(thresholds, recalls[:-1],    label='Recall',      color='#e41a1c', lw=2)
    ax2.plot(thresholds, f1_scores,       label='F1-Score',    color='#4daf4a', lw=2, linestyle='--')
    ax2.axvline(x=best_thr_f1,     color='#d62728', lw=1.5, linestyle=':',
                label=f"{T['f1_best_lbl']}={best_thr_f1:.3f}")
    ax2.axvline(x=best_thr_recall, color='#ff7f00', lw=1.5, linestyle='-.',
                label=f"{T['recall_best_lbl']}={best_thr_recall:.3f}")
    ax2.axvline(x=0.5,             color='gray',    lw=1.2, linestyle='--',
                label=T['default_thr'])
    ax2.set_xlabel(T['decision_thr'], fontsize=12)
    ax2.set_ylabel('Metric Value' if lang == 'en' else '指标值', fontsize=12)
    ax2.set_title(T['thr_title'], fontsize=12, fontweight='bold')
    ax2.set_xlim([0.1, 0.9]); ax2.set_ylim([0.0, 1.05])
    ax2.legend(fontsize=8.5, loc='center left'); ax2.grid(True, alpha=0.3)

    path = os.path.join(output_dir, 'pr_curve_threshold.png')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Chart/{lang.upper()}] pr_curve_threshold.png → {path}")


# =============================================================================
# 16. 汇总打印与 CSV 保存
# =============================================================================
def print_and_save_summary(results: dict, zh_root: str, en_root: str,
                           stacking_output_dir: str):
    print("\n" + "=" * 96)
    print("                              实验结果汇总")
    print("=" * 96)
    print(f"  {'实验名称':<48} {'AUC':>7} {'准确率':>7} {'召回率':>7} "
          f"{'F1':>7} {'KS':>7} {'阈值':>7}")
    print("-" * 96)

    best_auc = max(m['auc'] for m in results.values())
    rows_zh  = []
    rows_en  = []

    def _ks(m):
        y_true  = m.get('y_true_ref')
        y_proba = m.get('y_proba')
        if y_true is None or y_proba is None:
            return float('nan')
        n, pos_tot = len(y_true), y_true.sum()
        neg_tot = n - pos_tot
        if pos_tot == 0 or neg_tot == 0:
            return float('nan')
        order   = np.argsort(y_proba)[::-1]
        ys      = np.array(y_true)[order]
        cum_pos = np.cumsum(ys)       / pos_tot
        cum_neg = np.cumsum(1 - ys)   / neg_tot
        return float(np.max(np.abs(cum_pos - cum_neg)))

    def _collect_row(name, m):
        marker = " ★" if abs(m['auc'] - best_auc) < 1e-6 else "  "
        ks_val = _ks(m)
        ks_str = f"{ks_val:.4f}" if not np.isnan(ks_val) else "  N/A "
        print(f"  {name:<48}{marker} {m['auc']:>6.4f}  {m['accuracy']:>6.4f}  "
              f"{m['recall']:>6.4f}  {m['f1']:>6.4f}  {ks_str:>7}  "
              f"{m['threshold']:>6.3f}")
        rows_zh.append({
            '实验名称': name,
            'AUC':      round(m['auc'], 4),
            '准确率':   round(m['accuracy'], 4),
            '召回率':   round(m['recall'], 4),
            '精确率':   round(m['precision'], 4),
            'F1-Score': round(m['f1'], 4),
            'KS统计量': round(ks_val, 4) if not np.isnan(ks_val) else None,
            '阈值':     round(m['threshold'], 3),
        })
        rows_en.append({
            'Experiment':  name,
            'AUC':         round(m['auc'], 4),
            'Accuracy':    round(m['accuracy'], 4),
            'Recall':      round(m['recall'], 4),
            'Precision':   round(m['precision'], 4),
            'F1-Score':    round(m['f1'], 4),
            'KS':          round(ks_val, 4) if not np.isnan(ks_val) else None,
            'Threshold':   round(m['threshold'], 3),
        })

    for name, m in results.items():
        if name.startswith('EXP-1'):
            _collect_row(name, m)
    print("  " + "─ " * 48)
    for name, m in results.items():
        if not name.startswith('EXP-1'):
            _collect_row(name, m)

    print("=" * 96)
    print("★ 为所有方案中 AUC 最高者")

    # 保存到三个位置：zh_root / en_root / stacking_output_dir（根目录）
    for save_dir, rows, enc in [
        (zh_root,              rows_zh, 'utf-8-sig'),
        (en_root,              rows_en, 'utf-8-sig'),
        (stacking_output_dir,  rows_en, 'utf-8-sig'),
    ]:
        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, 'experiment_summary.csv')
        pd.DataFrame(rows).to_csv(csv_path, index=False, encoding=enc)
        print(f"[CSV] 汇总表 → {csv_path}")


# =============================================================================
# 主程序
# =============================================================================
def main():
    print("=" * 76)
    print("  银行客户信用风险评估 —— 多方案实验对比（含进阶优化）")
    print()
    print("  EXP-1 : 8 个单模型基线（含决策树）")
    print("          LR / NaiveBayes / SVM / DT / RF / XGB / LGBM / CatBoost")
    print("  EXP-2 : Stacking [XGB+LGBM+RF] → LR")
    print("  EXP-3 : Stacking [XGB+LGBM+RF+SVM+NB] → LR")
    print("  EXP-4 : Stacking [XGB+LGBM+RF+SVM+NB] → XGB")
    print("  EXP-5 : Stacking [XGB+LGBM+RF+SVM+NB] → XGB + passthrough")
    print("  EXP-6 : EXP-5 + PR 曲线最优阈值调优")
    print("  EXP-7 : SMOTE + 深度特征工程 + Bayesian 超参数优化（XGB）")
    print("  EXP-8 : DNN（MLPClassifier：256→128→64）")
    print("  Post  : KS/Lift 业务指标图 + SHAP 可解释性分析")
    print("=" * 76 + "\n")

    if XGB_GPU_PARAMS:
        print("[硬件] XGBoost / LightGBM GPU 加速已启用。")
    else:
        print("[硬件] 使用 CPU 运行（未检测到 GPU）。")

    # ── 路径 ──────────────────────────────────────────────────────────────────
    data_path, stacking_output_dir, dataset_name = get_input_file_and_output_dir()

    # 双语目录
    zh_root = os.path.join(stacking_output_dir, 'Chinese')
    en_root = os.path.join(stacking_output_dir, 'English')
    os.makedirs(stacking_output_dir, exist_ok=True)
    os.makedirs(zh_root, exist_ok=True)
    os.makedirs(en_root, exist_ok=True)

    print(f"\n[输出根目录] {stacking_output_dir}")
    print(f"  ├── Chinese\\  （中文图表）")
    print(f"  └── English\\  （英文图表）\n")

    # ── 数据准备 ───────────────────────────────────────────────────────────────
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
    print(f"[样本权重] scale_pos_weight = {spw:.2f}"
          f"（好客户 {neg_cnt} : 坏客户 {pos_cnt}）\n")

    results = {}
    models  = {}
    base_3  = ['xgboost', 'lightgbm', 'randomforest']
    base_5  = ['xgboost', 'lightgbm', 'randomforest', 'svm', 'naivebayes']

    steps = [
        'EXP-1 单模型基线（8个）',
        'EXP-2 Stacking[3]+LR',
        'EXP-3 Stacking[5]+LR',
        'EXP-4 Stacking[5]+XGB',
        'EXP-5 Stacking[5]+XGB+PT',
        'EXP-6 阈值调优',
        'EXP-7 SMOTE+FE+Bayesian',
        'EXP-8 DNN',
        '绘图（ROC/Metrics/CM/PR）',
        'KS + Lift 图',
        'SHAP 分析',
        '汇总',
    ]

    with tqdm(total=len(steps), desc="流水线进度",
              dynamic_ncols=True, mininterval=0.2) as pbar:

        # ── EXP-1 ──────────────────────────────────────────────────────────
        print(f"\n{'─'*68}")
        print(f"  EXP-1 : 单模型基线（8 个模型逐一运行）")
        print(f"{'─'*68}")
        zh_exp1_root = os.path.join(zh_root, 'EXP-1')
        en_exp1_root = os.path.join(en_root, 'EXP-1')
        exp1_results = run_exp1_all_single_models(
            X_train, X_test, y_train, y_test,
            dataset_name,
            zh_exp1_root,
            en_exp1_root,
        )
        for k in exp1_results:
            exp1_results[k]['y_true_ref'] = y_test.values
        results.update(exp1_results)
        pbar.update(1)

        # ── EXP-2 ~ EXP-5 (Stacking) ───────────────────────────────────────
        for cfg in [
            ('EXP-2 Stacking[3 base+LR]',               base_3, 'lr',  False, 'EXP-2'),
            ('EXP-3 Stacking[5 base+LR]',               base_5, 'lr',  False, 'EXP-3'),
            ('EXP-4 Stacking[5 base+XGB]',              base_5, 'xgb', False, 'EXP-4'),
            ('EXP-5 Stacking[5 base+XGB+passthrough]',  base_5, 'xgb', True,  'EXP-5'),
        ]:
            exp_key, bases, meta, pt, dirname = cfg
            zh_dir = os.path.join(zh_root, dirname)
            en_dir = os.path.join(en_root, dirname)
            print(f"\n{'─'*68}\n  {exp_key}\n{'─'*68}")
            m = run_stacking_experiment(exp_key, bases, meta, pt,
                                        X_train, y_train, spw)
            results[exp_key] = calc_metrics(m, X_test, y_test)
            results[exp_key]['y_true_ref'] = y_test.values
            models[exp_key] = m
            print(f"  AUC = {results[exp_key]['auc']:.4f}")
            save_experiment_outputs(exp_key, y_test,
                                    results[exp_key]['y_pred'],
                                    results[exp_key]['y_proba'],
                                    zh_dir, lang='zh')
            save_experiment_outputs(exp_key, y_test,
                                    results[exp_key]['y_pred'],
                                    results[exp_key]['y_proba'],
                                    en_dir, lang='en')
            pbar.update(1)

        # ── EXP-6 阈值调优 ──────────────────────────────────────────────────
        stacking_only = {k: v for k, v in results.items()
                         if k.startswith('EXP-') and 'Stacking' in k}
        best_stk_key  = max(stacking_only, key=lambda k: stacking_only[k]['auc'])
        best_proba    = results[best_stk_key]['y_proba']
        print(f"\n{'─'*68}\n  EXP-6 阈值调优（基于 {best_stk_key}）\n{'─'*68}")

        thr_f1,    prec, rec, thrs = find_best_threshold(y_test, best_proba, 'f1')
        thr_recall, *_             = find_best_threshold(y_test, best_proba, 'recall')

        exp_key = 'EXP-6 Optimal threshold (based on EXP-5)'
        zh_dir  = os.path.join(zh_root, 'EXP-6')
        en_dir  = os.path.join(en_root, 'EXP-6')
        results[exp_key] = calc_metrics(models[best_stk_key], X_test, y_test,
                                        threshold=thr_f1)
        results[exp_key]['y_true_ref'] = y_test.values
        print(f"  AUC={results[exp_key]['auc']:.4f}  "
              f"Recall={results[exp_key]['recall']:.4f}  "
              f"F1={results[exp_key]['f1']:.4f}")
        save_experiment_outputs(exp_key, y_test,
                                results[exp_key]['y_pred'],
                                results[exp_key]['y_proba'],
                                zh_dir, lang='zh')
        save_experiment_outputs(exp_key, y_test,
                                results[exp_key]['y_pred'],
                                results[exp_key]['y_proba'],
                                en_dir, lang='en')
        pbar.update(1)

        # ── EXP-7 SMOTE + Bayesian ─────────────────────────────────────────
        print(f"\n{'─'*68}\n  EXP-7：SMOTE + 深度特征工程 + Bayesian 超参数优化\n{'─'*68}")
        exp7_m = run_exp7_smote_bayesian(
            X_train, X_test, y_train, y_test, spw,
            zh_output_dir=os.path.join(zh_root, 'EXP-7'),
            en_output_dir=os.path.join(en_root, 'EXP-7'),
        )
        exp7_m['y_true_ref'] = y_test.values
        results['EXP-7 SMOTE+FE+Bayesian(XGB)'] = exp7_m
        pbar.update(1)

        # ── EXP-8 DNN ──────────────────────────────────────────────────────
        print(f"\n{'─'*68}\n  EXP-8：DNN（MLPClassifier）\n{'─'*68}")
        exp8_m = run_exp8_dnn(
            X_train, X_test, y_train, y_test,
            zh_output_dir=os.path.join(zh_root, 'EXP-8'),
            en_output_dir=os.path.join(en_root, 'EXP-8'),
        )
        exp8_m['y_true_ref'] = y_test.values
        results['EXP-8 DNN(MLP)'] = exp8_m
        pbar.update(1)

        # ── 综合对比图表（双语）────────────────────────────────────────────
        print(f"\n{'─'*68}\n  生成综合对比图表（中英双版）...\n{'─'*68}")
        best_key = max(results, key=lambda k: results[k]['auc'])
        for lang, root in [('zh', zh_root), ('en', en_root)]:
            plot_roc_all(results, root, lang=lang)
            plot_metrics_bar(results, root, lang=lang)
            plot_best_confusion_matrix(
                y_test, results[best_key]['y_pred'], best_key, root, lang=lang
            )
            plot_pr_threshold(
                y_test, best_proba, thr_f1, thr_recall,
                prec, rec, thrs, root, lang=lang
            )
        pbar.update(1)

        # ── KS + Lift 图（双语）────────────────────────────────────────────
        print(f"\n{'─'*68}\n  生成 KS + 提升图（中英双版）...\n{'─'*68}")
        for lang, root in [('zh', zh_root), ('en', en_root)]:
            plot_ks_lift(results, root, lang=lang)
        pbar.update(1)

        # ── SHAP 分析（输出到 zh_root/SHAP_analysis）──────────────────────
        print(f"\n{'─'*68}\n  SHAP 可解释性分析...\n{'─'*68}")
        tree_candidates = [
            k for k in results
            if k.startswith('EXP-1') and
               any(t in k for t in ('XGBoost', 'CatBoost', 'LightGBM',
                                    'RandomForest', 'DecisionTree'))
        ]
        shap_key = max(tree_candidates, key=lambda k: results[k]['auc']) \
            if tree_candidates else None

        if shap_key and results[shap_key].get('model') is not None:
            shap_dir = os.path.join(zh_root, 'SHAP_analysis')
            print(f"  SHAP 目标模型：{shap_key}")
            run_shap_analysis(
                model         = results[shap_key]['model'],
                X_train       = X_train,
                X_test        = X_test,
                feature_names = X.columns.tolist(),
                exp_name      = shap_key,
                output_dir    = shap_dir,
            )
        else:
            print("  [SHAP] 未找到合适的树模型，跳过。")
        pbar.update(1)

        # ── 汇总 ──────────────────────────────────────────────────────────
        print_and_save_summary(results, zh_root, en_root, stacking_output_dir)
        pbar.update(1)

    # ── 最终统计 ───────────────────────────────────────────────────────────────
    best_exp1_auc = max(v['auc'] for k, v in results.items() if k.startswith('EXP-1'))
    best_key      = max(results, key=lambda k: results[k]['auc'])
    best_auc      = results[best_key]['auc']
    delta         = (best_auc - best_exp1_auc) * 100

    print(f"\n{'=' * 76}")
    print(f"  ✅ 全部实验完成！")
    print(f"     EXP-1 单模型最高 AUC  = {best_exp1_auc:.4f}")
    print(f"     全局最优实验           : [{best_key}]")
    print(f"     全局最优 AUC           = {best_auc:.4f}  "
          f"({'↑' if delta >= 0 else '↓'}{abs(delta):.2f}% vs 单模型最优)")
    print(f"\n  📁 全部输出根目录：{stacking_output_dir}/")
    print(f"     ├── Chinese/                        中文版图表")
    print(f"     │   ├── EXP-1/{{ModelName}}/         单模型输出")
    print(f"     │   ├── EXP-2/ ~ EXP-8/             各集成实验")
    print(f"     │   ├── SHAP_analysis/              SHAP 可解释性图表")
    print(f"     │   ├── roc_all_experiments.png")
    print(f"     │   ├── metrics_bar_chart.png")
    print(f"     │   ├── confusion_matrix_best.png")
    print(f"     │   ├── pr_curve_threshold.png")
    print(f"     │   ├── ks_lift_chart.png")
    print(f"     │   └── experiment_summary.csv")
    print(f"     ├── English/                        英文版图表（同结构）")
    print(f"     └── experiment_summary.csv          根目录汇总（英文列名）")
    print(f"{'=' * 76}")


if __name__ == "__main__":
    main()