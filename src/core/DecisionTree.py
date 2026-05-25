# =============================================================================
# 毕业设计：基于机器学习的银行客户信用风险评估模型研究与实现
# 单体模型：决策树 (Decision Tree)
#
# API 规范（与 core/ 其他模块完全一致）：
#   train_model(X_train, y_train)                     → model
#   evaluate_model(model, X_test, y_test, output_dir) → float (AUC)
#   plot_feature_importance(model, feature_names, output_dir)
#
# 特点：
#   · GridSearchCV 搜索最优结构参数（深度 / 叶节点 / 划分准则）
#   · CCP（代价复杂度剪枝）进一步防止过拟合
#   · class_weight='balanced' 应对类别不平衡
#   · 额外输出树结构可视化（前 3 层）
# =============================================================================

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_val_score, train_test_split,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    recall_score, f1_score, precision_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)

warnings.filterwarnings('ignore')

# ── 全局随机种子（Stacking.py 会在 import 后注入覆盖此值）───────────────────
RANDOM_SEED = 42

# ── 中文字体 ──────────────────────────────────────────────────────────────────
_zh_fonts = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Noto Sans CJK SC']
for _f in _zh_fonts:
    if any(_f.lower() in fp.name.lower() for fp in fm.fontManager.ttflist):
        plt.rcParams['font.sans-serif'] = [_f]
        plt.rcParams['axes.unicode_minus'] = False
        break


# =============================================================================
# 1. 模型训练
# =============================================================================
def train_model(X_train, y_train):
    """
    两阶段训练策略：
    Stage-1  GridSearchCV  搜索 max_depth / min_samples_leaf / criterion 等结构参数
    Stage-2  CCP 剪枝      在最优结构参数基础上用 CV 选最优 ccp_alpha
    最终在完整训练集上用最优参数组合训练模型。
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    # ── Stage-1：结构参数网格搜索 ──────────────────────────────────────────
    param_grid = {
        'max_depth':         [4, 6, 8, 10, None],
        'min_samples_leaf':  [5, 10, 20, 30],
        'min_samples_split': [10, 20, 40],
        'criterion':         ['gini', 'entropy'],
    }
    base_dt = DecisionTreeClassifier(
        class_weight='balanced', random_state=RANDOM_SEED
    )
    gs = GridSearchCV(
        base_dt, param_grid,
        cv=cv, scoring='roc_auc',
        n_jobs=-1, refit=True, verbose=0,
    )
    gs.fit(X_train, y_train)
    best_struct = gs.best_params_
    print(f"    [DT-GridCV]  最优结构参数：{best_struct}  "
          f"AUC(CV)={gs.best_score_:.4f}")

    # ── Stage-2：CCP 剪枝 α 选择 ───────────────────────────────────────────
    dt_path = DecisionTreeClassifier(
        class_weight='balanced', random_state=RANDOM_SEED, **best_struct
    )
    path   = dt_path.cost_complexity_pruning_path(X_train, y_train)
    alphas = path.ccp_alphas[1:]                           # alpha=0 即未剪枝，跳过

    # 最多抽 20 个候选 alpha，避免搜索过慢
    sample_alphas = alphas[::max(1, len(alphas) // 20)] if len(alphas) > 0 else [0.0]

    best_alpha, best_cv_auc = 0.0, 0.0
    for alpha in sample_alphas:
        dt_tmp = DecisionTreeClassifier(
            class_weight='balanced', random_state=RANDOM_SEED,
            ccp_alpha=alpha, **best_struct
        )
        scores = cross_val_score(dt_tmp, X_train, y_train, cv=cv, scoring='roc_auc')
        if scores.mean() > best_cv_auc:
            best_cv_auc = scores.mean()
            best_alpha  = alpha

    print(f"    [DT-CCP]     最优 ccp_alpha={best_alpha:.6f}  "
          f"AUC(CV)={best_cv_auc:.4f}")

    # ── 最终模型 ───────────────────────────────────────────────────────────
    final = DecisionTreeClassifier(
        class_weight='balanced', random_state=RANDOM_SEED,
        ccp_alpha=best_alpha, **best_struct
    )
    final.fit(X_train, y_train)
    print(f"    [DT-Final]   树深={final.get_depth()}，"
          f"叶节点数={final.get_n_leaves()}")
    return final


# =============================================================================
# 2. 模型评估
# =============================================================================
def evaluate_model(model, X_test, y_test, output_dir: str) -> float:
    """
    评估模型并保存：混淆矩阵、ROC 曲线、分类报告、metrics.csv
    返回 AUC 值（与其他 core/ 模块保持一致）。
    """
    os.makedirs(output_dir, exist_ok=True)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = model.predict(X_test)

    auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, zero_division=0)
    pre = precision_score(y_test, y_pred, zero_division=0)
    f1  = f1_score(y_test, y_pred, zero_division=0)

    print(f"  [DecisionTree] AUC={auc:.4f}  Acc={acc:.4f}  "
          f"Recall={rec:.4f}  Pre={pre:.4f}  F1={f1:.4f}")

    # ── 混淆矩阵 ──────────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['好客户 (0)', '坏客户 (1)'],
    ).plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title('混淆矩阵 - DecisionTree', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
    plt.close(fig)

    # ── ROC 曲线 ──────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC={auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC 曲线 - DecisionTree')
    ax.legend(loc='lower right'); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=150)
    plt.close(fig)

    # ── 分类报告 ──────────────────────────────────────────────────────────
    report = classification_report(
        y_test, y_pred, target_names=['好客户', '坏客户'], digits=4
    )
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w',
              encoding='utf-8') as f:
        f.write('分类报告 - DecisionTree\n' + '=' * 60 + '\n' + report)

    # ── 指标 CSV ──────────────────────────────────────────────────────────
    pd.DataFrame([{
        'accuracy': acc, 'auc': auc,
        'recall': rec, 'precision': pre, 'f1': f1,
    }]).to_csv(os.path.join(output_dir, 'metrics.csv'),
               index=False, encoding='utf-8-sig')

    return auc


# =============================================================================
# 3. 特征重要性
# =============================================================================
def plot_feature_importance(model, feature_names: list, output_dir: str):
    """
    ① Top-20 特征重要性（Gini 基尼增益）柱状图
    ② 树结构可视化（前 3 层，便于论文展示决策路径）
    """
    os.makedirs(output_dir, exist_ok=True)
    importances = model.feature_importances_

    # ── ① Top-20 特征重要性 ─────────────────────────────────────────────
    indices    = np.argsort(importances)[::-1][:20]
    top_names  = [feature_names[i] for i in indices]
    top_values = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(range(len(top_names)), top_values[::-1],
                   color='steelblue', alpha=0.82)
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel('Gini 重要性', fontsize=11)
    ax.set_title('DecisionTree — Top 20 特征重要性',
                 fontsize=12, fontweight='bold')
    for bar, val in zip(bars, top_values[::-1]):
        ax.text(bar.get_width() + 0.0005,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=8)
    ax.grid(True, alpha=0.25, axis='x')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=150)
    plt.close(fig)

    # ── ② 树结构可视化（前 3 层）─────────────────────────────────────────
    try:
        fig, ax = plt.subplots(figsize=(22, 9))
        plot_tree(
            model, ax=ax,
            feature_names=feature_names,
            class_names=['好客户', '坏客户'],
            filled=True, rounded=True,
            max_depth=3, fontsize=7,
            impurity=True, proportion=False,
        )
        ax.set_title('决策树结构（展示前 3 层）', fontsize=13, fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'tree_structure.png'),
                    dpi=100, bbox_inches='tight')
        plt.close(fig)
        print(f"  [DecisionTree] 树结构图已保存 → {output_dir}/tree_structure.png")
    except Exception as e:
        print(f"  [警告] 树结构图生成失败（通常因特征名含特殊字符）：{e}")


# =============================================================================
# 独立运行入口
# =============================================================================
if __name__ == '__main__':
    _SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _SRC_DIR not in sys.path:
        sys.path.insert(0, _SRC_DIR)

    base_input_dir  = os.path.join(_SRC_DIR, '..', 'data', 'input')
    base_output_dir = os.path.join(_SRC_DIR, '..', 'data', 'output')

    user_input = input('请输入数据文件名或完整路径：').strip()
    if os.path.exists(user_input):
        data_path = user_input
    else:
        fname     = user_input if os.path.splitext(user_input)[1] else user_input + '.csv'
        data_path = os.path.join(base_input_dir, fname)

    if not os.path.exists(data_path):
        print(f'❌ 找不到文件：{data_path}')
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f'[加载] {df.shape[0]} 行 × {df.shape[1]} 列')

    _candidates = ['Risk','risk','class','Class','default','Default',
                   'label','Label','target','Target']
    label_col = next((c for c in _candidates if c in df.columns), df.columns[-1])
    print(f'[标签列] {label_col}')

    df = df.drop(columns=[c for c in df.columns if c.lower().startswith('unnamed')])
    X = df.drop(columns=[label_col]).copy()
    y_raw = df[label_col].copy()

    for col in X.select_dtypes(include=[np.number]).columns:
        X[col].fillna(X[col].median(), inplace=True)
    for col in X.select_dtypes(include=['object']).columns:
        X[col].fillna(X[col].mode()[0], inplace=True)
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    le_y = LabelEncoder()
    y    = pd.Series(le_y.fit_transform(y_raw.astype(str)), name=label_col)
    print(f'[标签] 正样本（坏客户）占比：{y.mean():.2%}')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
    )
    print(f'[划分] 训练 {len(X_train)} 条，测试 {len(X_test)} 条\n')

    dataset_name = os.path.splitext(os.path.basename(data_path))[0]
    out_dir = os.path.join(base_output_dir, 'DecisionTree', dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    print('─── 训练决策树 ───')
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test, out_dir)
    plot_feature_importance(model, X.columns.tolist(), out_dir)
    print(f'\n✅ 完成。输出目录：{out_dir}')
