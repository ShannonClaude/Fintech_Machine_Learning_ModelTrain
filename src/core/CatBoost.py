# =============================================================================
# 毕业设计：基于机器学习的银行客户信用风险评估模型研究与实现
# 数据集：German Credit Data (german_credit_data.csv)
# 核心模型：CatBoost 分类器
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from catboost import CatBoostClassifier

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[提示] 未检测到 shap 库，SHAP 可解释性部分将跳过。")
    print("       可通过 `pip install shap` 安装。\n")

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

# =============================================================================
# 0. 辅助函数：获取用户输入的文件名，并确定输出目录
#    本脚本位于 src/core/，因此：
#      输入目录 → ../../data/input
#      输出目录 → ../../output/CatBoost/<数据集名>
# =============================================================================
def get_input_file_and_output_dir(
    base_input_dir=os.path.join("..", "..", "data", "input"),
    base_output_dir=os.path.join("..", "..", "output"),
):
    while True:
        user_input = input("请输入数据文件名：").strip()
        if not user_input:
            print("输入不能为空，请重新输入。")
            continue

        basename = os.path.basename(user_input)
        if not os.path.splitext(basename)[1]:
            filename = basename + ".csv"
        else:
            filename = basename

        file_path = os.path.join(base_input_dir, filename)
        if os.path.exists(file_path):
            subdir = os.path.splitext(filename)[0]
            output_dir = os.path.join(base_output_dir, "CatBoost", subdir)
            return file_path, output_dir
        else:
            print(f"错误：文件 {file_path} 不存在，请重新输入。")

# =============================================================================
# 1. 数据读取
# =============================================================================
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print("=" * 60)
        print(f"[错误] 找不到数据文件：{os.path.abspath(path)}")
        print("请确认数据文件已放在 data/input/ 目录下。")
        print("=" * 60)
        sys.exit(1)

    df = pd.read_csv(path)
    print(f"[数据加载] 成功读取数据，共 {df.shape[0]} 行 × {df.shape[1]} 列。")
    return df

# =============================================================================
# 2. 自动识别标签列
# =============================================================================
def detect_label_column(df: pd.DataFrame) -> str:
    candidates = ['Risk', 'risk', 'class', 'Class', 'default', 'Default',
                  'label', 'Label', 'target', 'Target']
    for col in candidates:
        if col in df.columns:
            print(f"[标签识别] 检测到标签列：'{col}'")
            return col

    last_col = df.columns[-1]
    print(f"[标签识别] 未找到常见标签列名，使用最后一列：'{last_col}'")
    return last_col

# =============================================================================
# 3. 数据预处理
# =============================================================================
def preprocess(df: pd.DataFrame, label_col: str):
    """
    CatBoost 可以原生处理类别特征（无需 LabelEncoding），
    但为了统一接口，这里仍进行编码。
    """
    drop_cols = [c for c in df.columns if c.lower().startswith('unnamed')]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"[预处理] 已删除无意义列：{drop_cols}")

    X = df.drop(columns=[label_col]).copy()
    y_raw = df[label_col].copy()

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    for col in num_cols:
        missing_cnt = X[col].isna().sum()
        if missing_cnt > 0:
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)
            print(f"[预处理] 数值列 '{col}' 有 {missing_cnt} 个缺失值，用中位数 {median_val:.2f} 填充。")

    for col in cat_cols:
        missing_cnt = X[col].isna().sum()
        if missing_cnt > 0:
            mode_val = X[col].mode()[0]
            X[col].fillna(mode_val, inplace=True)
            print(f"[预处理] 类别列 '{col}' 有 {missing_cnt} 个缺失值，用众数 '{mode_val}' 填充。")

    cat_feature_indices = [X.columns.tolist().index(c) for c in cat_cols]

    le_feat = LabelEncoder()
    for col in cat_cols:
        X[col] = le_feat.fit_transform(X[col].astype(str))
    print(f"[预处理] 已对 {len(cat_cols)} 个类别列进行 LabelEncoding：{cat_cols}")

    unique_labels = y_raw.dropna().unique()
    print(f"[预处理] 标签原始值：{unique_labels}")

    le_label = LabelEncoder()
    y_encoded = le_label.fit_transform(y_raw.astype(str))
    label_classes = le_label.classes_

    if len(label_classes) == 2:
        if label_classes[1].lower() in [k.lower() for k in ['good', 'Good', '0', 'no', 'No', 'safe']]:
            y_encoded = 1 - y_encoded
            print(f"[预处理] 标签已翻转：1 = 坏客户（'{label_classes[0]}'），0 = 好客户（'{label_classes[1]}'）")
        else:
            print(f"[预处理] 标签编码：0 = '{label_classes[0]}'，1 = '{label_classes[1]}'")
    else:
        print(f"[预处理] 标签编码完成，共 {len(label_classes)} 个类别。")

    y = pd.Series(y_encoded, name=label_col)
    print(f"\n[预处理完成] 特征维度：{X.shape}，正样本（坏客户）占比：{y.mean():.2%}")
    return X, y, X.columns.tolist(), cat_feature_indices

# =============================================================================
# 4. 模型训练（CatBoost）
# =============================================================================
def train_model(X_train, y_train, cat_feature_indices):
    """
    构建并训练 CatBoost 分类器。
    常用超参数说明：
      - iterations        : 树的数量（类似 n_estimators）
      - depth             : 树的深度
      - learning_rate     : 学习率
      - l2_leaf_reg       : L2 正则化系数，防止过拟合
      - border_count      : 数值特征的分桶数量
      - scale_pos_weight  : 正负样本权重比，处理类别不平衡
      - verbose           : 训练过程输出频率（0 = 静默）
    """
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
    print(f"\n[模型训练] 训练集样本：好客户 {neg_count} 个，坏客户 {pos_count} 个")
    print(f"[模型训练] scale_pos_weight = {scale_pos_weight:.2f}（自动处理类别不平衡）")

    model = CatBoostClassifier(
        iterations=200,
        depth=4,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        border_count=128,
        scale_pos_weight=scale_pos_weight,
        eval_metric='AUC',
        random_seed=RANDOM_SEED,
        verbose=0,
    )

    model.fit(X_train, y_train)
    print("[模型训练] CatBoost 训练完成！")
    return model

# =============================================================================
# 5. 模型评估
# =============================================================================
def evaluate_model(model, X_test, y_test, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print("\n" + "=" * 60)
    print("                    模型评估结果")
    print("=" * 60)
    print(f"  准确率  (Accuracy)  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  AUC 值  (ROC-AUC)   : {auc:.4f}  ← 银行风控核心指标")
    print("=" * 60)

    print("\n[分类报告]")
    print(classification_report(y_test, y_pred,
                                 target_names=['好客户 (0)', '坏客户 (1)']))

    # ROC 曲线
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
    ax_roc.plot(fpr, tpr, color='steelblue', lw=2,
                label=f'CatBoost ROC (AUC = {auc:.4f})')
    ax_roc.plot([0, 1], [0, 1], color='gray', lw=1.5,
                linestyle='--', label='Random Classifier')
    ax_roc.fill_between(fpr, tpr, alpha=0.1, color='steelblue')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate (误报率)', fontsize=13)
    ax_roc.set_ylabel('True Positive Rate (召回率)', fontsize=13)
    ax_roc.set_title('ROC 曲线 - 信用风险评估模型（CatBoost）', fontsize=15, fontweight='bold')
    ax_roc.legend(loc='lower right', fontsize=12)
    ax_roc.grid(True, alpha=0.3)
    roc_path = os.path.join(output_dir, 'roc_curve.png')
    fig_roc.tight_layout()
    fig_roc.savefig(roc_path, dpi=150)
    plt.close(fig_roc)
    print(f"\n[图表已保存] ROC 曲线 → {roc_path}")

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['好客户 (0)', '坏客户 (1)']
    )
    disp.plot(ax=ax_cm, colorbar=False, cmap='Blues')
    ax_cm.set_title('混淆矩阵', fontsize=14, fontweight='bold')
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    fig_cm.tight_layout()
    fig_cm.savefig(cm_path, dpi=150)
    plt.close(fig_cm)
    print(f"[图表已保存] 混淆矩阵   → {cm_path}")

    return auc

# =============================================================================
# 6. 特征重要性（CatBoost 自带）
# =============================================================================
def plot_feature_importance(model, feature_names, output_dir: str):
    """
    使用 CatBoost 内置的特征重要性（PredictionValuesChange）。
    该指标衡量每个特征对最终预测值变化的平均贡献。
    """
    importances = model.get_feature_importance()
    indices = np.argsort(importances)[::-1][:15]

    fig_fi, ax_fi = plt.subplots(figsize=(9, 6))
    ax_fi.barh(
        [feature_names[i] for i in reversed(indices)],
        [importances[i] for i in reversed(indices)],
        color='steelblue',
    )
    ax_fi.set_xlabel('特征重要性得分（PredictionValuesChange）', fontsize=12)
    ax_fi.set_title('CatBoost 特征重要性', fontsize=14, fontweight='bold')
    ax_fi.grid(True, alpha=0.3, axis='x')
    fi_path = os.path.join(output_dir, 'feature_importance.png')
    fig_fi.tight_layout()
    fig_fi.savefig(fi_path, dpi=150)
    plt.close(fig_fi)
    print(f"[图表已保存] 特征重要性 → {fi_path}")

# =============================================================================
# 7. SHAP 可解释性分析
# =============================================================================
def plot_shap_summary(model, X_train, output_dir: str):
    if not SHAP_AVAILABLE:
        return

    print("\n[SHAP] 正在计算 SHAP 值（可能需要数秒）...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    fig_shap, _ = plt.subplots(figsize=(10, 7))
    shap.summary_plot(sv, X_train, plot_type='dot', show=False, max_display=15)
    plt.title('SHAP Summary Plot - 特征对信用风险的影响', fontsize=14, fontweight='bold')
    shap_path = os.path.join(output_dir, 'shap_summary.png')
    plt.tight_layout()
    plt.savefig(shap_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[图表已保存] SHAP 可解释性 → {shap_path}")

# =============================================================================
# 主程序入口
# =============================================================================
def main():
    print("=" * 60)
    print("  银行客户信用风险评估模型 - 基于 CatBoost")
    print("=" * 60 + "\n")

    data_path, output_dir = get_input_file_and_output_dir()
    global RANDOM_SEED
    RANDOM_SEED = 42

    df = load_data(data_path)
    print(f"\n数据集基本信息：")
    print(df.dtypes.to_string())
    print(f"\n前 3 行预览：\n{df.head(3)}\n")

    label_col = detect_label_column(df)
    X, y, feature_names, cat_feature_indices = preprocess(df, label_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
    )
    print(f"\n[数据划分] 训练集：{len(X_train)} 条，测试集：{len(X_test)} 条")

    model = train_model(X_train, y_train, cat_feature_indices)
    auc   = evaluate_model(model, X_test, y_test, output_dir)

    print("\n[特征重要性] 正在绘制特征重要性图...")
    plot_feature_importance(model, feature_names, output_dir)
    plot_shap_summary(model, X_train, output_dir)

    print("\n" + "=" * 60)
    print(f"  ✅ 全部流程完成！最终 AUC = {auc:.4f}")
    print(f"  📁 所有图表已保存至：{output_dir}/")
    print("     ├── roc_curve.png          ROC 曲线")
    print("     ├── confusion_matrix.png   混淆矩阵")
    print("     ├── feature_importance.png 特征重要性")
    if SHAP_AVAILABLE:
        print("     └── shap_summary.png      SHAP 可解释性")
    print("=" * 60)


if __name__ == "__main__":
    main()
