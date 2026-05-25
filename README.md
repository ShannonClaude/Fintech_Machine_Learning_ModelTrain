# 银行客户信用风险评估（BankCreditRisk） — 中英文说明 / Bilingual README

本仓库包含一个可复现的机器学习实验流水线，用于比较单模型基线与 Stacking 集成，以及若干进阶优化方案（包含 SMOTE、深度特征工程、Optuna 超参搜索与 SHAP 可解释性分析）。

This repository provides a reproducible ML experiment pipeline for bank customer credit risk assessment. It compares single-model baselines with Stacking ensembles and several advanced optimizations (including SMOTE, deep feature engineering, Optuna hyperparameter search, and SHAP explainability).

---

## 中文概览（Chinese summary）

- 主入口：`src/Stacking.py`
- 支持的实验：
  - EXP-1: 单模型基线（8 个模型：LR, NaiveBayes, SVM, DecisionTree, RandomForest, XGBoost, LightGBM, CatBoost）
  - EXP-2 ~ EXP-5: 多种 Stacking 配置（不同基学习器与元学习器）
  - EXP-6: PR 曲线最优阈值调优
  - EXP-7: 深度特征工程 + SMOTE + Optuna（XGBoost）
  - EXP-8: DNN（MLPClassifier）
  - Post: KS / Lift 图与 SHAP 可解释性分析（树模型）

## English summary

- Entry point: `src/Stacking.py`
- Supported experiments:
  - EXP-1: Single-model baseline (8 models: LR, NaiveBayes, SVM, DecisionTree, RandomForest, XGBoost, LightGBM, CatBoost)
  - EXP-2 ~ EXP-5: Several Stacking configurations (different base and meta learners)
  - EXP-6: Threshold tuning via PR curve
  - EXP-7: Deep feature engineering + SMOTE + Optuna (XGBoost)
  - EXP-8: DNN (MLPClassifier)
  - Post: KS / Lift charts and SHAP explainability (for tree models)

---

## 项目结构 / Project structure

- `src/` — 主代码 / main code
  - `Stacking.py` — 主实验脚本（交互式）/ main experiment script (interactive)
  - `core/` — 各单模型实现 / implementations of single models
  - `reporters/` — 报告与汇总辅助脚本 / reporting helpers
- `data/input/` — 输入数据（CSV）/ input CSV files
- `data/output/Stacking/{timestamp}/` — 实验输出目录（按时间戳）/ outputs per run
- `requirements.txt` — 依赖清单 / dependencies

---

## 运行前准备（建议） / Setup (recommended)

1) 在 Windows PowerShell 中创建并启用虚拟环境（示例）：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) 安装依赖：

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3) 可选但推荐的包（用于进阶功能）：

```powershell
python -m pip install shap optuna imbalanced-learn
```

Notes: `requirements.txt` already lists core packages (xgboost, lightgbm, catboost, scikit-learn, etc.). For GPU acceleration, follow each library's official installation guide (e.g., XGBoost with CUDA support).

---

## 如何运行 / How to run

在项目根目录下运行主脚本（会以交互方式提示输入数据文件名或完整路径）：

```powershell
python .\src\Stacking.py
```

提示输入：
- 若数据在 `data/input/`，可直接输入文件名（带或不带 `.csv`）例如 `cs-training_data.csv` 或 `german_credit_data.csv`。
- 或输入完整路径，例如 `E:\BankCreditRisk\data\input\cs-training_data.csv`。

脚本输出目录固定为：

`E:\BankCreditRisk\data\output\Stacking\{timestamp}\`（每次运行按时间戳新建目录）

Outputs are organized into `Chinese/` and `English/` subfolders, each containing per-experiment outputs (EXP-1..EXP-8), summary charts, CSV reports and optional SHAP outputs.

---

## 运行示例 / Example run

```powershell
python .\src\Stacking.py
# 当提示输入时，键入： cs-training_data.csv
```

等待程序完成；视数据量和机器性能，训练与绘图可能耗时较长。

---

## 输出说明 / Important outputs

在 `data/output/Stacking/{timestamp}/` 下（示例文件）：

- `Chinese/` 与 `English/`：中英文两套图表与报告
  - `EXP-1/{ModelName}/`：单模型输出（混淆矩阵、ROC、特征重要性等）
  - `EXP-2/` ~ `EXP-8/`：各集成/优化方案输出
  - `SHAP_analysis/`：若安装 `shap`，生成 SHAP 图（bar, beeswarm, waterfall, dependence）
  - `roc_all_experiments.png`：所有实验 ROC 对比
  - `metrics_bar_chart.png`：AUC/Accuracy/Recall/F1 柱状对比
  - `confusion_matrix_best.png`：最优实验的混淆矩阵
  - `pr_curve_threshold.png`：PR 曲线与阈值权衡
  - `ks_lift_chart.png`：KS 与 Lift 图
  - `experiment_summary.csv`：汇总表（中/英文 CSV）

---

## 可选组件与注意事项 / Optional components & notes

- SHAP: Install `shap` to enable SHAP explainability. SHAP can be memory/time intensive.
- SMOTE: Requires `imbalanced-learn` (package name `imbalanced-learn` / `imblearn`). If absent, the SMOTE step is skipped.
- Optuna: For Bayesian hyperparameter search. If absent, default XGBoost params are used.
- GPU: The script attempts to enable GPU params for XGBoost/LightGBM when available. If your environment lacks GPU support, install CPU-only versions or follow library docs for GPU builds.
- Fonts: The script searches for common Chinese fonts (SimHei / Microsoft YaHei). If not found, plots will use default/English fonts.

---

## 常见问题（FAQ） / FAQ

Q: 报错找不到文件？ / File not found error?
- 确认输入的文件名或路径正确。把数据放入 `data/input/` 并直接输入文件名通常最简单。

Q: 导入某些包时报错？ / ImportError for some packages?
- 先安装 `requirements.txt` 中列出的依赖；对于可选包（shap/optuna/imblearn），根据需要单独安装。脚本会在运行时检测并跳过不可用的可选步骤。

Q: 如何只运行单个模型（例如 XGBoost）？ / How to run a single model (e.g., XGBoost)?
- 在 `src/core/` 下可找到各模型实现（`train_model` / `evaluate_model`）。可自写小脚本调用这些方法以仅运行单模型训练/评估。

---

## 贡献 / Contributing

欢迎提交 issue 与 PR。若需新增模型、调整流水线或改进文档，建议在本地修改 `src/Stacking.py` 与 `src/core/` 并附带复现实验输出样例。

---

感谢使用！祝实验顺利。 / Thank you — good luck with your experiments.

