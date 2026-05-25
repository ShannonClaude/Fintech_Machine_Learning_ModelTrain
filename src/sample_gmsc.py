# =============================================================================
# GMSC 数据集分层抽样脚本
# 从 Give Me Some Credit 原始 15 万条数据中抽取 50,000 条
# 保持正负样本比例（约 6.7% 违约率）
#
# 运行方式：
#   cd src/
#   python sample_gmsc.py
#
# 输出：
#   ../data/input/GMC_sampled_50000.csv
# =============================================================================

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# ── 路径配置 ──────────────────────────────────────────────────────────────────
_SRC_DIR       = os.path.dirname(os.path.abspath(__file__))
_INPUT_DIR     = os.path.join(_SRC_DIR, "..", "data", "input")

SAMPLE_SIZE    = 50_000
RANDOM_SEED    = 42
LABEL_COL      = "SeriousDlqin2yrs"

# ── 支持的原始文件名（按优先级匹配）──────────────────────────────────────────
CANDIDATE_NAMES = [
    "cs-training.csv",
    "GMC.csv",
    "GiveMeSomeCredit.csv",
    "give_me_some_credit.csv",
]

# =============================================================================
def find_raw_file() -> str:
    """在 data/input/ 中寻找原始 GMSC 文件"""
    for name in CANDIDATE_NAMES:
        path = os.path.join(_INPUT_DIR, name)
        if os.path.exists(path):
            return path

    # 兜底：让用户手动输入
    print("[提示] 未在 data/input/ 中找到已知 GMSC 文件名。")
    print(f"       已尝试：{CANDIDATE_NAMES}")
    while True:
        user = input("请输入原始文件完整路径或仅文件名（位于 data/input/）：").strip()
        if os.path.exists(user):
            return user
        candidate = os.path.join(_INPUT_DIR, user)
        if os.path.exists(candidate):
            return candidate
        print(f"[错误] 找不到文件：{user}，请重新输入。")


def main():
    print("=" * 60)
    print("  GMSC 数据集分层抽样")
    print(f"  目标样本量：{SAMPLE_SIZE:,} 条 / 随机种子：{RANDOM_SEED}")
    print("=" * 60)

    # ── 读取原始数据 ──────────────────────────────────────────────────────────
    raw_path = find_raw_file()
    print(f"\n[读取] {raw_path}")
    df = pd.read_csv(raw_path, index_col=0)          # cs-training 第 0 列是行号
    df.columns = df.columns.str.strip()               # 去除列名空格
    print(f"[原始] {len(df):,} 行 × {df.shape[1]} 列")

    # ── 校验标签列 ────────────────────────────────────────────────────────────
    if LABEL_COL not in df.columns:
        print(f"[错误] 未找到标签列 '{LABEL_COL}'，当前列：{df.columns.tolist()}")
        sys.exit(1)

    raw_rate = df[LABEL_COL].mean()
    print(f"[原始] 违约率 = {raw_rate:.4%}  "
          f"（违约 {df[LABEL_COL].sum():,} / 正常 {(df[LABEL_COL]==0).sum():,}）")

    # ── 检查样本量是否合理 ────────────────────────────────────────────────────
    if SAMPLE_SIZE >= len(df):
        print(f"[警告] 目标样本量 {SAMPLE_SIZE:,} ≥ 原始数据量 {len(df):,}，无需抽样，直接保存。")
        df_sampled = df
    else:
        # ── 分层抽样 ──────────────────────────────────────────────────────────
        df_sampled, _ = train_test_split(
            df,
            train_size=SAMPLE_SIZE,
            random_state=RANDOM_SEED,
            stratify=df[LABEL_COL],
        )

    # ── 验证比例 ──────────────────────────────────────────────────────────────
    sampled_rate = df_sampled[LABEL_COL].mean()
    print(f"\n[抽样] {len(df_sampled):,} 行 × {df_sampled.shape[1]} 列")
    print(f"[抽样] 违约率 = {sampled_rate:.4%}  "
          f"（违约 {df_sampled[LABEL_COL].sum():,} / "
          f"正常 {(df_sampled[LABEL_COL]==0).sum():,}）")
    print(f"[验证] 违约率偏差 = {abs(sampled_rate - raw_rate):.6%}  "
          f"{'✅ 正常' if abs(sampled_rate - raw_rate) < 0.005 else '⚠️ 偏差较大'}")

    # ── 保存 ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(_INPUT_DIR, f"GMC_sampled_{SAMPLE_SIZE}.csv")
    df_sampled.reset_index(drop=True).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[保存] → {out_path}")
    print("\n✅ 抽样完成！后续在 Stacking.py 中输入文件名：")
    print(f"   GMC_sampled_{SAMPLE_SIZE}.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
