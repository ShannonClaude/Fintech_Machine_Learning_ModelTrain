# =============================================================================
# 毕业设计：基于机器学习的银行客户信用风险评估模型研究与实现
# 报告生成器：扫描 data/output/ 目录，汇总所有实验图片
# 输出：JSON + Markdown 两份报告
#
# 运行方式（在项目根目录或 src/reporters/ 下均可）：
#   python src/reporters/summarize.py
#   python src/reporters/summarize.py [自定义output路径]
#
# 兼容的输出结构：
#   output/
#   ├── stacking_20260320_142301/          ← Stacking 整体实验
#   │    ├── EXP-1/
#   │    │    ├── XGBoost/<dataset>/       ← 单模型基线图片
#   │    │    └── ...
#   │    ├── EXP-2/                        ← Stacking 各方案图片
#   │    ├── roc_all_experiments.png       ← Stacking 综合对比图
#   │    └── experiment_summary.csv
#   ├── XGBoost/<dataset>/                 ← 单独运行单模型的图片（旧结构兼容）
#   └── ...
# =============================================================================

import os
import sys
import json
from pathlib import Path
from datetime import datetime


# =============================================================================
# 1. 定位 output/ 目录
# =============================================================================
def get_output_dir() -> Path:
    if len(sys.argv) > 1:
        custom = Path(sys.argv[1])
        if custom.exists() and custom.is_dir():
            return custom
        print(f"[警告] 指定路径不存在：{custom}，将使用默认路径。")

    script_dir = Path(__file__).resolve().parent      # src/reporters/
    output_dir = script_dir.parent.parent / "data" / "output"
    return output_dir


# =============================================================================
# 2. 判断一个目录是否是 stacking_<时间戳> 格式
# =============================================================================
def is_stacking_run(d: Path) -> bool:
    name = d.name
    if not name.startswith("stacking_"):
        return False
    # stacking_20260320_142301 → 后缀长度 = 15
    suffix = name[len("stacking_"):]
    return len(suffix) == 15 and suffix[:8].isdigit() and suffix[9:].isdigit()


# =============================================================================
# 3. 扫描单次 Stacking 实验目录
#    返回结构：
#    {
#      'meta': {'run_dir': ..., 'timestamp': ...},
#      'EXP-1': {'ModelName': {'img_stem': 'abs_path', ...}, ...},
#      'EXP-2': {'confusion_matrix': '...', 'roc_curve': '...', ...},
#      ...
#      'summary': {'roc_all_experiments': '...', ...}   ← 根目录综合图
#    }
# =============================================================================
def scan_stacking_run(run_dir: Path) -> dict:
    result = {
        'meta': {
            'run_dir':   str(run_dir),
            'timestamp': run_dir.name.replace("stacking_", ""),
        },
        'summary': {},
    }

    for item in sorted(run_dir.iterdir()):
        # ── 根目录下的综合对比图 ─────────────────────────────────────────────
        if item.is_file() and item.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
            result['summary'][item.stem] = str(item.resolve())
            continue

        if not item.is_dir():
            continue

        exp_name = item.name   # EXP-1, EXP-2, ...

        if exp_name == 'EXP-1':
            # EXP-1 下是 <ModelName>/<dataset>/ 三层
            result['EXP-1'] = {}
            for model_dir in sorted(item.iterdir()):
                if not model_dir.is_dir():
                    continue
                model_name = model_dir.name
                result['EXP-1'][model_name] = {}
                for img in sorted(model_dir.rglob('*')):
                    if img.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
                        result['EXP-1'][model_name][img.stem] = str(img.resolve())
        else:
            # EXP-2~6：目录下直接放图片
            result[exp_name] = {}
            for img in sorted(item.rglob('*')):
                if img.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
                    result[exp_name][img.stem] = str(img.resolve())

    return result


# =============================================================================
# 4. 扫描旧结构单模型目录（output/<ModelName>/<dataset>/）
#    兼容单独运行 core/<Model>.py 时的输出
# =============================================================================
KNOWN_MODELS = {
    'LogisticRegression', 'NaiveBayes', 'SVM',
    'RandomForest', 'XGBoost', 'LightGBM', 'CatBoost',
}

def scan_legacy_models(output_dir: Path) -> dict:
    """扫描直接放在 output/ 根目录下的单模型文件夹（旧结构）。"""
    legacy = {}
    for d in sorted(output_dir.iterdir()):
        if not d.is_dir():
            continue
        if is_stacking_run(d):
            continue
        if d.name in KNOWN_MODELS or any(
            img.suffix.lower() in {'.png', '.jpg', '.jpeg'}
            for img in d.rglob('*')
        ):
            legacy[d.name] = {}
            for img in sorted(d.rglob('*')):
                if img.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
                    # 用相对于 d 的路径作为 key，保留层级信息
                    rel = img.relative_to(d)
                    key = str(rel.with_suffix('')).replace(os.sep, '/')
                    legacy[d.name][key] = str(img.resolve())
    return legacy


# =============================================================================
# 5. 主扫描入口
# =============================================================================
def scan_all(output_dir: Path) -> dict:
    report = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'output_dir':   str(output_dir),
        'stacking_runs': [],
        'legacy_models': {},
    }

    if not output_dir.exists():
        print(f"❌ 错误：找不到目录 '{output_dir}'")
        return report

    # ── Stacking 实验 ─────────────────────────────────────────────────────────
    stacking_dirs = sorted(
        [d for d in output_dir.iterdir() if d.is_dir() and is_stacking_run(d)],
        key=lambda d: d.name,
    )
    for run_dir in stacking_dirs:
        run_data = scan_stacking_run(run_dir)
        report['stacking_runs'].append(run_data)
        print(f"  [Stacking] {run_dir.name}  "
              f"包含 EXP: {[k for k in run_data if k not in ('meta','summary')]}")

    # ── 旧结构单模型（兼容）──────────────────────────────────────────────────
    legacy = scan_legacy_models(output_dir)
    report['legacy_models'] = legacy
    if legacy:
        print(f"  [旧结构单模型] {list(legacy.keys())}")

    return report


# =============================================================================
# 6. 保存 JSON
# =============================================================================
def save_json(report: dict, output_dir: Path) -> Path:
    path = output_dir / "model_evaluation_results.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
    print(f"✅ JSON → {path}")
    return path


# =============================================================================
# 7. 保存 Markdown
# =============================================================================
def _img_line(stem: str, abs_path: str) -> str:
    safe = abs_path.replace('\\', '/')
    return f"![{stem}](file:///{safe})\n\n"


def save_markdown(report: dict, output_dir: Path) -> Path:
    path = output_dir / "model_evaluation_results.md"

    with open(path, 'w', encoding='utf-8') as f:
        f.write("# 模型评估结果汇总\n\n")
        f.write(f"> 生成时间：{report['generated_at']}\n\n")

        # ── Stacking 实验 ─────────────────────────────────────────────────────
        for run in report['stacking_runs']:
            meta = run['meta']
            ts   = meta['timestamp']
            f.write(f"---\n\n## 📦 Stacking 实验  `{ts}`\n\n")

            # EXP-1 单模型基线
            if 'EXP-1' in run:
                f.write("### EXP-1 单模型基线\n\n")
                for model_name, imgs in sorted(run['EXP-1'].items()):
                    if not imgs:
                        continue
                    f.write(f"#### {model_name}\n\n")
                    for stem, p in sorted(imgs.items()):
                        f.write(f"**{stem}**\n\n")
                        f.write(_img_line(stem, p))

            # EXP-2 ~ EXP-N
            for key in sorted(k for k in run if k.startswith('EXP-') and k != 'EXP-1'):
                f.write(f"### {key}\n\n")
                imgs = run[key]
                if not imgs:
                    f.write("*该实验下未找到图片*\n\n")
                    continue
                for stem, p in sorted(imgs.items()):
                    f.write(f"**{stem}**\n\n")
                    f.write(_img_line(stem, p))

            # 综合对比图
            if run.get('summary'):
                f.write("### 综合对比图\n\n")
                for stem, p in sorted(run['summary'].items()):
                    f.write(f"**{stem}**\n\n")
                    f.write(_img_line(stem, p))

        # ── 旧结构单模型（兼容）──────────────────────────────────────────────
        if report['legacy_models']:
            f.write("---\n\n## 🗂 单独运行的单模型（旧结构）\n\n")
            for model_name, imgs in sorted(report['legacy_models'].items()):
                f.write(f"### {model_name}\n\n")
                if not imgs:
                    f.write("*未找到图片*\n\n")
                    continue
                for stem, p in sorted(imgs.items()):
                    f.write(f"**{stem}**\n\n")
                    f.write(_img_line(stem, p))

    print(f"✅ Markdown → {path}")
    return path


# =============================================================================
# 主入口
# =============================================================================
if __name__ == "__main__":
    output_dir = get_output_dir()
    print(f"\n正在扫描：{output_dir}\n")

    report = scan_all(output_dir)

    total_runs   = len(report['stacking_runs'])
    total_legacy = len(report['legacy_models'])

    if total_runs == 0 and total_legacy == 0:
        print("\n⚠️  未找到任何图片，请先运行实验脚本。")
    else:
        print(f"\n扫描完成：{total_runs} 个 Stacking 实验，{total_legacy} 个旧结构单模型目录")
        save_json(report, output_dir)
        save_markdown(report, output_dir)
        print("\n🎉 报告生成完毕！")