import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_checkpoint(path: Path) -> Dict:
    if not path.exists():
        return {}
    return torch.load(path, map_location="cpu")


def load_error_analysis(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)


def to_float(value) -> float:
    if value is None:
        return float("nan")
    return float(value)


def load_run(name: str, checkpoint_dir: Path) -> Dict:
    config = load_json(checkpoint_dir / "run_config.json")
    status = load_json(checkpoint_dir / "status.json")
    sft_state = load_checkpoint(checkpoint_dir / "sft_last.pt")
    dpo_state = load_checkpoint(checkpoint_dir / "dpo_last.pt")
    error_df = load_error_analysis(checkpoint_dir / "error_analysis.csv")

    sft_dev_accuracies = sft_state.get("sft_dev_accuracies", [])
    dpo_dev_accuracies = dpo_state.get("dpo_dev_accuracies", [])

    best_sft_accuracy = (
        dpo_state.get("best_sft_accuracy")
        or sft_state.get("best_sft_accuracy")
        or status.get("best_sft_accuracy")
    )
    best_dpo_accuracy = dpo_state.get("best_dpo_accuracy") or status.get("best_dpo_accuracy")

    return {
        "name": name,
        "checkpoint_dir": checkpoint_dir,
        "config": config,
        "status": status,
        "sft_dev_accuracies": [float(v) for v in sft_dev_accuracies],
        "dpo_dev_accuracies": [float(v) for v in dpo_dev_accuracies],
        "best_sft_accuracy": to_float(best_sft_accuracy),
        "best_dpo_accuracy": to_float(best_dpo_accuracy),
        "error_df": error_df,
    }


def build_stage_curve_df(runs: List[Dict], key: str) -> pd.DataFrame:
    rows = []
    for run in runs:
        for epoch, accuracy in enumerate(run[key], start=1):
            rows.append({"run": run["name"], "epoch": epoch, "accuracy": accuracy})
    return pd.DataFrame(rows)


def save_visualizations(runs: List[Dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Paraphrase Checkpoint Analysis: Smoke vs Full", fontsize=16, fontweight="bold")

    sft_df = build_stage_curve_df(runs, "sft_dev_accuracies")
    if not sft_df.empty:
        sns.lineplot(data=sft_df, x="epoch", y="accuracy", hue="run", marker="o", ax=axes[0, 0])
        axes[0, 0].set_title("SFT Dev Accuracy by Epoch")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Accuracy")
    else:
        axes[0, 0].text(0.5, 0.5, "No SFT curve data found", ha="center", va="center")
        axes[0, 0].set_axis_off()

    dpo_df = build_stage_curve_df(runs, "dpo_dev_accuracies")
    if not dpo_df.empty:
        sns.lineplot(data=dpo_df, x="epoch", y="accuracy", hue="run", marker="o", ax=axes[0, 1])
        axes[0, 1].set_title("DPO Dev Accuracy by Epoch")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy")
    else:
        axes[0, 1].text(0.5, 0.5, "No DPO curve data found", ha="center", va="center")
        axes[0, 1].set_axis_off()

    best_rows = []
    for run in runs:
        best_rows.append({"run": run["name"], "stage": "SFT best", "accuracy": run["best_sft_accuracy"]})
        best_rows.append({"run": run["name"], "stage": "DPO best", "accuracy": run["best_dpo_accuracy"]})
    best_df = pd.DataFrame(best_rows).dropna(subset=["accuracy"])
    if not best_df.empty:
        sns.barplot(data=best_df, x="run", y="accuracy", hue="stage", ax=axes[1, 0])
        axes[1, 0].set_title("Best Accuracy Comparison")
        axes[1, 0].set_xlabel("Run")
        axes[1, 0].set_ylabel("Accuracy")
    else:
        axes[1, 0].text(0.5, 0.5, "No best-metric data found", ha="center", va="center")
        axes[1, 0].set_axis_off()

    error_frames = []
    for run in runs:
        if run["error_df"] is not None and "confidence" in run["error_df"].columns:
            df = run["error_df"].copy()
            df["run"] = run["name"]
            error_frames.append(df)
    if error_frames:
        all_errors = pd.concat(error_frames, ignore_index=True)
        sns.histplot(
            data=all_errors,
            x="confidence",
            hue="run",
            bins=30,
            stat="density",
            common_norm=False,
            element="step",
            ax=axes[1, 1],
        )
        axes[1, 1].set_title("Error Confidence Distribution")
        axes[1, 1].set_xlabel("Confidence (abs(yes_logits - no_logits))")
        axes[1, 1].set_ylabel("Density")
    else:
        axes[1, 1].text(0.5, 0.5, "No error_analysis.csv found", ha="center", va="center")
        axes[1, 1].set_axis_off()

    plt.tight_layout()
    fig.savefig(output_dir / "smoke_vs_full_visualization.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_findings(runs: List[Dict], output_dir: Path) -> None:
    rows = []
    lines = ["# Paraphrase Analysis Findings", ""]

    for run in runs:
        best_sft = run["best_sft_accuracy"]
        best_dpo = run["best_dpo_accuracy"]
        gain = best_dpo - best_sft if pd.notna(best_sft) and pd.notna(best_dpo) else float("nan")
        error_count = int(len(run["error_df"])) if run["error_df"] is not None else 0
        high_conf_errors = 0
        if run["error_df"] is not None and "confidence" in run["error_df"].columns:
            high_conf_errors = int((run["error_df"]["confidence"] >= 3.0).sum())

        rows.append(
            {
                "run": run["name"],
                "checkpoint_dir": str(run["checkpoint_dir"]),
                "sft_epochs_recorded": len(run["sft_dev_accuracies"]),
                "dpo_epochs_recorded": len(run["dpo_dev_accuracies"]),
                "best_sft_accuracy": best_sft,
                "best_dpo_accuracy": best_dpo,
                "dpo_minus_sft": gain,
                "error_count": error_count,
                "high_confidence_errors_ge_3": high_conf_errors,
            }
        )

        lines.extend(
            [
                f"## {run['name']}",
                f"- Checkpoint: `{run['checkpoint_dir']}`",
                f"- Best SFT accuracy: `{best_sft:.4f}`" if pd.notna(best_sft) else "- Best SFT accuracy: unavailable",
                f"- Best DPO accuracy: `{best_dpo:.4f}`" if pd.notna(best_dpo) else "- Best DPO accuracy: unavailable",
                f"- DPO gain over SFT: `{gain:+.4f}`" if pd.notna(gain) else "- DPO gain over SFT: unavailable",
                f"- Error rows analyzed: `{error_count}`",
                f"- High-confidence errors (confidence >= 3.0): `{high_conf_errors}`",
                "",
            ]
        )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)

    if len(summary_df) == 2:
        run_to_row = {row["run"]: row for _, row in summary_df.iterrows()}
        run_names = list(run_to_row.keys())
        left_name, right_name = run_names[0], run_names[1]
        left_row, right_row = run_to_row[left_name], run_to_row[right_name]
        if pd.notna(left_row["best_dpo_accuracy"]) and pd.notna(right_row["best_dpo_accuracy"]):
            delta = right_row["best_dpo_accuracy"] - left_row["best_dpo_accuracy"]
            lines.extend(
                [
                    "## Cross-run comparison",
                    f"- DPO best accuracy difference ({right_name} - {left_name}): `{delta:+.4f}`",
                    "",
                ]
            )

    (output_dir / "findings.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize and summarize paraphrase smoke/full checkpoint metrics."
    )
    parser.add_argument("--smoke-dir", required=True, type=Path, help="Path to smoke checkpoint directory.")
    parser.add_argument("--full-dir", required=True, type=Path, help="Path to full checkpoint directory.")
    parser.add_argument(
        "--output-dir",
        default=Path("Report/paraphrase_visualization"),
        type=Path,
        help="Directory for generated figures and summary files.",
    )
    parser.add_argument("--smoke-label", default="smoke", help="Label for smoke run in plots/reports.")
    parser.add_argument("--full-label", default="full", help="Label for full run in plots/reports.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = [
        load_run(args.smoke_label, args.smoke_dir),
        load_run(args.full_label, args.full_dir),
    ]
    save_visualizations(runs, args.output_dir)
    save_findings(runs, args.output_dir)
    print(f"Saved visualization and findings under: {args.output_dir}")


if __name__ == "__main__":
    main()
