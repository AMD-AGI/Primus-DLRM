#!/usr/bin/env python3
"""Generate a markdown comparison report from all experiment results."""
import json
import sys
from pathlib import Path


RESULTS_DIR = Path("results")
OUTPUT_PATH = Path("doc/comparison_report.md")

RUNS_OF_INTEREST = [
    # Round 1
    "counter_v1",
    "dlrm_dot",
    "dlrm_dcnv2",
    "onetrans_small",
    "onetrans_base",
    "onetrans_deep",
    "onetrans_wide",
    "onetrans_long",
    # Round 2
    "concat_v2",
    "dcnv2_v2",
    "onetrans_v2",
    "onetrans_v3",
    "onetrans_v4",
    "onetrans_v5",
    "onetrans_v6",
    "onetrans_v7",
]


def load_result(run_name: str) -> dict | None:
    path = RESULTS_DIR / run_name / "results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def get_last_eval(data: dict) -> dict:
    """Extract the final epoch eval results."""
    epochs = data.get("history", {}).get("epoch_results", [])
    for ep in reversed(epochs):
        if "eval" in ep:
            return ep
    return {}


def fmt(val, precision=4):
    if isinstance(val, float):
        return f"{val:.{precision}f}"
    if val is None:
        return "-"
    return str(val)


def delta_pct(val, baseline):
    """Relative improvement as a percentage string."""
    if baseline == 0 or val == 0:
        return "-"
    d = (val - baseline) / baseline * 100
    return f"{d:+.1f}%"


ROUND2_RUNS = {"concat_v2", "dcnv2_v2", "onetrans_v2", "onetrans_v3",
               "onetrans_v4", "onetrans_v5", "onetrans_v6", "onetrans_v7"}


def main():
    rows = []
    for run_name in RUNS_OF_INTEREST:
        data = load_result(run_name)
        if data is None:
            print(f"  [skip] {run_name}: no results.json found")
            continue

        mc = data.get("model_config", {})
        tc = data.get("train_config", {})
        dc = data.get("data_config", {})
        ep = get_last_eval(data)
        gl = ep.get("eval", {}).get("global", {})
        pu = ep.get("eval", {}).get("peruser", {})

        model_type = mc.get("model_type", "dlrm")
        if model_type == "onetrans":
            ot = mc.get("onetrans", {})
            arch_detail = f"d={ot.get('d_model', '?')}, L={ot.get('n_layers', '?')}, ns={ot.get('n_ns_tokens', '?')}"
        else:
            arch_detail = mc.get("interaction_type", "concat_mlp")

        n_epochs = len(data.get("history", {}).get("epoch_results", []))
        counters_str = str(dc.get("counter_windows_days", [30])) if dc.get("enable_counters") else "none"

        rows.append({
            "run": run_name,
            "round": 2 if run_name in ROUND2_RUNS else 1,
            "model": model_type,
            "arch": arch_detail,
            "emb_dim": mc.get("embedding_dim", "?"),
            "hist_len": dc.get("history_length", "?"),
            "epochs": n_epochs,
            "counters": counters_str,
            "lr": tc.get("lr", "?"),
            "contrastive": fmt(tc.get("contrastive_weight", 0)),
            "params_total": data.get("num_params", 0),
            "params_dense": data.get("num_dense_params", 0),
            "throughput": ep.get("throughput_samples_s", 0),
            "loss": ep.get("avg_loss", 0),
            "g_ndcg10": gl.get("ndcg@10", 0),
            "g_ndcg100": gl.get("ndcg@100", 0),
            "g_recall10": gl.get("recall@10", 0),
            "g_recall100": gl.get("recall@100", 0),
            "p_ndcg10": pu.get("ndcg@10", 0),
            "p_ndcg100": pu.get("ndcg@100", 0),
            "p_recall10": pu.get("recall@10", 0),
            "p_recall100": pu.get("recall@100", 0),
            "time_s": data.get("total_time_s", 0),
        })

    if not rows:
        print("No results found!")
        return

    # Sort by peruser ndcg@100 descending
    rows.sort(key=lambda r: r["p_ndcg100"], reverse=True)

    # Find baseline for delta computation
    baseline_p100 = 0
    baseline_g100 = 0
    for r in rows:
        if r["run"] == "counter_v1":
            baseline_p100 = r["p_ndcg100"]
            baseline_g100 = r["g_ndcg100"]
            break

    lines = []
    lines.append("# Model Comparison Report\n")
    lines.append("Dataset: Yambda 50M | batch_size=4096 | counters=[30] | contrastive_weight=0.5\n")
    lines.append("- **Round 1**: 1 epoch, embedding_dim=16, lr=0.001")
    lines.append("- **Round 2**: 3 epochs, embedding_dim=32-64, lr=0.0003-0.0005, warmup=2000 steps")
    lines.append(f"- Reference baseline: `counter_v1` (DLRM + ConcatMLP + counters + contrastive, Round 1)\n")

    # Summary table
    lines.append("## Results Summary (sorted by P-NDCG@100)\n")
    lines.append("| Run | Rnd | Model | Architecture | Emb | Epochs | LR | Dense Params | Thru | Loss | P-NDCG@100 | vs base | G-NDCG@100 | vs base |")
    lines.append("|-----|-----|-------|-------------|-----|--------|-----|-------------|------|------|------------|---------|------------|---------|")

    for r in rows:
        lines.append(
            f"| {r['run']} | R{r['round']} | {r['model']} | {r['arch']} | {r['emb_dim']} | {r['epochs']} "
            f"| {r['lr']} | {r['params_dense']:,} | {r['throughput']:.0f} | {fmt(r['loss'])} "
            f"| {fmt(r['p_ndcg100'])} | {delta_pct(r['p_ndcg100'], baseline_p100)} "
            f"| {fmt(r['g_ndcg100'])} | {delta_pct(r['g_ndcg100'], baseline_g100)} |"
        )

    # Detailed global metrics
    lines.append("\n## Global Evaluation (Top 5000 Popular Items)\n")
    lines.append("| Run | NDCG@10 | NDCG@50 | NDCG@100 | Recall@10 | Recall@50 | Recall@100 |")
    lines.append("|-----|---------|---------|----------|-----------|-----------|------------|")
    for r in rows:
        gl = load_result(r["run"])
        ep = get_last_eval(gl) if gl else {}
        g = ep.get("eval", {}).get("global", {})
        lines.append(
            f"| {r['run']} "
            f"| {fmt(g.get('ndcg@10', 0))} | {fmt(g.get('ndcg@50', 0))} | {fmt(g.get('ndcg@100', 0))} "
            f"| {fmt(g.get('recall@10', 0))} | {fmt(g.get('recall@50', 0))} | {fmt(g.get('recall@100', 0))} |"
        )

    # Detailed peruser metrics
    lines.append("\n## Per-User Evaluation (Top 100 from User's Train Items)\n")
    lines.append("| Run | NDCG@10 | NDCG@50 | NDCG@100 | Recall@10 | Recall@50 | Recall@100 |")
    lines.append("|-----|---------|---------|----------|-----------|-----------|------------|")
    for r in rows:
        gl = load_result(r["run"])
        ep = get_last_eval(gl) if gl else {}
        p = ep.get("eval", {}).get("peruser", {})
        lines.append(
            f"| {r['run']} "
            f"| {fmt(p.get('ndcg@10', 0))} | {fmt(p.get('ndcg@50', 0))} | {fmt(p.get('ndcg@100', 0))} "
            f"| {fmt(p.get('recall@10', 0))} | {fmt(p.get('recall@50', 0))} | {fmt(p.get('recall@100', 0))} |"
        )

    # Training efficiency
    lines.append("\n## Training Efficiency\n")
    lines.append("| Run | Total Params | Embedding Params | Dense Params | Throughput (samples/s) | Total Time (s) |")
    lines.append("|-----|-------------|-----------------|-------------|----------------------|----------------|")
    for r in rows:
        lines.append(
            f"| {r['run']} | {r['params_total']:,} | {r['params_total'] - r['params_dense']:,} "
            f"| {r['params_dense']:,} | {r['throughput']:.0f} | {r['time_s']:.0f} |"
        )

    lines.append("")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines))
    print(f"Report written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
