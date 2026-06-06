# -*- coding: utf-8 -*-
import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
TRAIN_SCRIPT = SCRIPT_DIR / "train_rankglu.py"

EPOCH_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)\s+\|\s+train_loss\s+(?P<train_loss>[-+0-9.eE]+)\s+\|\s+"
    r"test_ic\s+(?P<ic>[-+0-9.eE]+)\s+\|\s+test_icir\s+(?P<icir>[-+0-9.eE]+)\s+\|\s+"
    r"test_ric\s+(?P<ric>[-+0-9.eE]+)\s+\|\s+test_ricir\s+(?P<ricir>[-+0-9.eE]+)"
)


def parse_seed_list(value):
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("seed list cannot be empty")
    return seeds


def parse_sections(value):
    allowed = {"main", "ablation", "diagnostic"}
    sections = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(sections) - allowed)
    if unknown:
        raise ValueError(f"unsupported sections: {unknown}; allowed: {sorted(allowed)}")
    return sections or ["main", "ablation", "diagnostic"]


def with_overrides(base, **overrides):
    config = dict(base)
    config.update({key: str(value) for key, value in overrides.items()})
    return config


def common_env(args):
    return {
        "UNIVERSE": args.universe,
        "PREFIX": args.prefix,
        "GPU": str(args.gpu),
        "N_EPOCH": str(args.n_epoch),
        "LR": str(args.lr),
        "D_MODEL": "256",
        "T_NHEAD": "4",
        "S_NHEAD": "2",
        "DROPOUT": "0.5",
        "FEATURE_LAYER_TYPE": "linear",
        "FEATURE_BOTTLENECK": "64",
        "FEATURE_DROPOUT": "0.0",
        "MARKET_GATE_ALPHA": "1.0",
        "MARKET_GATE_NORM": "softmax",
        "TEMPORAL_AGG_TYPE": "attention",
        "TEMPORAL_SCORE_TYPE": "dot",
        "TEMPORAL_GATE_RATIO": "0.1",
        "TEMPORAL_GATE_BOTTLENECK": "",
        "TEMPORAL_LAST_BLEND_RATIO": "0.0",
        "T_SCORE_TYPE": "dot",
        "T_FFN_TYPE": "relu",
        "T_FFN_BOTTLENECK": "170",
        "S_SCORE_DOT_RATIO": "0.1",
        "S_ATTN_NORM": "softmax",
        "S_ATTN_RES_SCALE": "1.0",
        "S_FFN_RES_SCALE": "1.0",
        "S_FFN_RES_SCALE_LEARNABLE": "0",
    }


def rankglu_env():
    return {
        "CS_NORM": "zscore",
        "LOSS_MODE": "mse_ic",
        "IC_WEIGHT": "0.1",
        "STRONGER_HEAD": "1",
        "DECODER_TYPE": "residual_bottleneck_glu",
        "DECODER_BOTTLENECK": "128",
        "DECODER_GLU_SCALE": "1.0",
        "S_SCORE_TYPE": "dot",
        "S_VALUE_GATE_TYPE": "none",
        "S_VALUE_GATE_RATIO": "0.0",
        "S_FFN_TYPE": "relu",
        "S_FFN_BOTTLENECK": "170",
    }


def relation_stress_env():
    return with_overrides(
        rankglu_env(),
        DECODER_BOTTLENECK="64",
        S_SCORE_TYPE="cosine",
        S_VALUE_GATE_TYPE="centered_glu",
        S_VALUE_GATE_RATIO="1.0",
    )


def main_configs():
    base = rankglu_env()
    return [
        {
            "group": "main_comparison",
            "name": "original_backbone",
            "description": "Original reproduced backbone: MSE loss, no CS score normalization, linear decoder.",
            "env": with_overrides(base, CS_NORM="none", LOSS_MODE="mse", STRONGER_HEAD="0", DECODER_TYPE="baseline"),
        },
        {
            "group": "main_comparison",
            "name": "ranking_aware_backbone",
            "description": "Ranking-aware backbone: CS z-score normalization, MSE-IC objective, stronger MLP decoder.",
            "env": with_overrides(base, DECODER_TYPE="stronger"),
        },
        {
            "group": "main_comparison",
            "name": "rankglu",
            "description": "RankGLU: ranking-aware protocol with residual bottleneck GLU score formation.",
            "env": base,
        },
    ]


def ablation_configs():
    full = relation_stress_env()
    return [
        {
            "group": "core_ablation",
            "name": "relation_path_stress_full",
            "description": "Relation-path stress setting: cosine inter-stock score, centered value gate, residual GLU head.",
            "env": full,
        },
        {
            "group": "core_ablation",
            "name": "no_value_gate",
            "description": "Remove centered value-path gate only.",
            "env": with_overrides(full, S_VALUE_GATE_TYPE="none", S_VALUE_GATE_RATIO="0.0"),
        },
        {
            "group": "core_ablation",
            "name": "dot_score",
            "description": "Replace cosine inter-stock score with original dot-product score.",
            "env": with_overrides(full, S_SCORE_TYPE="dot"),
        },
        {
            "group": "core_ablation",
            "name": "no_glu_head",
            "description": "Replace residual bottleneck GLU head with stronger MLP decoder.",
            "env": with_overrides(full, DECODER_TYPE="stronger"),
        },
        {
            "group": "core_ablation",
            "name": "no_relation_path",
            "description": "Remove both cosine scoring and centered value gating; equivalent to head-only RankGLU except bottleneck size.",
            "env": with_overrides(full, S_SCORE_TYPE="dot", S_VALUE_GATE_TYPE="none", S_VALUE_GATE_RATIO="0.0"),
        },
        {
            "group": "core_ablation",
            "name": "no_core_components",
            "description": "Remove relation-path calibration and GLU prediction head; keep ranking-aware protocol.",
            "env": with_overrides(
                full,
                S_SCORE_TYPE="dot",
                S_VALUE_GATE_TYPE="none",
                S_VALUE_GATE_RATIO="0.0",
                DECODER_TYPE="stronger",
            ),
        },
    ]


def diagnostic_configs():
    full = relation_stress_env()
    return [
        {
            "group": "diagnostic_exploration",
            "name": "s_ffn_b170_head_b128",
            "description": "Inter-stock FFN bottleneck GLU, bottleneck 170, residual GLU head b128.",
            "env": with_overrides(
                full,
                DECODER_BOTTLENECK="128",
                S_SCORE_TYPE="dot",
                S_VALUE_GATE_TYPE="none",
                S_VALUE_GATE_RATIO="0.0",
                S_FFN_TYPE="bottleneck_glu",
                S_FFN_BOTTLENECK="170",
            ),
        },
        {
            "group": "diagnostic_exploration",
            "name": "s_ffn_b256_head_b128",
            "description": "Inter-stock FFN bottleneck GLU, bottleneck 256, residual GLU head b128.",
            "env": with_overrides(
                full,
                DECODER_BOTTLENECK="128",
                S_SCORE_TYPE="dot",
                S_VALUE_GATE_TYPE="none",
                S_VALUE_GATE_RATIO="0.0",
                S_FFN_TYPE="bottleneck_glu",
                S_FFN_BOTTLENECK="256",
            ),
        },
        {
            "group": "diagnostic_exploration",
            "name": "value_gate_only_head_b128",
            "description": "Centered value gate only, dot-product score, residual GLU head b128.",
            "env": with_overrides(full, DECODER_BOTTLENECK="128", S_SCORE_TYPE="dot", S_VALUE_GATE_TYPE="centered_glu"),
        },
        {
            "group": "diagnostic_exploration",
            "name": "cosine_value_gate_head_b128",
            "description": "Cosine score plus centered value gate, residual GLU head b128.",
            "env": with_overrides(full, DECODER_BOTTLENECK="128"),
        },
        {
            "group": "diagnostic_exploration",
            "name": "temporal_value_gate_r005",
            "description": "Temporal aggregation value-path gate ratio 0.05 under relation-path stress.",
            "env": with_overrides(full, TEMPORAL_AGG_TYPE="value_gated", TEMPORAL_GATE_RATIO="0.05"),
        },
        {
            "group": "diagnostic_exploration",
            "name": "feature_residual_glu_b64",
            "description": "Feature-layer residual bottleneck GLU replacement.",
            "env": with_overrides(full, FEATURE_LAYER_TYPE="residual_bottleneck_glu", FEATURE_BOTTLENECK="64"),
        },
        {
            "group": "diagnostic_exploration",
            "name": "market_gate_entmax15",
            "description": "Market-conditioned feature gate normalization: entmax15.",
            "env": with_overrides(full, MARKET_GATE_NORM="entmax15"),
        },
        {
            "group": "diagnostic_exploration",
            "name": "market_gate_sparsemax",
            "description": "Market-conditioned feature gate normalization: sparsemax.",
            "env": with_overrides(full, MARKET_GATE_NORM="sparsemax"),
        },
        {
            "group": "diagnostic_exploration",
            "name": "s_attn_entmax15",
            "description": "Inter-stock attention normalization: entmax15.",
            "env": with_overrides(full, S_ATTN_NORM="entmax15"),
        },
    ]


def build_plan(args):
    plan = []
    sections = parse_sections(args.sections)
    if "main" in sections:
        for seed in parse_seed_list(args.main_seeds):
            plan.extend((config, seed) for config in main_configs())
    if "ablation" in sections:
        for seed in parse_seed_list(args.ablation_seeds):
            plan.extend((config, seed) for config in ablation_configs())
    if "diagnostic" in sections:
        seed = int(args.diagnostic_seed)
        plan.extend((config, seed) for config in diagnostic_configs())
    return plan


def run_id(group, name, seed):
    return f"{group}__{name}__seed{seed}"


def parse_best(line, best):
    match = EPOCH_RE.search(line)
    if not match:
        return best
    metrics = {
        "best_epoch": int(match.group("epoch")),
        "train_loss_at_best": float(match.group("train_loss")),
        "best_ic": float(match.group("ic")),
        "best_icir": float(match.group("icir")),
        "best_ric": float(match.group("ric")),
        "best_ricir": float(match.group("ricir")),
    }
    if best is None or metrics["best_ic"] > best["best_ic"]:
        return metrics
    return best


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_one(args, config, seed, index, total, out_dir):
    group = config["group"]
    name = config["name"]
    rid = run_id(group, name, seed)
    log_path = out_dir / "logs" / f"{rid}.log"
    env_path = out_dir / "env" / f"{rid}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.parent.mkdir(parents=True, exist_ok=True)

    selected_env = {}
    selected_env.update(common_env(args))
    selected_env.update(config["env"])
    selected_env["SEED"] = str(seed)
    selected_env["PYTHONHASHSEED"] = str(seed)
    selected_env["VARIANT_NAME"] = rid
    selected_env["PYTHONUNBUFFERED"] = "1"

    env = os.environ.copy()
    env.update(selected_env)
    command = [sys.executable, str(TRAIN_SCRIPT)]
    row = {
        "group": group,
        "name": name,
        "seed": seed,
        "description": config["description"],
        "status": "DRY_RUN" if args.dry_run else "PENDING",
        "best_epoch": "",
        "best_ic": "",
        "best_icir": "",
        "best_ric": "",
        "best_ricir": "",
        "train_loss_at_best": "",
        "seconds": "",
        "return_code": "",
        "log_path": str(log_path),
    }

    with open(env_path, "w", encoding="utf-8") as f:
        json.dump({key: selected_env[key] for key in sorted(selected_env)}, f, indent=2)

    header = f"[{index}/{total}] group={group} | name={name} | seed={seed} | variant={rid}"
    print("=" * 120, flush=True)
    print(header, flush=True)
    print(config["description"], flush=True)
    print(f"log: {log_path}", flush=True)
    if args.dry_run:
        print("DRY RUN: command not executed.", flush=True)
        return row

    start = time.time()
    best = None
    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write(header + "\n")
            log_file.write(config["description"] + "\n")
            log_file.write("Command: " + " ".join(command) + "\n")
            log_file.write("Selected environment:\n")
            for key in sorted(selected_env):
                log_file.write(f"{key}={selected_env[key]}\n")
            log_file.write("=" * 120 + "\n")
            process = subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
                best = parse_best(line, best)
            row["return_code"] = process.wait()
    except KeyboardInterrupt:
        row["status"] = "INTERRUPTED"
        row["seconds"] = f"{time.time() - start:.2f}"
        write_rows(out_dir / "runs_partial.csv", [row])
        raise
    except Exception as exc:
        row["status"] = "ERROR"
        row["return_code"] = "EXCEPTION"
        row["seconds"] = f"{time.time() - start:.2f}"
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"\nRunner exception: {repr(exc)}\n")
        return row

    row["seconds"] = f"{time.time() - start:.2f}"
    row["status"] = "OK" if row["return_code"] == 0 and best is not None else "FAILED"
    if best is not None:
        row.update(best)
    print(
        "RUN RESULT | group={group} | name={name} | seed={seed} | status={status} | "
        "best_ic={best_ic} | best_epoch={best_epoch} | seconds={seconds}".format(**row),
        flush=True,
    )
    return row


def safe_float(value):
    if value == "" or value is None:
        return None
    return float(value)


def mean_std(values):
    values = [value for value in values if value is not None]
    if not values:
        return "", ""
    if len(values) == 1:
        return f"{values[0]:.6f}", "0.000000"
    return f"{statistics.mean(values):.6f}", f"{statistics.stdev(values):.6f}"


def aggregate_rows(rows):
    groups = {}
    for row in rows:
        if row["status"] == "OK":
            groups.setdefault((row["group"], row["name"]), []).append(row)
    output = []
    for (group, name), items in sorted(groups.items()):
        ic_mean, ic_std = mean_std([safe_float(item["best_ic"]) for item in items])
        icir_mean, icir_std = mean_std([safe_float(item["best_icir"]) for item in items])
        ric_mean, ric_std = mean_std([safe_float(item["best_ric"]) for item in items])
        ricir_mean, ricir_std = mean_std([safe_float(item["best_ricir"]) for item in items])
        output.append(
            {
                "group": group,
                "name": name,
                "n_ok": len(items),
                "best_ic_mean": ic_mean,
                "best_ic_std": ic_std,
                "best_icir_mean": icir_mean,
                "best_icir_std": icir_std,
                "best_ric_mean": ric_mean,
                "best_ric_std": ric_std,
                "best_ricir_mean": ricir_mean,
                "best_ricir_std": ricir_std,
            }
        )
    return output


def core_ablation_delta_rows(rows):
    ok = [row for row in rows if row["status"] == "OK" and row["group"] == "core_ablation"]
    by_seed = {}
    for row in ok:
        by_seed.setdefault(int(row["seed"]), {})[row["name"]] = row
    output = []
    for seed, items in sorted(by_seed.items()):
        full = items.get("relation_path_stress_full")
        if full is None:
            continue
        full_ic = float(full["best_ic"])
        for name, row in sorted(items.items()):
            output.append(
                {
                    "seed": seed,
                    "name": name,
                    "full_ic": f"{full_ic:.6f}",
                    "ablation_ic": f"{float(row['best_ic']):.6f}",
                    "delta_vs_full": f"{full_ic - float(row['best_ic']):.6f}",
                    "best_epoch": row["best_epoch"],
                    "description": row["description"],
                }
            )
    return output


def write_outputs(out_dir, rows, args, elapsed):
    write_rows(out_dir / "runs.csv", rows)
    aggregates = aggregate_rows(rows)
    if aggregates:
        write_rows(out_dir / "aggregate.csv", aggregates)
    deltas = core_ablation_delta_rows(rows)
    if deltas:
        write_rows(out_dir / "core_ablation_delta.csv", deltas)

    ok_count = sum(1 for row in rows if row["status"] == "OK")
    dry_run_count = sum(1 for row in rows if row["status"] == "DRY_RUN")
    failed_rows = [row for row in rows if row["status"] not in {"OK", "DRY_RUN"}]
    report_path = out_dir / "final_results.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("RankGLU multi-seed experiment report\n")
        f.write("=" * 100 + "\n")
        f.write("This file is generated by scripts/run_multiseed_protocol.py.\n")
        f.write("If a run is interrupted, this report still keeps completed rows; full logs are under logs/.\n\n")
        f.write("[Protocol]\n")
        for key, value in [
            ("UNIVERSE", args.universe),
            ("PREFIX", args.prefix),
            ("GPU", args.gpu),
            ("N_EPOCH", args.n_epoch),
            ("LR", args.lr),
            ("Main comparison seeds", args.main_seeds),
            ("Core ablation seeds", args.ablation_seeds),
            ("Diagnostic exploration seed", args.diagnostic_seed),
            ("Sections", args.sections),
            ("Total planned/completed rows", len(rows)),
            ("OK rows", ok_count),
            ("Dry-run rows", dry_run_count),
            ("Failed/interrupted rows", len(failed_rows)),
            ("Elapsed seconds", f"{elapsed:.2f}"),
        ]:
            f.write(f"{key}={value}\n")
        f.write("\n[Output files]\n")
        f.write("runs.csv: best IC/ICIR/RIC/RICIR and best epoch for each variant and seed.\n")
        f.write("aggregate.csv: multi-seed means and standard deviations.\n")
        f.write("core_ablation_delta.csv: IC deltas relative to the relation-path stress setting.\n")
        f.write("logs/: full stdout/stderr log for each training run.\n")
        f.write("env/: experiment-only environment settings for each training run.\n\n")
        f.write("[Aggregate results]\n")
        if aggregates:
            header = ["group", "name", "n_ok", "best_ic_mean", "best_ic_std", "best_icir_mean", "best_ric_mean"]
            f.write("\t".join(header) + "\n")
            for row in aggregates:
                f.write("\t".join(str(row[key]) for key in header) + "\n")
        else:
            f.write("No completed aggregate results yet.\n")
        if failed_rows:
            f.write("\n[Failed or interrupted runs]\n")
            header = ["group", "name", "seed", "status", "return_code", "log_path"]
            f.write("\t".join(header) + "\n")
            for row in failed_rows:
                f.write("\t".join(str(row.get(key, "")) for key in header) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run RankGLU main comparison, ablation, and diagnostic experiments.")
    parser.add_argument("--universe", default=os.getenv("UNIVERSE", "csi300"))
    parser.add_argument("--prefix", default=os.getenv("PREFIX", "opensource"))
    parser.add_argument("--gpu", default=os.getenv("GPU", "0"))
    parser.add_argument("--n-epoch", default=os.getenv("N_EPOCH", "40"))
    parser.add_argument("--lr", default=os.getenv("LR", "1e-5"))
    parser.add_argument("--main-seeds", default="0,1,2,3,4")
    parser.add_argument("--ablation-seeds", default="0,1,2")
    parser.add_argument("--diagnostic-seed", default="0")
    parser.add_argument("--sections", default="main,ablation,diagnostic")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"missing train script: {TRAIN_SCRIPT}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "results" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = build_plan(args)
    plan_rows = [
        {"group": config["group"], "name": config["name"], "seed": seed, "description": config["description"]}
        for config, seed in plan
    ]
    write_rows(out_dir / "plan.csv", plan_rows)

    print("=" * 120, flush=True)
    print("RankGLU multi-seed protocol started", flush=True)
    print(f"workdir: {REPO_ROOT}", flush=True)
    print(f"out_dir: {out_dir}", flush=True)
    print(f"sections: {args.sections}", flush=True)
    print(f"total runs: {len(plan)}", flush=True)
    print("=" * 120, flush=True)

    rows = []
    start = time.time()
    for index, (config, seed) in enumerate(plan, start=1):
        row = run_one(args, config, seed, index, len(plan), out_dir)
        rows.append(row)
        write_outputs(out_dir, rows, args, time.time() - start)

    elapsed = time.time() - start
    write_outputs(out_dir, rows, args, elapsed)
    print("=" * 120, flush=True)
    print(f"RankGLU multi-seed protocol finished in {elapsed:.2f}s", flush=True)
    print(f"Run-level results: {out_dir / 'runs.csv'}", flush=True)
    print(f"Final text report: {out_dir / 'final_results.txt'}", flush=True)
    print("=" * 120, flush=True)


if __name__ == "__main__":
    main()
