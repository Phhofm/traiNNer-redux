#!/usr/bin/env python3
"""
LUCID — Learnable Under-sampling Consistency & Integrity Discovery
================================================================
Master Orchestrator
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd):
    print(f"\n>> Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"!! Error: Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="LUCID Pipeline Orchestrator")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to run")

    # Command: train
    p_train = subparsers.add_parser("train", help="Train the SR Probe")
    p_train.add_argument("--train", required=True, help="Path to training HR images")
    p_train.add_argument("--val", help="Path to validation HR images (optional)")
    p_train.add_argument("--output", default="sr_probe.pth", help="Output weights path")
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--batch", type=int, default=16)

    # Command: stage1
    p_s1 = subparsers.add_parser("stage1", help="Run Stage 1 Signal Filtering")
    p_s1.add_argument("--input", required=True, help="Input dataset directory")
    p_s1.add_argument("--output", required=True, help="Stage 1 output directory")
    p_s1.add_argument("--csv", default="lucid_stage1_stats.csv", help="Stats CSV path")
    p_s1.add_argument("--tile_size", type=int, default=512)
    p_s1.add_argument("--workers", type=int, default=8)

    # Command: stage2
    p_s2 = subparsers.add_parser("stage2", help="Run Stage 2 Consistency Filtering")
    p_s2.add_argument("--input", required=True, help="Stage 1 output directory")
    p_s2.add_argument("--output", required=True, help="Final output directory")
    p_s2.add_argument("--weights", default="sr_probe.pth", help="Probe weights path")
    p_s2.add_argument("--csv", help="Detailed PSNR logging CSV (optional)")
    p_s2.add_argument("--batch_size", type=int, default=32)

    # Command: run-all (The "Intelligent" entry point)
    p_all = subparsers.add_parser(
        "run-all", help="Run Stage 1 and Stage 2 sequentially"
    )
    p_all.add_argument("--input", required=True, help="Input dataset directory")
    p_all.add_argument("--output", required=True, help="Final output directory")
    p_all.add_argument("--weights", default="sr_probe.pth", help="Probe weights path")
    p_all.add_argument("--tile_size", type=int, default=512)
    p_all.add_argument(
        "--temp",
        default="./lucid_stage1_tmp",
        help="Temporary directory for Stage 1 results",
    )

    args = parser.parse_args()

    # Get script directory to find teammates
    script_dir = Path(__file__).parent.absolute()

    if args.command == "train":
        cmd = [
            sys.executable,
            str(script_dir / "train_probe.py"),
            "--train",
            args.train,
            "--output",
            args.output,
            "--epochs",
            str(args.epochs),
            "--batch",
            str(args.batch),
        ]
        if args.val:
            cmd.extend(["--val", args.val])
        run_cmd(cmd)

    elif args.command == "stage1":
        cmd = [
            sys.executable,
            str(script_dir / "lucid_stage1.py"),
            args.input,
            args.output,
            args.csv,
            "--tile_size",
            str(args.tile_size),
            "--workers",
            str(args.workers),
        ]
        run_cmd(cmd)

    elif args.command == "stage2":
        cmd = [
            sys.executable,
            str(script_dir / "lucid_stage2.py"),
            "--input",
            args.input,
            "--output",
            args.output,
            "--weights",
            args.weights,
            "--batch_size",
            str(args.batch_size),
        ]
        if args.csv:
            cmd.extend(["--csv", args.csv])
        run_cmd(cmd)

    elif args.command == "run-all":
        # 1. Stage 1
        s1_out = Path(args.temp)
        s1_csv = s1_out.parent / "lucid_stage1_stats.csv"
        print("\n=== Starting LUCID Stage 1 ===")
        cmd1 = [
            sys.executable,
            str(script_dir / "lucid_stage1.py"),
            args.input,
            str(s1_out),
            str(s1_csv),
            "--tile_size",
            str(args.tile_size),
        ]
        run_cmd(cmd1)

        # 2. Stage 2
        print("\n=== Starting LUCID Stage 2 ===")
        s2_csv = Path(args.output).parent / "lucid_stage2_psnr.csv"
        cmd2 = [
            sys.executable,
            str(script_dir / "lucid_stage2.py"),
            "--input",
            str(s1_out),
            "--output",
            args.output,
            "--weights",
            args.weights,
            "--csv",
            str(s2_csv),
        ]
        run_cmd(cmd2)

        print("\n=== LUCID Pipeline Complete ===")
        print(f"Final dataset: {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
