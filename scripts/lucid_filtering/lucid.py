#!/usr/bin/env python3
"""
LUCID — Learnable Under-sampling Consistency & Integrity Discovery
================================================================
Master Orchestrator (Stability Optimized)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd) -> None:
    print(f"\n>> Running: {' '.join(cmd)}")
    # Inherit existing env but add safety if needed
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"!! Error: Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main() -> None:
    # Reserve one CPU core by default to keep system responsive
    default_workers = max(1, os.cpu_count() - 1)

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
    p_s1 = subparsers.add_parser(
        "stage1", help="Run Stage 1 Signal Filtering (RAM & CPU Safe)"
    )
    p_s1.add_argument("--input", required=True, help="Input dataset directory")
    p_s1.add_argument("--output", required=True, help="Stage 1 output directory")
    p_s1.add_argument("--csv", default="lucid_stage1_stats.csv", help="Stats CSV path")
    p_s1.add_argument("--tile_size", type=int, default=256)
    p_s1.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help=f"Workers (default: {default_workers})",
    )

    # Command: stage2
    p_s2 = subparsers.add_parser(
        "stage2", help="Run Stage 2 Consistency Filtering (I/O Safe)"
    )
    p_s2.add_argument("--input", required=True, help="Stage 1 output directory")
    p_s2.add_argument("--output", required=True, help="Final output directory")
    p_s2.add_argument("--weights", default="sr_probe.pth", help="Probe weights path")
    p_s2.add_argument("--csv", help="Detailed PSNR logging CSV (optional)")
    p_s2.add_argument("--batch_size", type=int, default=32)
    p_s2.add_argument(
        "--workers", type=int, default=4, help="DataLoader workers (default: 4)"
    )

    # Command: run-all (The "Intelligent" entry point)
    p_all = subparsers.add_parser(
        "run-all", help="Run Stage 1 and Stage 2 in batches (Disk Safe)"
    )
    p_all.add_argument("--input", required=True, help="Input dataset directory")
    p_all.add_argument("--output", required=True, help="Final output directory")
    p_all.add_argument("--weights", default="sr_probe.pth", help="Probe weights path")
    p_all.add_argument("--tile_size", type=int, default=256)
    p_all.add_argument(
        "--temp", default="./lucid_stage1_tmp", help="Temp folder for tiles"
    )
    p_all.add_argument(
        "--batch_images",
        type=int,
        default=10000,
        help="Images per batch (default: 10000)",
    )
    p_all.add_argument(
        "--start_batch",
        type=int,
        default=1,
        help="Batch to start from (default: 1)",
    )
    p_all.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help=f"Stage 1 workers (default: {default_workers})",
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
            "--workers",
            str(args.workers),
        ]
        if args.csv:
            cmd.extend(["--csv", args.csv])
        run_cmd(cmd)

    elif args.command == "run-all":
        in_dir = Path(args.input)
        out_dir = Path(args.output)
        temp_dir = Path(args.temp)

        # 0. Find all images
        print(f"Scanning for images in {in_dir}...")
        valid_exts = {".png", ".jpg", ".jpeg"}
        all_images = sorted(
            [str(p) for p in in_dir.rglob("*") if p.suffix.lower() in valid_exts]
        )
        num_imgs = len(all_images)

        if num_imgs == 0:
            print("!! No images found. Exiting.")
            sys.exit(0)

        print(f"Found {num_imgs} images. Batch size: {args.batch_images}")

        # 1. Batch Processing Loop
        num_batches = (num_imgs + args.batch_images - 1) // args.batch_images

        # Pre-cleanup of stats files only if starting fresh
        s1_csv = out_dir.parent / "lucid_stage1_stats.csv"
        s2_csv = out_dir.parent / "lucid_stage2_psnr.csv"
        if args.start_batch <= 1:
            for f in [s1_csv, s2_csv]:
                if f.exists():
                    f.unlink()

        for i in range(args.start_batch - 1, num_batches):
            start_idx = i * args.batch_images
            end_idx = min((i + 1) * args.batch_images, num_imgs)
            batch = all_images[start_idx:end_idx]

            print(f"\n--- Batch {i + 1}/{num_batches} ({len(batch)} images) ---")

            # Write batch file list
            batch_list_path = temp_dir.parent / "lucid_batch_list.txt"
            temp_dir.mkdir(parents=True, exist_ok=True)
            with open(batch_list_path, "w") as f:
                for img in batch:
                    f.write(f"{img}\n")

            # Run Stage 1
            cmd1 = [
                sys.executable,
                str(script_dir / "lucid_stage1.py"),
                args.input,
                str(temp_dir),
                str(s1_csv),
                "--tile_size",
                str(args.tile_size),
                "--workers",
                str(args.workers),
                "--file_list",
                str(batch_list_path),
            ]
            run_cmd(cmd1)

            # Run Stage 2
            cmd2 = [
                sys.executable,
                str(script_dir / "lucid_stage2.py"),
                "--input",
                str(temp_dir),
                "--output",
                str(out_dir),
                "--weights",
                args.weights,
                "--csv",
                str(s2_csv),
            ]
            run_cmd(cmd2)

            # CLEANUP: Delete temporary tiles
            print(f"Cleaning up {temp_dir}...")
            # Delete all png files in temp_dir
            for f in temp_dir.glob("*.png"):
                try:
                    f.unlink()
                except Exception:
                    pass

        # Final cleanup of the batch file
        if batch_list_path.exists():
            batch_list_path.unlink()

        print("\n=== LUCID Pipeline Complete ===")
        print(f"Final dataset: {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
