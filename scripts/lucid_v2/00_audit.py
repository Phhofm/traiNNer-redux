#!/usr/bin/env python3
"""
LUCID v2: Step 00 - Auditor (For Existing Tiles)
===============================================
Use this if you ALREADY have tiles from older scripts and just need to:
1. Verify bitstream integrity (catch corrupted/truncated files).
2. Fix color profiles (convert Palette/CMYK to RGB).
3. Eliminate PIL transparency warnings.

Safe: Uses os.nice(15).
"""

import argparse
import os
import shutil
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = False


def process_image(args_tuple: tuple[Path, bool]) -> dict[str, Any]:
    img_path, fix_enabled = args_tuple
    issues = []
    is_corrupted = False
    was_fixed = False

    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with Image.open(img_path) as img:
                img.load()  # Deep integrity check
                for warning in w:
                    issues.append(f"Warning: {warning.message}")

                if img.mode != "RGB":
                    issues.append(f"Non-standard mode: {img.mode}")
                    if fix_enabled:
                        img.convert("RGB").save(img_path, "PNG")
                        was_fixed = True

                if img.mode == "P" and "transparency" in img.info:
                    issues.append("Metadata: Palette transparency")
                    if fix_enabled and not was_fixed:
                        img.convert("RGB").save(img_path, "PNG")
                        was_fixed = True

    except Exception as e:
        is_corrupted = True
        issues.append(f"CRITICAL: {e}")

    return {
        "path": str(img_path),
        "is_corrupted": is_corrupted,
        "was_fixed": was_fixed,
        "issues": issues,
    }


def main() -> None:
    try:
        os.nice(15)
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="LUCID v2 - Existing Tiles Auditor")
    parser.add_argument("--input", required=True, help="Input folder of existing tiles")
    parser.add_argument(
        "--fix", action="store_true", help="Fix profiles and standardize to RGB"
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count())

    args = parser.parse_args()
    input_dir = Path(args.input)
    image_paths = sorted(input_dir.rglob("*.png"))

    if not image_paths:
        print("No PNG tiles found.")
        return

    print(f"--- LUCID v2 Auditor: {input_dir.name} ---")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        worker_args = [(p, args.fix) for p in image_paths]
        results = list(
            tqdm(
                executor.map(process_image, worker_args),
                total=len(image_paths),
                desc="Auditing Quality",
            )
        )

    corrupted_paths = []
    issue_count = 0
    fixed_count = 0

    for res in results:
        if res["is_corrupted"]:
            corrupted_paths.append(res["path"])
        if res["issues"]:
            issue_count += 1
        if res["was_fixed"]:
            fixed_count += 1

    print(
        f"\nResults: {len(image_paths)} Scanned | {issue_count} Issues | {len(corrupted_paths)} Corrupted"
    )
    if args.fix:
        print(f"Fixed: {fixed_count}")

    if corrupted_paths:
        corr_dir = input_dir / "corrupted_audit"
        corr_dir.mkdir(exist_ok=True)
        for p in corrupted_paths:
            shutil.move(p, str(corr_dir / Path(p).name))
        print(f"Moved corrupted tiles to: {corr_dir}")


if __name__ == "__main__":
    main()
