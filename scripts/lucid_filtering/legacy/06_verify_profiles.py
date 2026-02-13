import argparse
import os
import shutil
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image, ImageFile
from tqdm import tqdm

# IMPORTANT: Ensure PIL raises errors for truncated files so we can catch them.
# If this were True, PIL would silently load partial images, which we DON'T want for training.
ImageFile.LOAD_TRUNCATED_IMAGES = False

# ========================= AUDIT LOGIC =========================


def process_image(args_tuple: tuple[Path, bool]) -> dict[str, Any]:
    """
    Performs a deep bitstream and profile audit on a single image.
    """
    img_path, fix_enabled = args_tuple

    issues = []
    is_corrupted = False
    was_fixed = False

    try:
        # Standard PIL audit can throw various warnings we want to track
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # 1. Start Deep Bitstream Decode
            # Image.open is lazy; it doesn't read the pixel data yet.
            with Image.open(img_path) as img:
                # Force a full decode of the pixel data to catch truncated files
                # This is the "Integrity Check" segment.
                img.load()

                # Check for metadata/decoding warnings (like the Palette Transparency one)
                for warning in w:
                    issues.append(f"Warning: {warning.message}")

                # 2. Channel & Mode Analysis
                # Training usually expects 3-channel RGB.
                if img.mode not in ["RGB"]:
                    issues.append(f"Non-standard mode: {img.mode}")

                    if fix_enabled:
                        # Standardize to RGB
                        # We discard Alpha for standard SISR if present (RGBA -> RGB)
                        rgb_img = img.convert("RGB")
                        rgb_img.save(img_path, "PNG")
                        was_fixed = True

                # 3. Targeted Fix for Palette Transparency (PIL Warning 1047)
                if img.mode == "P" and "transparency" in img.info:
                    if not any("Palette transparency" in issue for issue in issues):
                        issues.append("Metadata: Palette transparency detected")
                    if fix_enabled and not was_fixed:
                        rgb_img = img.convert("RGB")
                        rgb_img.save(img_path, "PNG")
                        was_fixed = True

    except (OSError, SyntaxError, ValueError) as e:
        # These are usually raised by load() if the file is truncated or has a bad header
        is_corrupted = True
        issues.append(f"CRITICAL Bitstream Corruption: {e}")
    except Exception as e:
        # Catch-all for other unexpected issues
        is_corrupted = True
        issues.append(f"Unexpected Error: {e}")

    return {
        "path": str(img_path),
        "is_corrupted": is_corrupted,
        "was_fixed": was_fixed,
        "issues": issues,
    }


# ========================= MAIN =========================


def main() -> None:
    # Lower process priority to keep system responsive
    try:
        os.nice(15)
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="LUCID Master Elite Unified Auditor: Integrity + Profiles"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Input directory of tiles"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-standardize non-RGB images and fix metadata",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="Number of concurrent processes",
    )

    args = parser.parse_args()
    input_dir = Path(args.input)

    # Discovery
    extensions = [".png", ".jpg", ".jpeg", ".webp"]
    image_paths = []
    for ext in extensions:
        image_paths.extend(list(input_dir.rglob(f"*{ext}")))

    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    print("--- LUCID Master Elite Auditor ---")
    print(f"Input: {input_dir}")
    print(f"Mode:  {'AUDIT + FIX' if args.fix else 'AUDIT ONLY'}")
    print(f"Scanned: {len(image_paths)} images")

    corrupted_paths = []
    issue_paths = []
    fixed_count = 0

    # Prepare data for pool
    worker_args = [(p, args.fix) for p in image_paths]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(
            tqdm(
                executor.map(process_image, worker_args),
                total=len(image_paths),
                desc="Auditing Quality",
            )
        )

    # Sorting results
    for res in results:
        if res["is_corrupted"]:
            corrupted_paths.append(res["path"])
        if res["issues"]:
            issue_paths.append((res["path"], res["issues"]))
        if res["was_fixed"]:
            fixed_count += 1

    print("\n=== AUDIT RESULTS ===")
    print(f"  ✅ High-Quality Tiles:  {len(image_paths) - len(issue_paths)}")
    print(f"  ⚠️  Profile/Meta Issues: {len(issue_paths) - len(corrupted_paths)}")
    print(f"  ❌ Bitstream Corrupted:  {len(corrupted_paths)}")

    if args.fix:
        print(f"  🛠️  Successfully Fixed:  {fixed_count}")

    # Move corrupted to a separate folder so they don't break training
    if corrupted_paths:
        corrupted_dir = input_dir / "corrupted_audit"
        corrupted_dir.mkdir(exist_ok=True)
        print(f"\nMoving {len(corrupted_paths)} corrupted images to {corrupted_dir}...")
        for p in corrupted_paths:
            try:
                shutil.move(p, corrupted_dir / Path(p).name)
            except Exception as e:
                print(f"Error moving {p}: {e}")

    # Detailed logs
    if issue_paths:
        log_file = input_dir / "auditor_report.txt"
        with open(log_file, "w") as f:
            f.write("LUCID MASTER ELITE AUDITOR REPORT\n")
            f.write("=================================\n\n")
            for path, logs in issue_paths:
                f.write(f"{path}:\n")
                for l in logs:
                    f.write(f"  - {l}\n")
        print(f"\nDetailed Auditor Report written to: {log_file}")

    if not corrupted_paths and not (len(issue_paths) - len(corrupted_paths)):
        print("\n✨ Dataset is perfect. No issues detected.")


if __name__ == "__main__":
    main()
