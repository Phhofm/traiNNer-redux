# 🧶 LUCID Project: Current Work & Roadmap
**Date: Feb 21, 2026**

This file serves as a status report for the LUCID project (Elite SISR Dataset creation).

---

## 🏗️ 1. ULTIMATE ELITE (Ongoing Ingest)
**Status:** Processing remaining "Elite" datasets on the External HDD (One Touch).
- **Core Strategy:** Multi-scale tiling (1.0x, 0.75x, 0.5x, 0.25x) to maximize tile complexity and "yield."
- **Current Run:** `mass_ingest_256.sh` is running with single-worker stability (to prevent HDD I/O lockup).
- **Next Step:** Once ingest completes, run `02_dedupe.py` at **0.94** to prune redundant scales.

## 🚀 2. LUCID-CC0 (Commercial-Safe Project)
**Status:** Ingesting `Spawning/pd12m-full` (12M images) onto the **Crucial X9 SSD**.
- **"Turbo-Stream" Architecture:** Refactored `01_cc0_stream_ingest.py` into a parallel 3-stage pipeline (Fetcher -> CPU Workers -> GPU Consumer) to saturate network and GPU.
- **Fast Resumption:** Switched to Hugging Face `.skip()` logic for near-instant resume after index 21k+.
- **Stability Hardened:**
  - `Image.MAX_IMAGE_PIXELS = None` (Unlocks massive 10k+ resolution images).
  - `os.nice(15)` (Keeps Ubuntu responsive).
  - `HF_HUB_READ_TIMEOUT = 120` (Handles metadata-heavy connections).

## 🧠 3. Strategic Observations (The "Grass Trap")
- **Discovery:** Complexity filtering (ICNet) favors foliage/grass because it is high-frequency noise.
- **Diversity Fix:**
  1. We will balance PD12M with structural sources: **The MET Open Access** (faces/textures) and **LIU4K-v2** (architecture).
  2. Aggressive deduplication will "blindly" prune 80% of redundant grass.

## 📋 4. Future Action Items
1. **Complete PD12M Pilot:** Ingest the first 1M images to verify SSD behavior.
2. **Deduplicate & Finalize:** Use `03_finalize_turbo.py` to move from CSV-registry to physical `.png` folders.
3. **ParagonSR3 Integration:** Evaluate the new `paragonsr3_arch.py` against HAT-L benchmarks using this new dataset.

---
**Handoff Info:** If resuming in a new chat, tell the AI: *"Read scripts/lucid_v2/current_work.md and continue the LUCID-CC0 Turbo Ingest."*
