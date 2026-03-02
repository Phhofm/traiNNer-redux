# 🧶 LUCID Project: Current Work & Roadmap
**Date: Feb 28, 2026**

This file serves as a status report for the LUCID project (Elite SISR Dataset creation).

---

## 🏗️ 1. ULTIMATE ELITE (Ongoing Ingest)
**Status:** Processing remaining "Elite" datasets on the External HDD (One Touch).
- **Core Strategy:** Multi-scale tiling (1.0x, 0.75x, 0.5x, 0.25x) to maximize tile complexity and "yield."
- **Current Run:** `mass_ingest_256.sh` is running with single-worker stability.
- **Next Step:** Once ingest completes, the CC0 pool will be merged for a Global Dedupe.

## 🚀 2. LUCID-CC0 (Commercial-Safe Project)
**Status:** Auditing Diversity on 1.7 Million Tiles (**X9 SSD**).
- **"Turbo-Dedupe" (Stability Mode):** Refactored `02_dedupe.py` for massive scale:
  - **Thread-Cap (8 workers):** Limits file-handle and RAM spikes.
  - **Memory Guard:** Implemented `4096px` check to skip "Decompression Bombs".
  - **Aggressive GC:** Every 10 batches, Python's GC is forced to prevent OOM.
  - **Priority:** Use `ionice -c 2 -n 7` and `nice -n 19` for overnight runs.
- **Milestone:** 1TB reached.

## 🧠 3. Strategic Observations (The "Grass Trap")
- **Discovery:** Complexity filtering (ICNet) favors foliage/grass because it is high-frequency noise.
- **Diversity Fix:**
  1. We will balance PD12M with structural sources: **The MET Open Access** (faces/textures) and **LIU4K-v2** (architecture).
  2. Aggressive deduplication will "blindly" prune 80% of redundant grass.

## 📋 4. Status & Action Items
1. **1TB Milestone:** We have reached ~1TB of CC0 tiles on the X9 SSD.
2. **Structural Switch:** If PD12M continues to be "Grass Heavy," we will pivot to **The MET** and **LIU4K** to use the remaining 1.2TB efficiently.
3. **Deduplicate & Finalize:** Once the SSD is full or we have enough variety, run `02_dedupe.py` at **0.94** to prune redundant scales/foliage.
4. **ParagonSR3 Integration:** Evaluate the new architecture against HAT-L benchmarks.

---
**Handoff Info:** If resuming in a new chat, tell the AI: *"Read scripts/lucid_v2/current_work.md and continue the LUCID-CC0 Turbo Ingest."*
