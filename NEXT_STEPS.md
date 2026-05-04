# Next Steps for Safe Corruption Checking

## Immediate Actions Before Running Full Scan

### 1. Verify USB Connection Stability
- Try different USB 3.0 ports on your system
- If available, test with a different USB-C cable
- After any freeze, check for USB errors:
  ```bash
  journalctl -b -1 | grep -i -E "usb|xhci|timeout|reset"
  ```

### 2. Baseline Resource Monitoring
Run these in separate terminals before starting:
```bash
# Terminal 1: I/O statistics (shows USB/SDD utilization)
watch -n 1 "iostat -xz | grep -E '(sdb|nvme|await)'"

# Terminal 2: Memory and swap usage
watch -n 1 "free -h"

# Terminal 3: CPU load
watch -n 1 "vmstat 1"

# Terminal 4: GPU usage (if applicable)
watch -n 1 "nvidia-smi"
```

### 3. Start with a Small Subset Test
Before scanning all 1.7M images, test with a manageable subset:

```bash
# Create a test list of 1000 random images
find "/media/phips/Crucial X9/MASTER_ELITE" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.webp" -o -name "*.bmp" \) | shuf -n 1000 > /tmp/test_subset.txt

# Modify the script to use this list temporarily
# Edit 09_check_corrupt.py and replace the file discovery section with:
# image_files = [line.strip() for line in open("/tmp/test_subset.txt")]

# Then run:
cd /home/phips/Documents/GitHub/traiNNer-redux/scripts/lucid_v2
python3 09_check_corrupt.py
```

## If Subset Test Succeeds (No Freeze):

### Progressive Scaling
1. **Increase subset size**: Try 5000, then 10000 images
2. **Adjust parameters cautiously**:
   - First try: Increase `max_workers` to 4
   - If stable: Decrease `delay_between_checks` to 0.005
   - If stable: Decrease batch pause to 1 second
   - If stable: Increase `batch_size` to 1000

## If System Still Freezes During Testing:

### More Conservative Settings
Modify these values in 09_check_corrupt.py:
- `max_workers = 2` (or even 1)
- `delay_between_checks = 0.05` (50ms)
- `batch_size = 100`
- `time.sleep(5)` between batches

### Alternative Approach: Copy to Internal SSD
If USB subsystem seems to be the bottleneck:
1. Copy a portion of dataset to internal SSD:
   ```bash
   rsync -avh --progress "/media/phips/Crucial X9/MASTER_ELITE/" "/home/phips/dataset_subset/" --include="*/" --include="*.png" --include="*.jpg" --include="*.jpeg" --include="*.webp" --include="*.bmp" --exclude="*" --max-files=5000
   ```
2. Run corruption check on the internal copy
3. If successful, gradually increase the copied amount

## Monitoring During Execution

Watch for these warning signs:
- **I/O wait % consistently > 20%** in iostat
- **Available memory dropping below 1GB**
- **Swap usage increasing steadily**
- **USB reset messages in dmesg/journalctl**

If you see these, pause the script immediately with Ctrl+C and let the system recover.

## After Corruption Check Completes

1. Review `corrupted_images.txt` - if it's large (>100 files), consider checking your SSD health
2. Remove corrupted files from your training dataset:
   ```bash
   # Create a backup list first
   cp corrupted_images.txt corrupted_images.txt.backup
   
   # Remove files (use with extreme caution!)
   # while read line; do rm "$line"; done < corrupted_images.txt
   ```
   **Better approach**: Move to a quarantine folder instead of deleting immediately
   ```bash
   mkdir -p /media/phips/Crucial X9/MASTER_ELITE_QUARANTINE
   while read line; do mv "$line" /media/phips/Crucial X9/MASTER_ELITE_QUARANTINE/; done < corrupted_images.txt
   ```
3. Resume training with the cleaned dataset

## Training-Specific Freeze Prevention

If training still freezes after corruption check:
1. Reduce DataLoader `num_workers` to 2 or 1
2. Monitor VRAM usage with: `watch -n 1 "nvidia-smi --query-gpu=memory.used,memory.total --format=csv"`
3. Consider reducing batch size if VRAM is consistently high
4. Add gradient checking in training code to detect NaN values early
5. Use mixed precision training if supported to reduce VRAM pressure

## Emergency Procedures

If system freezes:
1. Wait 3-5 minutes - sometimes USB controller resets itself
2. If no recovery, perform hard reset
3. Immediately after reboot, check logs:
   ```bash
   journalctl -b -1 | grep -i -E "usb|xhci|timeout|i/o|mem" | tail -20
   ```
4. Consider checking SSD health:
   ```bash
   sudo apt install smartmontools
   sudo smartctl -a /dev/sdX  # Replace X with your SSD device letter
   ```

## Expected Timeline

With conservative settings (3 workers, 10ms delay, 500 batch size, 2s pause):
- Estimated time for 1.7M images: 8-24 hours
- Actual time depends on:
  - USB transfer speed of your Crucial X9
  - Percentage of corrupted files (early termination skips remaining files in batch)
  - System specifications (CPU, memory)

## Success Criteria

✅ Corruption check completes without system freeze
✅ Corrupted files identified and quarantined
✅ Training resumes and completes at least one epoch without freeze
✅ System remains responsive (mouse/keyboard work) during both operations

---

Remember: The goal is not speed, but preventing system freezes. It's better to take 24 hours and keep your system usable than to rush and require frequent hard reboots which can corrupt your filesystem.