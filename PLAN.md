# Plan to Address System Freezes During Training and Corruption Check

## Problem Analysis
The user experiences complete system freezes requiring hard reboot when:
1. Training HAT_M_Probe_CC0 model
2. Running corruption check script on external SSD (Crucial X9)

Initial error log shows failure to load image file `/media/phips/Crucial X9/MASTER_ELITE/1047037_pd12m-full.png` with both pyvips and OpenCV.

## Root Cause Hypotheses
1. **USB Subsystem Overload**: High I/O from external SSD causing system unresponsiveness
2. **Corrupted Files**: Specific problematic files causing driver/kernel issues
3. **Resource Exhaustion**: Memory or GPU memory leaks during training
4. **USB Hardware Issues**: Cable, port, or SSD controller problems

## Immediate Actions for Corruption Check (Safe Execution)

### Modified Corruption Check Script Improvements
1. **Reduce Worker Count**: Lower from 8 to 2-4 workers to decrease USB pressure
2. **Add Processing Delay**: Small delay (5-10ms) between file checks to prevent USB saturation
3. **Batch Processing**: Process files in small batches (100-500) with pauses between batches
4. **Memory Optimization**: Use generator instead of building full list in memory
5. **Enhanced Monitoring**: Log system stats (CPU, RAM, I/O wait) periodically
6. **Progressive Testing**: Start with small subset before full scan

### Script Modifications to Implement
```python
# Key changes to make in 09_check_corrupt.py:
# 1. Reduce max_workers to 2-4
# 2. Add time.sleep(0.01) between submissions or use bounded queue
# 3. Process in batches with periodic status logging
# 4. Use generator for file discovery to reduce memory footprint
# 5. Add optional resource monitoring (psutil.cpu_percent, etc.)
```

## Diagnostic Steps Before Full Scan

### 1. USB Connection Verification
- Try different USB 3.0 ports on the system
- Test with alternative USB-C cable if available
- Check dmesg for USB errors after freeze/reboot
- Verify SSD is mounted with appropriate options (noatime, etc.)

### 2. SSD Health Check
```bash
# Install smartmontools if not present
sudo apt install smartmontools
# Check SSD health (if USB bridge supports SMART)
sudo smartctl -a /dev/sdX  # replace with actual device
```

### 3. Resource Baseline Monitoring
Run these in separate terminals before starting corruption check:
```bash
# Monitor I/O wait
iostat -xz 1
# Monitor memory
vmstat 1
# Monitor GPU usage
watch -n 1 nvidia-smi
```

## Progressive Testing Strategy

### Phase 1: Minimal Test
- Create test directory with 10 known good images
- Run modified script with 2 workers, verify no freeze

### Phase 2: Small Subset Test
- Select 1000 random images from MASTER_ELITE
- Run script with 2 workers, 5ms delay, monitor resources
- If successful, increase to 2000 images

### Phase 3: Full Scan with Safeguards
- If Phase 2 passes, run full scan with:
  - 3 workers
  - 10ms delay between checks
  - Batch size of 500 files with 2-second pause between batches
  - Continuous resource logging to file

## Training-Specific Considerations

### If Corruption Check Completes Without Freeze:
1. Remove identified corrupted files from dataset
2. Resume training with same parameters
3. Monitor training with:
   - Reduced batch size if OOM suspected
   - Gradient checking for NaN values
   - VRAM usage logging

### If Training Still Freezes:
1. Check for memory leaks in training code
2. Verify GPU driver stability (consider driver rollback/update)
3. Test with synthetic data to isolate data loading issue
4. Reduce num_workers in DataLoader if applicable

## Emergency Recovery Procedures

### During Freeze:
1. Wait 2-3 minutes to see if system recovers (sometimes USB reset occurs)
2. If no recovery, perform hard reboot
3. Check journalctl after reboot for kernel messages:
   ```bash
   journalctl -b -1 | grep -i -E "usb|xhci|timeout|i/o"
   ```

### Prevention:
1. Set CPU governor to powersave during intensive I/O:
   ```bash
   echo powersave | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
   ```
2. Use nice/ionice for training process:
   ```bash
   nice -n 10 ionice -c2 -n7 python train.py
   ```

## Success Criteria
1. Corruption check completes without system freeze
2. Identified corrupted files are documented and removed
3. Training resumes and completes at least one epoch without freeze
4. System remains responsive during both operations

## Estimated Time Investment
- Script modification: 30 minutes
- Diagnostic tests: 1-2 hours
- Progressive testing: 2-4 hours (depending on dataset size)
- Full corruption scan: 6-24 hours (based on 1.7M files)

## Dependencies
- psutil (already in script)
- tqdm (already in script)
- OpenCV (already in script)
- smartmontools (for SSD health check)