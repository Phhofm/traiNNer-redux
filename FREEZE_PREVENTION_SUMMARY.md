# Freeze Prevention Summary (Ubuntu Budgie)

## Applied Settings (via sysctl)
- vm.swappiness=20 (was 60) - less aggressive swapping
- vm.dirty_ratio=15 (was 20) - earlier cache flush
- vm.dirty_background_ratio=5 (was 10) - smoother I/O

## Immediate Actions Before Training/Corruption Check
1. Monitor I/O wait (key for USB freezes):
   watch -n 1 "iostat -xz 1 | awk '{print \$1\"% Util: \"\$14\"%\"}' | grep -v '^$'"
2. Monitor memory pressure:
   watch -n 1 "free -h | grep -v + && cat /proc/pressure/memory"
3. Run with nice/ionice:
   nice -n 10 ionice -c2 -n7 python your_script.py

## Corruption Check Script (Already Modified)
- Uses 3 workers (down from 8)
- 10ms delay between checks
- Batch size 500 with 2-second pauses
- Logs CPU/RAM before each batch

## If Freezes Persist
1. Try different USB 3.0+ port (blue) on motherboard
2. Use different USB cable
3. Reduce script workers to 2
4. Increase delay to 50ms
5. Increase batch pause to 5 seconds

## Training-Specific
- Monitor VRAM: watch -n 1 "nvidia-smi --query-gpu=memory.used,memory.total --format=csv"
- Reduce DataLoader num_workers to 1-2 if needed
- Consider lowering batch size if VRAM high

## Emergency During Slowdown
1. Switch to TTY (Ctrl+Alt+F2)
2. Run: top (kill memory-heavy processes with k <PID>)
3. If unresponsive after 3 min, hard reset
