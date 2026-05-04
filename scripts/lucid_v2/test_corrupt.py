import os
import cv2
import psutil
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def check_image(file_path):
    """
    Tries to read the image using OpenCV. 
    Returns the file path if it fails, otherwise None.
    """
    try:
        # cv2.imread returns None if the image is corrupted or empty
        img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return str(file_path)
    except Exception as e:
        return str(file_path)
         
    return None

def main():
    # Set this process to a lower priority to prevent system UI freezes
    # This was an issue previously with the high USB I/O from the Crucial X9
    try:
        os.nice(10)
        p = psutil.Process(os.getpid())
        if hasattr(psutil, "IOPRIO_CLASS_IDLE"):
            p.ionice(psutil.IOPRIO_CLASS_IDLE)
    except Exception as e:
        print(f"Note: Could not set lower IO priority: {e}")

    folder_path = "/tmp/corrupt_test"
    output_txt = "corrupted_corrupt_test.txt"
    
    # Clear the file first if it exists
    if os.path.exists(output_txt):
        os.remove(output_txt)
        
    print(f"Scanning for images in {folder_path}...")
    valid_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    
    # Use generator to avoid storing all paths in memory at once
    def get_image_files():
        base_path = Path(folder_path)
        for f_path in base_path.rglob("*"):
            if f_path.is_file() and f_path.suffix.lower() in valid_extensions:
                yield str(f_path)
    
    # Count files first for progress reporting
    print("Counting images...")
    image_files = list(get_image_files())  # We need the list for counting but will process in batches
    print(f"Found {len(image_files)} images. Starting corruption check...")
    print(f"Results will be written to: {os.path.abspath(output_txt)}")
    
    corrupted_count = 0
    # Reduced workers to decrease USB pressure
    max_workers = 2
    # Delay between checks to prevent USB saturation
    delay_between_checks = 0.01  # 10ms
    # Batch size for periodic reporting and pauses
    batch_size = 2
    
    print(f"Using {max_workers} workers, {delay_between_checks}s delay between checks, batch size {batch_size}")
    
    # Process in batches to allow for pauses and resource monitoring
    for i in range(0, len(image_files), batch_size):
        batch = image_files[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(image_files) + batch_size - 1) // batch_size
        
        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} images)")
        
        # Log system resources before each batch
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        print(f"  CPU: {cpu_percent}%, RAM: {memory.percent}% ({memory.used // (1024**2)}MB/{memory.total // (1024**2)}MB)")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit batch tasks with small delays between submissions
            futures = {}
            for j, path in enumerate(batch):
                future = executor.submit(check_image, path)
                futures[future] = path
                # Small delay between submissions to prevent USB request flooding
                if j < len(batch) - 1:  # Don't delay after the last submission
                    time.sleep(delay_between_checks)
            
            # Process completed tasks
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Batch {batch_num}"):
                res = future.result()
                
                # If the result is not None, the image failed to load
                if res is not None:
                    corrupted_count += 1
                    # Append to the file as soon as we find it, in case the script is interrupted
                    with open(output_txt, "a") as f:
                        f.write(f"{res}\n")
        
        # Pause between batches to let system recover
        if i + batch_size < len(image_files):
            print(f"  Batch {batch_num} complete. Pausing for 2 seconds...")
            time.sleep(2)
    
    if corrupted_count > 0:
        print(f"\nDone! Found {corrupted_count} corrupted images.")
        print(f"The paths have been saved to {os.path.abspath(output_txt)}")
    else:
        print("\nDone! No corrupted images found.")

if __name__ == "__main__":
    main()