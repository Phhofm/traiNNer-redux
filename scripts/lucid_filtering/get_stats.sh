#!/bin/bash
FILE=$1
NAME=$2
COL=$3
echo "Processing $NAME from $FILE..."
# Extract column, remove header/None, sort and save
awk -F, -v c="$COL" 'NR>1 {print $c}' "$FILE" | grep -v "None" | sort -n > "${NAME// /_}.tmp"
count=$(wc -l < "${NAME// /_}.tmp")
echo "--- $NAME Summary ---"
echo "Valid Samples: $count"
if [ "$count" -gt 0 ]; then
    echo "Min PSNR: $(head -n 1 "${NAME// /_}.tmp")"
    echo "50th (Median): $(sed -n "$((count/2))p" "${NAME// /_}.tmp")"
    echo "75th Percentile: $(sed -n "$((count*75/100))p" "${NAME// /_}.tmp")"
    echo "90th Percentile: $(sed -n "$((count*90/100))p" "${NAME// /_}.tmp")"
    echo "95th Percentile: $(sed -n "$((count*95/100))p" "${NAME// /_}.tmp")"
    echo "Max PSNR: $(tail -n 1 "${NAME// /_}.tmp")"
fi
# rm "${NAME// /_}.tmp" # Keep for now to be safe
