#!/usr/bin/env python3
"""
Merge CSV score files into master lineage CSV.
Adds complexity_score and other metrics from score datasets to master lineage.
"""

import csv

def load_score_data(score_file_path):
    """
    Load score CSV into a dictionary mapping tile_name to metrics.
    
    Returns:
        dict: {tile_name: {metric_name: value}}
    """
    score_data = {}
    with open(score_file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tile_name = row['tile_name']
            # Store all metrics except tile_name
            metrics = {k: v for k, v in row.items() if k != 'tile_name'}
            score_data[tile_name] = metrics
    return score_data

def main():
    # Paths
    master_lineage_path = '/media/phips/Crucial X9/master_lineage.csv'
    pd12m_scores_path = '/media/phips/Crucial X9/lucid_cc0/pd12m-full/pd12m-full_scores.csv'
    liu4k_scores_path = '/media/phips/Crucial X9/lucid_cc0/liu4k/training_scores.csv'
    uhdiqa_scores_path = '/media/phips/Crucial X9/lucid_cc0/uhdiqa/uhdiqatraining_scores.csv'
    output_path = '/media/phips/Crucial X9/master_lineage_complexity.csv'
    
    # Load score datasets
    print("Loading score datasets...")
    pd12m_scores = load_score_data(pd12m_scores_path)
    liu4k_scores = load_score_data(liu4k_scores_path)
    uhdiqa_scores = load_score_data(uhdiqa_scores_path)
    
    print(f"Loaded {len(pd12m_scores)} pd12m-full scores")
    print(f"Loaded {len(liu4k_scores)} liu4k scores")
    print(f"Loaded {len(uhdiqa_scores)} uhdiqa scores")
    
    # Process master lineage
    print("Processing master lineage...")
    with open(master_lineage_path, 'r') as infile, open(output_path, 'w', newline='') as outfile:
        reader = csv.DictReader(infile)
        # Ensure we have fieldnames (CSV should have header)
        if reader.fieldnames is None:
            raise ValueError("Master lineage CSV has no header")
        fieldnames = list(reader.fieldnames) + ['complexity_score', 'entropy', 'lap_var', 'grad_energy', 
                                        'blockiness', 'noise_ratio', 'aliasing']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        matched = 0
        unmatched = 0
        
        for row in reader:
            source_dataset = row['source_dataset']
            original_name = row['original_name']
            
            # Determine which score dataset to use
            score_dict = None
            if source_dataset == 'pd12m-full':
                score_dict = pd12m_scores
            elif source_dataset == 'liu4k':
                score_dict = liu4k_scores
            elif source_dataset == 'uhdiqa':
                score_dict = uhdiqa_scores
            else:
                # Unknown dataset, leave metrics empty
                score_dict = {}
            
            # Look up metrics
            metrics = score_dict.get(original_name, {})
            
            # Add metrics to row
            for metric in ['complexity_score', 'entropy', 'lap_var', 'grad_energy', 
                         'blockiness', 'noise_ratio', 'aliasing']:
                row[metric] = metrics.get(metric, '')
            
            if metrics:
                matched += 1
            else:
                unmatched += 1
            
            writer.writerow(row)
    
    print(f"Finished. Matched: {matched}, Unmatched: {unmatched}")
    print(f"Output written to: {output_path}")

if __name__ == '__main__':
    main()