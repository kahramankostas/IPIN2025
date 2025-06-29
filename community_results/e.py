import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import argparse
import os
from collections import defaultdict

def parse_arguments():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description='Generate confusion matrix for clustering results')
    parser.add_argument('--gt_file', type=str, required=True, help='JSON file containing ground-truth labels')
    parser.add_argument('--result_file', type=str, required=True, help='CSV file containing clustering results')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save the confusion matrix')
    return parser.parse_args()

def load_ground_truth(json_file):
    """Loads ground truth labels from a JSON file."""
    with open(json_file, 'r') as f:
        gt_data = json.load(f)
    ap_to_floor = {int(k): v for k, v in gt_data.items()}
    return ap_to_floor

def load_clustering_result(csv_file):
    """Loads clustering results from a CSV file."""
    clusters = []
    with open(csv_file, 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            cluster = [int(item.strip()) for item in row if item.strip() and item.strip().isdigit()]
            if cluster:
                clusters.append(cluster)
    ap_to_cluster = {}
    for cluster_id, aps in enumerate(clusters):
        for ap in aps:
            ap_to_cluster[ap] = cluster_id
    return ap_to_cluster, clusters

def map_clusters_to_floors(ap_to_floor, ap_to_cluster):
    """Maps clusters to floors (majority floor in the cluster becomes its label)."""
    cluster_floor_counts = defaultdict(lambda: defaultdict(int))
    common_aps = set(ap_to_floor.keys()) & set(ap_to_cluster.keys())
    for ap in common_aps:
        floor = ap_to_floor[ap]
        cluster = ap_to_cluster[ap]
        cluster_floor_counts[cluster][floor] += 1
    cluster_to_floor = {}
    for cluster, floor_counts in cluster_floor_counts.items():
        cluster_to_floor[cluster] = max(floor_counts.items(), key=lambda x: x[1])[0]
    return cluster_to_floor

def create_true_pred_arrays(ap_to_floor, ap_to_cluster, cluster_to_floor):
    """Creates arrays of true and predicted labels for evaluation."""
    common_aps = sorted(set(ap_to_floor.keys()) & set(ap_to_cluster.keys()))
    y_true = np.array([ap_to_floor[ap] for ap in common_aps])
    y_pred_raw = np.array([ap_to_cluster[ap] for ap in common_aps])
    y_pred_mapped = np.array([cluster_to_floor[ap_to_cluster[ap]] for ap in common_aps])
    return common_aps, y_true, y_pred_raw, y_pred_mapped

def generate_confusion_matrix(y_true, y_pred, output_file, title):
    """Generates and saves a confusion matrix plot with large fonts."""
    cm = confusion_matrix(y_true, y_pred)
    
    # Set up the plot with very large figure size
    plt.figure(figsize=(10, 8))
    
    # Set font sizes to be very large
    plt.rcParams.update({
        'font.size': 24,
        'axes.titlesize': 32,
        'axes.labelsize': 28,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24
    })
    
    # Create heatmap with large annotations
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                annot_kws={'size': 20}, cbar_kws={'shrink': 0.8})
    
    plt.xlabel('Predicted', fontsize=28, fontweight='bold')
    plt.ylabel('True', fontsize=28, fontweight='bold')
    plt.title(title, fontsize=32, fontweight='bold', pad=30)
    
    # Adjust layout to prevent clipping
    plt.tight_layout()
    
    # Save with high DPI for better quality
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Reset font parameters to default
    plt.rcParams.update(plt.rcParamsDefault)

def main():
    """Main function"""
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    ap_to_floor = load_ground_truth(args.gt_file)
    ap_to_cluster, clusters = load_clustering_result(args.result_file)
    cluster_to_floor = map_clusters_to_floors(ap_to_floor, ap_to_cluster)
    
    # Create prediction arrays
    common_aps, y_true, y_pred_raw, y_pred_mapped = create_true_pred_arrays(
        ap_to_floor, ap_to_cluster, cluster_to_floor
    )
    
    # Generate confusion matrices
    print("Generating confusion matrices...")
    
    generate_confusion_matrix(
        y_true, y_pred_raw,
        os.path.join(args.output_dir, 'confusion_matrix_raw.pdf'),
        'Confusion Matrix for Raw Clusters'
    )
    
    generate_confusion_matrix(
        y_true, y_pred_mapped,
        os.path.join(args.output_dir, 'confusion_matrix_mapped.pdf'),
        'Confusion Matrix for Mapped Clusters'
    )
    
    print(f"Confusion matrices saved to {args.output_dir}")
    print("- confusion_matrix_raw.pdf: Raw cluster labels vs true floor labels")
    print("- confusion_matrix_mapped.pdf: Mapped cluster labels vs true floor labels")

if __name__ == "__main__":
    main()