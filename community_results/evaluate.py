import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import confusion_matrix, silhouette_score
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
import argparse
import os
from collections import defaultdict

def parse_arguments():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate clustering performance')
    parser.add_argument('--gt_file', type=str, required=True, help='JSON file containing ground-truth labels')
    parser.add_argument('--result_file', type=str, required=True, help='CSV file containing clustering results')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save the output')
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

def evaluate_clustering(y_true, y_pred, prefix=""):
    """Evaluates clustering performance and returns metrics."""
    results = {}
    results[f"{prefix}ari"] = adjusted_rand_score(y_true, y_pred)
    results[f"{prefix}nmi"] = normalized_mutual_info_score(y_true, y_pred)
    results[f"{prefix}homogeneity"] = homogeneity_score(y_true, y_pred)
    results[f"{prefix}completeness"] = completeness_score(y_true, y_pred)
    results[f"{prefix}v_measure"] = v_measure_score(y_true, y_pred)
    if prefix == "mapped_":
        results[f"{prefix}accuracy"] = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        results[f"{prefix}precision"] = precision
        results[f"{prefix}recall"] = recall
        results[f"{prefix}f1"] = f1
    return results

def generate_confusion_matrix(y_true, y_pred, output_file, title):
    """Generates and saves a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.savefig(output_file)
    plt.close()

def analyze_clusters(clusters, ap_to_floor, output_dir):
    """Analyzes each cluster and visualizes floor distribution."""
    results = []
    for cluster_id, aps in enumerate(clusters):
        floor_counts = defaultdict(int)
        valid_aps = 0
        for ap in aps:
            if ap in ap_to_floor:
                floor_counts[ap_to_floor[ap]] += 1
                valid_aps += 1
        if not valid_aps:
            continue
        percentages = {floor: (count / valid_aps) * 100 for floor, count in floor_counts.items()}
        dominant_floor = max(percentages.items(), key=lambda x: x[1]) if percentages else (None, 0)
        results.append({
            'cluster_id': cluster_id,
            'total_aps': len(aps),
            'valid_aps': valid_aps,
            'floor_counts': dict(floor_counts),
            'floor_percentages': percentages,
            'dominant_floor': dominant_floor[0],
            'dominant_percentage': dominant_floor[1]
        })
        if valid_aps > 0:
            plt.figure(figsize=(10, 6))
            floors = list(floor_counts.keys())
            counts = list(floor_counts.values())
            bars = plt.bar(floors, counts)
            plt.xlabel('Floor')
            plt.ylabel('Number of APs')
            plt.title(f'Cluster {cluster_id} Floor Distribution')
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                         f'{int(height)}', ha='center', va='bottom')
            plt.savefig(os.path.join(output_dir, f'cluster_{cluster_id}_distribution.pdf'))
            plt.close()
    with open(os.path.join(output_dir, 'cluster_analysis.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    return results

def main():
    """Main function"""
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    ap_to_floor = load_ground_truth(args.gt_file)
    ap_to_cluster, clusters = load_clustering_result(args.result_file)
    cluster_to_floor = map_clusters_to_floors(ap_to_floor, ap_to_cluster)
    common_aps, y_true, y_pred_raw, y_pred_mapped = create_true_pred_arrays(
        ap_to_floor, ap_to_cluster, cluster_to_floor
    )
    raw_results = evaluate_clustering(y_true, y_pred_raw, prefix="raw_")
    mapped_results = evaluate_clustering(y_true, y_pred_mapped, prefix="mapped_")
    all_results = {**raw_results, **mapped_results}
    
    print("\n=== CLUSTERING PERFORMANCE METRICS ===")
    print("\n--- Raw Cluster Evaluation ---")
    print(f"Adjusted Rand Index: {raw_results['raw_ari']:.4f}")
    print(f"Normalized Mutual Information: {raw_results['raw_nmi']:.4f}")
    print(f"Homogeneity: {raw_results['raw_homogeneity']:.4f}")
    print(f"Completeness: {raw_results['raw_completeness']:.4f}")
    print(f"V-measure: {raw_results['raw_v_measure']:.4f}")
    
    print("\n--- Cluster-to-Floor Mapped Evaluation ---")
    print(f"Accuracy: {mapped_results['mapped_accuracy']:.4f}")
    print(f"Precision: {mapped_results['mapped_precision']:.4f}")
    print(f"Recall: {mapped_results['mapped_recall']:.4f}")
    print(f"F1-Score: {mapped_results['mapped_f1']:.4f}")
    print(f"Adjusted Rand Index: {mapped_results['mapped_ari']:.4f}")
    print(f"Normalized Mutual Information: {mapped_results['mapped_nmi']:.4f}")
    
    with open(os.path.join(args.output_dir, 'performance_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    
    generate_confusion_matrix(
        y_true, y_pred_raw,
        os.path.join(args.output_dir, 'confusion_matrix_raw.png'),
        'Confusion Matrix for Raw Clusters'
    )
    generate_confusion_matrix(
        y_true, y_pred_mapped,
        os.path.join(args.output_dir, 'confusion_matrix_mapped.png'),
        'Confusion Matrix for Mapped Clusters'
    )
    
    cluster_analysis = analyze_clusters(clusters, ap_to_floor, args.output_dir)
    purities = [item['dominant_percentage'] for item in cluster_analysis]
    avg_purity = sum(purities) / len(purities) if purities else 0
    
    print(f"\n--- Cluster Purity Analysis ---")
    print(f"Average Cluster Purity: {avg_purity:.2f}%")
    print(f"Minimum Cluster Purity: {min(purities):.2f}%")
    print(f"Maximum Cluster Purity: {max(purities):.2f}%")
    
    print(f"\nAll results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
