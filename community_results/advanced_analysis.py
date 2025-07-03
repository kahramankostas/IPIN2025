import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from statsmodels.stats.contingency_tables import mcnemar
from collections import defaultdict
import os

# Your existing functions (no changes needed here)
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
    return ap_to_cluster

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
        if not floor_counts: continue
        cluster_to_floor[cluster] = max(floor_counts.items(), key=lambda x: x[1])[0]
    return cluster_to_floor

def create_true_pred_arrays(ap_to_floor, ap_to_cluster, cluster_to_floor):
    """Creates arrays of true and predicted labels for evaluation."""
    common_aps = sorted(list(set(ap_to_floor.keys()) & set(ap_to_cluster.keys())))
    y_true = []
    y_pred_mapped = []
    
    for ap in common_aps:
        cluster_id = ap_to_cluster[ap]
        if cluster_id in cluster_to_floor:
            y_true.append(ap_to_floor[ap])
            y_pred_mapped.append(cluster_to_floor[cluster_id])
            
    return np.array(y_true), np.array(y_pred_mapped)

def calculate_bootstrap_ci(y_true, y_pred, n_iterations=1000):
    """Calculates the 95% confidence interval for accuracy using bootstrapping."""
    n_samples = len(y_true)
    accuracies = []
    for _ in range(n_iterations):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        if len(indices) == 0: continue
        bootstrap_true = y_true[indices]
        bootstrap_pred = y_pred[indices]
        score = accuracy_score(bootstrap_true, bootstrap_pred)
        accuracies.append(score)
    
    alpha = 0.95
    lower_bound = np.percentile(accuracies, (1.0 - alpha) / 2.0 * 100)
    upper_bound = np.percentile(accuracies, (alpha + (1.0 - alpha) / 2.0) * 100)
    mean_accuracy = np.mean(accuracies)
    return mean_accuracy, lower_bound, upper_bound

def perform_mcnemar_test(y_true, y_pred_model1, y_pred_model2):
    """Performs McNemar's test to see if two models have a significant difference in error rates."""
    model1_outcomes = (y_pred_model1 == y_true)
    model2_outcomes = (y_pred_model2 == y_true)
    
    n10 = np.sum(model1_outcomes & ~model2_outcomes)
    n01 = np.sum(~model1_outcomes & model2_outcomes)
    
    contingency_table = [[0, n01], [n10, 0]]

    if (n10 + n01) < 25:
        print(f"    (Warning: Low number of disagreements ({n10 + n01}). McNemar's test may be less reliable.)")

    result = mcnemar(contingency_table, exact=False, correction=True)
    return result.pvalue

def main():
    """Main function to run all analyses."""
    
    # --- SETUP: DEFINE YOUR FILE PATHS HERE ---
    gt_file = 'GT.json' # IMPORTANT: Set path to your ground truth file

    # Path to your main method's result file
    main_method_file = 'node2vec.csv'
    main_method_name = 'Node2Vec+KMeans'

    baseline_files = {
        'Fast Greedy': 'fastgreedy.csv',
        'Infomap': 'infomap.csv',
        'Label Prop.': 'labelprop.csv',
        'Leiden': 'leiden.csv',
        'Louvain': 'louvain.csv'
    }
    
    if not os.path.exists(gt_file) or not os.path.exists(main_method_file):
        print("Error: Ground truth or main result file not found. Please update the paths.")
        return

    print("Loading data...")
    ap_to_floor = load_ground_truth(gt_file)
    
    # --- Load all results first ---
    main_data = {
        'ap_to_cluster': load_clustering_result(main_method_file)
    }
    main_data['cluster_to_floor'] = map_clusters_to_floors(ap_to_floor, main_data['ap_to_cluster'])

    baseline_data = {}
    for name, path in baseline_files.items():
        if not os.path.exists(path):
            print(f"Warning: File not found for baseline '{name}'. Skipping.")
            continue
        ap_to_cluster = load_clustering_result(path)
        cluster_to_floor = map_clusters_to_floors(ap_to_floor, ap_to_cluster)
        baseline_data[name] = {'ap_to_cluster': ap_to_cluster, 'cluster_to_floor': cluster_to_floor}

    # --- 2. Calculate Error Bars (Confidence Intervals) ---
    print("\n--- Calculating 95% Confidence Intervals for Accuracy (Error Bars) ---")
    
    y_true_main, y_pred_main = create_true_pred_arrays(ap_to_floor, main_data['ap_to_cluster'], main_data['cluster_to_floor'])
    mean_acc_main, lower_main, upper_main = calculate_bootstrap_ci(y_true_main, y_pred_main)
    
    results_ci = [{'Method': main_method_name, 'Accuracy': f"{mean_acc_main:.4f}", '95% CI': f"[{lower_main:.4f}, {upper_main:.4f}]"}]

    for name, data in baseline_data.items():
        y_true_base, y_pred_base = create_true_pred_arrays(ap_to_floor, data['ap_to_cluster'], data['cluster_to_floor'])
        mean_acc, lower, upper = calculate_bootstrap_ci(y_true_base, y_pred_base)
        results_ci.append({'Method': name, 'Accuracy': f"{mean_acc:.4f}", '95% CI': f"[{lower:.4f}, {upper:.4f}]"})
        
    df_ci = pd.DataFrame(results_ci)
    print(df_ci.to_string(index=False))

    # --- 3. Calculate P-Values (McNemar's Test) ---
    print(f"\n--- Calculating P-Values (McNemar's Test vs. {main_method_name}) ---")
    
    results_pval = []
    for name, data_base in baseline_data.items():
        # --- FIX: Align data points for EACH pairwise comparison ---
        common_aps = sorted(list(
            set(ap_to_floor.keys()) &
            set(main_data['ap_to_cluster'].keys()) &
            set(data_base['ap_to_cluster'].keys())
        ))

        y_true_common, y_pred_main_common, y_pred_base_common = [], [], []
        for ap in common_aps:
            cluster_main = main_data['ap_to_cluster'].get(ap)
            cluster_base = data_base['ap_to_cluster'].get(ap)

            if cluster_main in main_data['cluster_to_floor'] and cluster_base in data_base['cluster_to_floor']:
                y_true_common.append(ap_to_floor[ap])
                y_pred_main_common.append(main_data['cluster_to_floor'][cluster_main])
                y_pred_base_common.append(data_base['cluster_to_floor'][cluster_base])
        
        y_true_np = np.array(y_true_common)
        y_pred_main_np = np.array(y_pred_main_common)
        y_pred_base_np = np.array(y_pred_base_common)
        # --- END OF FIX ---
        
        p_value = perform_mcnemar_test(y_true_np, y_pred_main_np, y_pred_base_np)
        results_pval.append({'Comparison': f"{main_method_name} vs. {name}", 'p-value': p_value})
        
    df_pval = pd.DataFrame(results_pval)
    print(df_pval.to_string(index=False))
    print("\nNote: A p-value < 0.05 indicates the performance difference is statistically significant.")

if __name__ == "__main__":
    main()