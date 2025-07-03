import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from scipy import stats
from scipy.stats import chi2_contingency
import argparse
import os
from collections import defaultdict
import pandas as pd
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

def parse_arguments():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description='Generate confusion matrix and statistical validation for clustering results')
    parser.add_argument('--gt_file', type=str, required=True, help='JSON file containing ground-truth labels')
    parser.add_argument('--result_file', type=str, required=True, help='CSV file containing clustering results')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save the results')
    parser.add_argument('--baseline_file', type=str, help='CSV file containing baseline clustering results for comparison')
    parser.add_argument('--n_bootstrap', type=int, default=1000, help='Number of bootstrap samples for confidence intervals')
    parser.add_argument('--confidence_level', type=float, default=0.95, help='Confidence level for intervals')
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

def bootstrap_metric(y_true, y_pred, metric_func, n_bootstrap=1000, confidence_level=0.95):
    """Calculate bootstrap confidence intervals for a metric."""
    n_samples = len(y_true)
    bootstrap_scores = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sampling with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        try:
            score = metric_func(y_true_boot, y_pred_boot)
            bootstrap_scores.append(score)
        except:
            continue
    
    bootstrap_scores = np.array(bootstrap_scores)
    alpha = 1 - confidence_level
    lower_percentile = (alpha/2) * 100
    upper_percentile = (100 - alpha/2)
    
    ci_lower = np.percentile(bootstrap_scores, lower_percentile)
    ci_upper = np.percentile(bootstrap_scores, upper_percentile)
    mean_score = np.mean(bootstrap_scores)
    std_score = np.std(bootstrap_scores)
    
    return mean_score, std_score, ci_lower, ci_upper

def calculate_metrics_with_ci(y_true, y_pred, n_bootstrap=1000, confidence_level=0.95):
    """Calculate performance metrics with confidence intervals."""
    metrics = {}
    
    # Define metric functions
    metric_functions = {
        'accuracy': lambda yt, yp: accuracy_score(yt, yp),
        'precision_macro': lambda yt, yp: precision_score(yt, yp, average='macro', zero_division=0),
        'recall_macro': lambda yt, yp: recall_score(yt, yp, average='macro', zero_division=0),
        'f1_macro': lambda yt, yp: f1_score(yt, yp, average='macro', zero_division=0),
        'precision_weighted': lambda yt, yp: precision_score(yt, yp, average='weighted', zero_division=0),
        'recall_weighted': lambda yt, yp: recall_score(yt, yp, average='weighted', zero_division=0),
        'f1_weighted': lambda yt, yp: f1_score(yt, yp, average='weighted', zero_division=0),
    }
    
    for metric_name, metric_func in metric_functions.items():
        mean_score, std_score, ci_lower, ci_upper = bootstrap_metric(
            y_true, y_pred, metric_func, n_bootstrap, confidence_level
        )
        metrics[metric_name] = {
            'mean': mean_score,
            'std': std_score,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    
    return metrics

def mcnemar_test(y_true, y_pred1, y_pred2):
    """Manual implementation of McNemar's test."""
    # Create contingency table for McNemar's test
    correct1 = (y_true == y_pred1).astype(int)
    correct2 = (y_true == y_pred2).astype(int)
    
    # Count discordant pairs
    only_1_correct = np.sum((correct1 == 1) & (correct2 == 0))
    only_2_correct = np.sum((correct1 == 0) & (correct2 == 1))
    
    # McNemar's test statistic with continuity correction
    if only_1_correct + only_2_correct == 0:
        chi2_stat = 0
        p_value = 1.0
    else:
        chi2_stat = (abs(only_1_correct - only_2_correct) - 1)**2 / (only_1_correct + only_2_correct)
        p_value = 1 - stats.chi2.cdf(chi2_stat, 1)
    
    return chi2_stat, p_value, only_1_correct, only_2_correct

def statistical_significance_test(y_true, y_pred1, y_pred2, test_type='mcnemar'):
    """Perform statistical significance test between two predictions."""
    results = {}
    
    if test_type == 'mcnemar':
        # McNemar's test for paired predictions
        chi2_stat, p_value, only_1_correct, only_2_correct = mcnemar_test(y_true, y_pred1, y_pred2)
        
        results['test_type'] = 'McNemar'
        results['chi2_statistic'] = chi2_stat
        results['p_value'] = p_value
        results['discordant_pairs'] = {
            'only_method1_correct': only_1_correct,
            'only_method2_correct': only_2_correct
        }
        
    elif test_type == 'paired_t':
        # Paired t-test on accuracy differences
        acc1 = (y_true == y_pred1).astype(float)
        acc2 = (y_true == y_pred2).astype(float)
        
        t_stat, p_value = stats.ttest_rel(acc1, acc2)
        
        results['test_type'] = 'Paired t-test'
        results['t_statistic'] = t_stat
        results['p_value'] = p_value
        results['mean_difference'] = np.mean(acc1 - acc2)
        
    elif test_type == 'wilcoxon':
        # Wilcoxon signed-rank test (non-parametric alternative)
        acc1 = (y_true == y_pred1).astype(float)
        acc2 = (y_true == y_pred2).astype(float)
        
        try:
            stat, p_value = stats.wilcoxon(acc1, acc2, alternative='two-sided')
            results['test_type'] = 'Wilcoxon Signed-Rank'
            results['statistic'] = stat
            results['p_value'] = p_value
        except ValueError:
            # All differences are zero
            results['test_type'] = 'Wilcoxon Signed-Rank'
            results['statistic'] = 0
            results['p_value'] = 1.0
            
    return results

def generate_confusion_matrix_with_ci(y_true, y_pred, output_file, title, n_bootstrap=1000):
    """Generates confusion matrix with confidence intervals for each cell."""
    cm = confusion_matrix(y_true, y_pred)
    n_samples = len(y_true)
    
    # Bootstrap confidence intervals for confusion matrix cells
    cm_bootstrap = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        cm_boot = confusion_matrix(y_true_boot, y_pred_boot, labels=np.unique(y_true))
        cm_bootstrap.append(cm_boot)
    
    cm_bootstrap = np.array(cm_bootstrap)
    cm_std = np.std(cm_bootstrap, axis=0)
    
    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16
    })
    
    # Original confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                annot_kws={'size': 14}, cbar_kws={'shrink': 0.8})
    ax1.set_xlabel('Predicted', fontsize=18, fontweight='bold')
    ax1.set_ylabel('True', fontsize=18, fontweight='bold')
    ax1.set_title(f'{title}\n(Counts)', fontsize=20, fontweight='bold')
    
    # Confusion matrix with standard errors
    annotations = []
    for i in range(cm.shape[0]):
        row = []
        for j in range(cm.shape[1]):
            row.append(f'{cm[i,j]}\n±{cm_std[i,j]:.1f}')
        annotations.append(row)
    
    sns.heatmap(cm_std, annot=annotations, fmt='', cmap='Reds', ax=ax2,
                annot_kws={'size': 12}, cbar_kws={'shrink': 0.8})
    ax2.set_xlabel('Predicted', fontsize=18, fontweight='bold')
    ax2.set_ylabel('True', fontsize=18, fontweight='bold')
    ax2.set_title(f'{title}\n(Counts ± Bootstrap SE)', fontsize=20, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.rcParams.update(plt.rcParamsDefault)

def generate_performance_report(metrics, output_file, method_name="Method"):
    """Generate a comprehensive performance report with confidence intervals."""
    report = f"Performance Report for {method_name}\n"
    report += "=" * 50 + "\n\n"
    
    for metric_name, metric_data in metrics.items():
        mean_val = metric_data['mean']
        std_val = metric_data['std']
        ci_lower = metric_data['ci_lower']
        ci_upper = metric_data['ci_upper']
        
        report += f"{metric_name.replace('_', ' ').title()}:\n"
        report += f"  Mean: {mean_val:.4f} ± {std_val:.4f}\n"
        report += f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]\n\n"
    
    with open(output_file, 'w') as f:
        f.write(report)
    
    return report

def create_performance_comparison_plot(metrics1, metrics2, output_file, 
                                     method1_name="Method 1", method2_name="Method 2"):
    """Create a comparison plot of performance metrics with error bars."""
    metric_names = list(metrics1.keys())
    means1 = [metrics1[m]['mean'] for m in metric_names]
    means2 = [metrics2[m]['mean'] for m in metric_names]
    stds1 = [metrics1[m]['std'] for m in metric_names]
    stds2 = [metrics2[m]['std'] for m in metric_names]
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    bars1 = ax.bar(x - width/2, means1, width, yerr=stds1, label=method1_name, 
                   capsize=5, alpha=0.8)
    bars2 = ax.bar(x + width/2, means2, width, yerr=stds2, label=method2_name, 
                   capsize=5, alpha=0.8)
    
    ax.set_xlabel('Metrics', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title('Performance Comparison with Error Bars', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metric_names], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function with enhanced statistical validation"""
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading data...")
    # Load main method data
    ap_to_floor = load_ground_truth(args.gt_file)
    ap_to_cluster, clusters = load_clustering_result(args.result_file)
    cluster_to_floor = map_clusters_to_floors(ap_to_floor, ap_to_cluster)
    
    # Create prediction arrays
    common_aps, y_true, y_pred_raw, y_pred_mapped = create_true_pred_arrays(
        ap_to_floor, ap_to_cluster, cluster_to_floor
    )
    
    print("Calculating performance metrics with confidence intervals...")
    # Calculate metrics with confidence intervals
    metrics_mapped = calculate_metrics_with_ci(
        y_true, y_pred_mapped, args.n_bootstrap, args.confidence_level
    )
    
    # Generate enhanced confusion matrices
    print("Generating enhanced confusion matrices...")
    generate_confusion_matrix_with_ci(
        y_true, y_pred_mapped,
        os.path.join(args.output_dir, 'confusion_matrix_enhanced.pdf'),
        'Enhanced Confusion Matrix with Bootstrap SE',
        args.n_bootstrap
    )
    
    # Generate performance report
    report = generate_performance_report(
        metrics_mapped, 
        os.path.join(args.output_dir, 'performance_report.txt'),
        "Proposed Method"
    )
    print("\nPerformance Report:")
    print(report)
    
    # If baseline is provided, perform comparison
    if args.baseline_file:
        print("Loading baseline data for comparison...")
        ap_to_cluster_baseline, _ = load_clustering_result(args.baseline_file)
        cluster_to_floor_baseline = map_clusters_to_floors(ap_to_floor, ap_to_cluster_baseline)
        _, _, _, y_pred_baseline = create_true_pred_arrays(
            ap_to_floor, ap_to_cluster_baseline, cluster_to_floor_baseline
        )
        
        # Calculate baseline metrics
        metrics_baseline = calculate_metrics_with_ci(
            y_true, y_pred_baseline, args.n_bootstrap, args.confidence_level
        )
        
        # Statistical significance testing
        print("Performing statistical significance tests...")
        sig_test_mcnemar = statistical_significance_test(
            y_true, y_pred_mapped, y_pred_baseline, 'mcnemar'
        )
        sig_test_ttest = statistical_significance_test(
            y_true, y_pred_mapped, y_pred_baseline, 'paired_t'
        )
        sig_test_wilcoxon = statistical_significance_test(
            y_true, y_pred_mapped, y_pred_baseline, 'wilcoxon'
        )
        
        # Create comparison plot
        create_performance_comparison_plot(
            metrics_mapped, metrics_baseline,
            os.path.join(args.output_dir, 'performance_comparison.pdf'),
            "Proposed Method", "Baseline Method"
        )
        
        # Enhanced comparison report
        comparison_report = f"\nStatistical Significance Testing:\n"
        comparison_report += "=" * 40 + "\n"
        comparison_report += f"McNemar's Test p-value: {sig_test_mcnemar['p_value']:.6f}\n"
        comparison_report += f"Paired t-test p-value: {sig_test_ttest['p_value']:.6f}\n"
        comparison_report += f"Wilcoxon test p-value: {sig_test_wilcoxon['p_value']:.6f}\n"
        comparison_report += f"Significance level: α = {1-args.confidence_level}\n\n"
        
        alpha = 1 - args.confidence_level
        significant_tests = []
        
        if sig_test_mcnemar['p_value'] < alpha:
            significant_tests.append("McNemar's test")
        if sig_test_ttest['p_value'] < alpha:
            significant_tests.append("Paired t-test")
        if sig_test_wilcoxon['p_value'] < alpha:
            significant_tests.append("Wilcoxon test")
            
        if significant_tests:
            comparison_report += f"Result: Statistically significant difference detected by: {', '.join(significant_tests)}\n"
        else:
            comparison_report += "Result: No statistically significant difference detected by any test\n"
        
        print(comparison_report)
        
        # Save detailed comparison
        with open(os.path.join(args.output_dir, 'statistical_comparison.txt'), 'w') as f:
            f.write(comparison_report)
            f.write(f"\nDetailed Results:\n")
            f.write(f"McNemar's Chi-square statistic: {sig_test_mcnemar['chi2_statistic']:.6f}\n")
            f.write(f"McNemar's discordant pairs: Method1 only correct={sig_test_mcnemar['discordant_pairs']['only_method1_correct']}, "
                   f"Method2 only correct={sig_test_mcnemar['discordant_pairs']['only_method2_correct']}\n")
            f.write(f"Paired t-test statistic: {sig_test_ttest['t_statistic']:.6f}\n")
            f.write(f"Mean accuracy difference: {sig_test_ttest['mean_difference']:.6f}\n")
            f.write(f"Wilcoxon statistic: {sig_test_wilcoxon['statistic']:.6f}\n")
    
    print(f"\nAll results saved to {args.output_dir}")
    print("Generated files:")
    print("- confusion_matrix_enhanced.pdf: Enhanced confusion matrix with bootstrap SE")
    print("- performance_report.txt: Detailed performance metrics with confidence intervals")
    if args.baseline_file:
        print("- performance_comparison.pdf: Comparison plot with error bars")
        print("- statistical_comparison.txt: Statistical significance test results")

if __name__ == "__main__":
    main()