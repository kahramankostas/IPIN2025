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

# Önceki fonksiyonlarınızda değişiklik yok...
def load_ground_truth(json_file):
    with open(json_file, 'r') as f:
        gt_data = json.load(f)
    return {int(k): v for k, v in gt_data.items()}

def load_clustering_result(csv_file):
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
    cluster_floor_counts = defaultdict(lambda: defaultdict(int))
    common_aps = set(ap_to_floor.keys()) & set(ap_to_cluster.keys())
    for ap in common_aps:
        cluster_floor_counts[ap_to_cluster[ap]][ap_to_floor[ap]] += 1
    cluster_to_floor = {}
    for cluster, floor_counts in cluster_floor_counts.items():
        if not floor_counts: continue
        cluster_to_floor[cluster] = max(floor_counts.items(), key=lambda x: x[1])[0]
    return cluster_to_floor

def create_true_pred_arrays(ap_to_floor, ap_to_cluster, cluster_to_floor):
    common_aps = sorted(list(set(ap_to_floor.keys()) & set(ap_to_cluster.keys())))
    y_true, y_pred_mapped = [], []
    for ap in common_aps:
        cluster_id = ap_to_cluster.get(ap)
        if cluster_id in cluster_to_floor:
            y_true.append(ap_to_floor[ap])
            y_pred_mapped.append(cluster_to_floor[cluster_id])
    return np.array(y_true), np.array(y_pred_mapped)

def calculate_bootstrap_ci(y_true, y_pred, n_iterations=1000):
    n_samples = len(y_true)
    accuracies = []
    for _ in range(n_iterations):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        if len(indices) == 0: continue
        score = accuracy_score(y_true[indices], y_pred[indices])
        accuracies.append(score)
    
    alpha = 0.95
    lower = np.percentile(accuracies, (1.0 - alpha) / 2.0 * 100)
    upper = np.percentile(accuracies, (alpha + (1.0 - alpha) / 2.0) * 100)
    mean_accuracy = np.mean(accuracies)
    return mean_accuracy, lower, upper

def perform_mcnemar_test(y_true, y_pred_model1, y_pred_model2):
    model1_outcomes = (y_pred_model1 == y_true)
    model2_outcomes = (y_pred_model2 == y_true)
    n10 = np.sum(model1_outcomes & ~model2_outcomes)
    n01 = np.sum(~model1_outcomes & model2_outcomes)
    if (n10 + n01) < 25:
        print(f"    (Warning: Low disagreements ({n10 + n01}). McNemar's test may be unreliable.)")
    return mcnemar([[0, n01], [n10, 0]], exact=False, correction=True).pvalue

# --- YENİ GÖRSELLEŞTİRME FONKSİYONU ---
def plot_accuracy_with_error_bars(df_ci, output_file):
    """
    Verilen DataFrame'den hata çubuklu bir çubuk grafik oluşturur ve kaydeder.
    """
    print(f"\n--- '{output_file}' adıyla grafik oluşturuluyor... ---")
    
    # Hata paylarını hesapla (ortalama ile alt/üst sınırlar arasındaki fark)
    # yerr için format: [alt_hata_payları, üst_hata_payları]
    y_err = np.array([
        df_ci['Accuracy'] - df_ci['95% CI Lower'],
        df_ci['95% CI Upper'] - df_ci['Accuracy']
    ])

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df_ci)))

    ax.bar(df_ci['Method'], df_ci['Accuracy'],
           yerr=y_err,
           align='center',
           alpha=0.8,
           ecolor='black',
           capsize=10)

    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('Model Accuracy with 95% Confidence Intervals', fontsize=16, fontweight='bold')
    ax.set_ylim(0, max(df_ci['95% CI Upper']) * 1.1) # Y limitini en yüksek bar'a göre ayarla
    plt.xticks( ha='right', fontsize=12) # X eksenindeki isimleri döndür
    plt.yticks(fontsize=12)
    
    # Barların üzerine doğruluk değerlerini yazdır
    for i, (acc, upper) in enumerate(zip(df_ci['Accuracy'], df_ci['95% CI Upper'])):
        ax.text(i, upper + 0.01, f'{acc:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout() # Grafiğin düzgün sığmasını sağla
    plt.savefig(output_file, dpi=300)
    plt.close()
    print("Grafik başarıyla kaydedildi.")

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
    
    # ... (Veri yükleme ve diğer kısımlar aynı)
    print("Loading data...")
    ap_to_floor = load_ground_truth(gt_file)
    main_data = {'ap_to_cluster': load_clustering_result(main_method_file)}
    main_data['cluster_to_floor'] = map_clusters_to_floors(ap_to_floor, main_data['ap_to_cluster'])

    baseline_data = {}
    for name, path in baseline_files.items():
        if not os.path.exists(path): continue
        ap_to_cluster = load_clustering_result(path)
        cluster_to_floor = map_clusters_to_floors(ap_to_floor, ap_to_cluster)
        baseline_data[name] = {'ap_to_cluster': ap_to_cluster, 'cluster_to_floor': cluster_to_floor}
    
    # --- 2. Güven Aralığı Hesaplama ---
    print("\n--- Calculating 95% Confidence Intervals for Accuracy (Error Bars) ---")
    y_true_main, y_pred_main = create_true_pred_arrays(ap_to_floor, main_data['ap_to_cluster'], main_data['cluster_to_floor'])
    mean_acc_main, lower_main, upper_main = calculate_bootstrap_ci(y_true_main, y_pred_main)
    results_ci = [{'Method': main_method_name, 'Accuracy': mean_acc_main, '95% CI Lower': lower_main, '95% CI Upper': upper_main}]

    for name, data in baseline_data.items():
        y_true_base, y_pred_base = create_true_pred_arrays(ap_to_floor, data['ap_to_cluster'], data['cluster_to_floor'])
        mean_acc, lower, upper = calculate_bootstrap_ci(y_true_base, y_pred_base)
        results_ci.append({'Method': name, 'Accuracy': mean_acc, '95% CI Lower': lower, '95% CI Upper': upper})
        
    df_ci = pd.DataFrame(results_ci).sort_values(by='Accuracy', ascending=False)
    # DataFrame'i ekrana daha okunaklı basalım
    df_ci_printable = df_ci.copy()
    df_ci_printable['95% CI'] = df_ci_printable.apply(lambda row: f"[{row['95% CI Lower']:.4f}, {row['95% CI Upper']:.4f}]", axis=1)
    print(df_ci_printable[['Method', 'Accuracy', '95% CI']].to_string(index=False))

    # --- YENİ ADIM: GRAFİĞİ OLUŞTUR ---
    plot_accuracy_with_error_bars(df_ci, 'accuracy_with_error_bars.pdf')

    # --- 3. P-Değeri Hesaplama ---
    # ... (P-değeri hesaplama kısmı aynı, değişiklik yok)
    print(f"\n--- Calculating P-Values (McNemar's Test vs. {main_method_name}) ---")
    results_pval = []
    for name, data_base in baseline_data.items():
        common_aps = sorted(list(set(ap_to_floor.keys()) & set(main_data['ap_to_cluster'].keys()) & set(data_base['ap_to_cluster'].keys())))
        y_true_common, y_pred_main_common, y_pred_base_common = [], [], []
        for ap in common_aps:
            cluster_main, cluster_base = main_data['ap_to_cluster'].get(ap), data_base['ap_to_cluster'].get(ap)
            if cluster_main in main_data['cluster_to_floor'] and cluster_base in data_base['cluster_to_floor']:
                y_true_common.append(ap_to_floor[ap])
                y_pred_main_common.append(main_data['cluster_to_floor'][cluster_main])
                y_pred_base_common.append(data_base['cluster_to_floor'][cluster_base])
        p_value = perform_mcnemar_test(np.array(y_true_common), np.array(y_pred_main_common), np.array(y_pred_base_common))
        results_pval.append({'Comparison': f"{main_method_name} vs. {name}", 'p-value': p_value})
    df_pval = pd.DataFrame(results_pval)
    print(df_pval.to_string(index=False))


if __name__ == "__main__":
    main()