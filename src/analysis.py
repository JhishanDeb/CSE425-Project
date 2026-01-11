import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score

def perform_clustering(data, method='kmeans', n_clusters=5):
    """
    Perform clustering using specified method.
    """
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(data)
    elif method == 'agglomerative':
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(data)
    elif method == 'dbscan':
        # Epsilon and min_samples need tuning based on distance distribution
        model = DBSCAN(eps=0.5, min_samples=5)
        labels = model.fit_predict(data)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
        
    return labels

def compute_purity(y_true, y_pred):
    """
    Calculate cluster purity.
    Purity = sum(max(intersection(cluster, class))) / total_samples
    """
    # Create contingency matrix
    contingency_matrix = np.zeros((len(np.unique(y_true)), len(np.unique(y_pred))))
    
    # Map labels to 0..N indices
    u_true, y_true_idx = np.unique(y_true, return_inverse=True)
    u_pred, y_pred_idx = np.unique(y_pred, return_inverse=True)
    
    for t, p in zip(y_true_idx, y_pred_idx):
        if p >= 0: # handling DBSCAN noise if -1
            contingency_matrix[t, p] += 1
            
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def evaluate_clustering(data, labels, true_labels=None):
    """
    Calculate Silhouette, CH, DB, ARI, NMI, Purity
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    # If using DBSCAN, filter noise (-1) for some metrics if strictly required, 
    # but here we compute on all for consistency unless single cluster.
    if n_clusters < 2:
        return {'sil': 0, 'ch': 0, 'db': 0, 'ari': 0, 'nmi': 0, 'purity': 0}
        
    sil = silhouette_score(data, labels)
    ch = calinski_harabasz_score(data, labels)
    db = davies_bouldin_score(data, labels)
    
    ari = 0
    nmi = 0
    purity = 0
    
    if true_labels is not None:
        # Align lengths if needed (though should match)
        if len(true_labels) == len(labels):
            ari = adjusted_rand_score(true_labels, labels)
            nmi = normalized_mutual_info_score(true_labels, labels)
            purity = compute_purity(true_labels, labels)
        
    return {'sil': sil, 'ch': ch, 'db': db, 'ari': ari, 'nmi': nmi, 'purity': purity}

def plot_latent_space(latent_data, labels=None, method='tsne', save_path=None, title="Latent Space"):
    print(f"Computing {method.upper()} projection...")
    
    if method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    else:
        raise ValueError("Method must be 'tsne' or 'pca'")
        
    embedding = reducer.fit_transform(latent_data)
    
    plt.figure(figsize=(10, 8))
    if labels is not None:
        unique_l = np.unique(labels)
        # Handle string labels or int
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels if np.issubdtype(labels.dtype, np.number) else None, cmap='tab10', alpha=0.6, s=15)
        
        # If labels are strings (e.g. genre names), we can't use 'c' directly easily without encoding.
        # But 'labels' commonly ints here. If strings, seaborn is better.
        
        if len(unique_l) < 20: 
            plt.legend(*scatter.legend_elements(), title="Clusters")
    else:
        plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6, s=15)
        
    plt.title(title)
    plt.xlabel(f"{method.upper()} 1")
    plt.ylabel(f"{method.upper()} 2")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    plt.close()

def plot_cluster_distribution(pred_labels, true_labels, save_path=None):
    """
    Plot bar chart of cluster distribution over true genres/labels
    """
    if true_labels is None: 
        return
        
    plt.figure(figsize=(12, 6))
    
    # Needs pandas for easy plotting usually, keeping dependecies minimal
    # Or use seaborn
    import pandas as pd
    df = pd.DataFrame({'Cluster': pred_labels, 'Genre': true_labels})
    
    sns.countplot(x='Cluster', hue='Genre', data=df, palette='tab10')
    plt.title("Cluster Distribution over Genres")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved dist plot to {save_path}")
    plt.close()

def compare_methods(vae_latent, raw_features, true_labels=None, clustering_methods=['kmeans'], n_clusters=5, save_dir=None):
    print("\n" + "="*50)
    print("COMPARISON: VAE vs BASELINE (PCA)")
    print("="*50)
    
    target_dim = vae_latent.shape[1]
    
    # 1. Baseline: PCA
    print(f"Reducing raw features ({raw_features.shape[1]}) -> PCA ({target_dim})...")
    pca = PCA(n_components=target_dim)
    pca_feats = pca.fit_transform(raw_features)
    
    results = {}
    
    for method in clustering_methods:
        print(f"\n--- Clustering Method: {method.upper()} ---")
        
        # VAE Clustering
        vae_labels = perform_clustering(vae_latent, method, n_clusters)
        vae_metrics = evaluate_clustering(vae_latent, vae_labels, true_labels)
        
        # PCA Clustering
        pca_labels = perform_clustering(pca_feats, method, n_clusters)
        pca_metrics = evaluate_clustering(pca_feats, pca_labels, true_labels)
        
        results[method] = {'vae': vae_metrics, 'pca': pca_metrics}
        
        # Header
        print(f"{'Representation':<15} | {'Sil':<6} | {'NMI':<6} | {'ARI':<6} | {'Purity':<6}")
        print("-" * 60)
        print(f"{'VAE':<15} | {vae_metrics['sil']:.3f}  | {vae_metrics['nmi']:.3f}  | {vae_metrics['ari']:.3f}  | {vae_metrics['purity']:.3f}")
        print(f"{'PCA':<15} | {pca_metrics['sil']:.3f}  | {pca_metrics['nmi']:.3f}  | {pca_metrics['ari']:.3f}  | {pca_metrics['purity']:.3f}")
        
        if save_dir:
            plot_latent_space(vae_latent, vae_labels, method='tsne', 
                              save_path=save_dir / f'viz_vae_{method}.png', title=f"VAE - {method.upper()}")
            
            if true_labels is not None:
                plot_cluster_distribution(vae_labels, true_labels, save_path=save_dir / f'viz_dist_vae_{method}.png')

    return results
