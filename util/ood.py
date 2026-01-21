import numpy as np
from scipy.stats import entropy
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd
def novel_cluster_detection(
    testadata,
    emb_obsm_key: str = 'embedding',
    prob_obsm_key: str = 'prob',
    elbo_obs_key: str = 'elbo_loss',
    n_neighbors: int = 10,
    resolution: float = 2,
    msp_threshold: float = 0.5,
    entropy_threshold: float = None,
    elbo_threshold: float = None,
    entropy_quantile: float = 0.9,
    elbo_quantile: float = 0.9,
    ood_proportion_threshold: float = 0.3
):
    """
    Perform Louvain clustering, calculate multiple uncertainty scores, identify bad cells using a voting mechanism, and detect OOD clusters.

    Args:
        testadata: AnnData object containing the single-cell data.
        emb_obsm_key (str): Key in `obsm` for the embedding matrix used for clustering (default: 'embedding').
        prob_obsm_key (str): Key in `obsm` for the probability matrix used to calculate MSP and Entropy (default: 'prob').
        elbo_obs_key (str): Key in `obs` for the ELBO loss values (default: 'elbo_loss').
        n_neighbors (int): Number of neighbors to use for constructing the neighborhood graph (default: 10).
        resolution (float): Resolution parameter for Louvain clustering; higher values lead to more clusters (default: 2).
        msp_threshold (float): Threshold for MSP score; cells with uncertainty > this value are voted as bad (default: 0.5).
        entropy_threshold (float): Absolute threshold for Entropy score. If None, quantile is used (default: None).
        elbo_threshold (float): Absolute threshold for ELBO loss. If None, quantile is used (default: None).
        entropy_quantile (float): Quantile to determine Entropy threshold if absolute threshold is not provided (default: 0.9).
        elbo_quantile (float): Quantile to determine ELBO threshold if absolute threshold is not provided (default: 0.9).
        ood_proportion_threshold (float): Threshold for the proportion of bad cells in a cluster to label it as OOD (default: 0.3).
    """
    # --- Step 1: Clustering ---
    sc.pp.neighbors(testadata, n_neighbors=n_neighbors, use_rep=emb_obsm_key)
    sc.tl.louvain(testadata, resolution=resolution)

    # Ensure the louvain column is of category type
    testadata.obs['louvain'] = testadata.obs['louvain'].astype('category')

    # --- Step 2: Calculate Uncertainty Scores ---
    scores_available = []

    if prob_obsm_key in testadata.obsm:
        prob_matrix = testadata.obsm[prob_obsm_key]
        testadata.obs['uncertainty_msp'] = 1 - prob_matrix.max(axis=1)

        # Fix entropy calculation
        try:
            # Method 1: Direct entropy calculation
            epsilon = 1e-12
            log_probs = np.log2(prob_matrix + epsilon)
            entropy_values = -np.sum(prob_matrix * log_probs, axis=1)
            testadata.obs['uncertainty_entropy'] = entropy_values
        except:
            # Method 2: Use scipy's entropy function
            testadata.obs['uncertainty_entropy'] = entropy(prob_matrix.T)

        scores_available.extend(['uncertainty_msp', 'uncertainty_entropy'])
    else:
        print(f"Warning: '{prob_obsm_key}' not found, unable to calculate probability-based uncertainty scores.")

    if elbo_obs_key in testadata.obs:
        scores_available.append(elbo_obs_key)
    else:
        print(f"Warning: '{elbo_obs_key}' not found, unable to use ELBO loss.")

    # --- Step 3: Three-Score Voting Mechanism ---
    print("\n--- Step 3: Execute Three-Score Voting Mechanism ---")

    # Initialize bad cell marker
    testadata.obs['is_bad_cell'] = False
    vote_details = {}

    # MSP Score Voting
    if 'uncertainty_msp' in scores_available:
        if msp_threshold is not None:
            msp_bad_mask = testadata.obs['uncertainty_msp'] > msp_threshold
            testadata.obs.loc[msp_bad_mask, 'is_bad_cell'] = True
            vote_details['MSP'] = msp_bad_mask.sum()
            print(f"MSP Voting: {msp_bad_mask.sum()} cells marked as bad cells (Threshold: {msp_threshold})")

    # Entropy Score Voting
    if 'uncertainty_entropy' in scores_available:
        if entropy_threshold is not None:
            entropy_threshold_val = entropy_threshold
            entropy_bad_mask = testadata.obs['uncertainty_entropy'] > entropy_threshold_val
        else:
            # Use quantile threshold
            entropy_threshold_val = testadata.obs['uncertainty_entropy'].quantile(entropy_quantile)
            entropy_bad_mask = testadata.obs['uncertainty_entropy'] > entropy_threshold_val
            print(f"Entropy Threshold (based on {entropy_quantile} quantile): {entropy_threshold_val:.4f}")

        testadata.obs.loc[entropy_bad_mask, 'is_bad_cell'] = True
        vote_details['Entropy'] = entropy_bad_mask.sum()
        print(f"Entropy Voting: {entropy_bad_mask.sum()} cells marked as bad cells")

    # ELBO Score Voting
    if elbo_obs_key in scores_available:
        if elbo_threshold is not None:
            elbo_threshold_val = elbo_threshold
            elbo_bad_mask = testadata.obs[elbo_obs_key] > elbo_threshold_val
        else:
            # Use quantile threshold
            elbo_threshold_val = testadata.obs[elbo_obs_key].quantile(elbo_quantile)
            elbo_bad_mask = testadata.obs[elbo_obs_key] > elbo_threshold_val
            print(f"ELBO Threshold (based on {elbo_quantile} quantile): {elbo_threshold_val:.4f}")

        testadata.obs.loc[elbo_bad_mask, 'is_bad_cell'] = True
        vote_details['ELBO'] = elbo_bad_mask.sum()
        print(f"ELBO Voting: {elbo_bad_mask.sum()} cells marked as bad cells")
    
    total_bad_cells = testadata.obs['is_bad_cell'].sum()
    print(f"Total bad cells: {total_bad_cells}/{testadata.n_obs} ({total_bad_cells/testadata.n_obs*100:.2f}%)")

    # --- Step 4: Identify OOD clusters based on bad cell proportion ---
    print("\n--- Step 4: Identify OOD Clusters ---")
    
    cluster_analysis = testadata.obs.groupby('louvain').agg({
        'is_bad_cell': ['size', 'sum']
    })
    cluster_analysis.columns = ['cluster_size', 'bad_cell_count']
    cluster_analysis['bad_cell_proportion'] = cluster_analysis['bad_cell_count'] / cluster_analysis['cluster_size']
    
    cluster_analysis['is_ood_cluster'] = cluster_analysis['bad_cell_proportion'] > ood_proportion_threshold
    
    ood_clusters = cluster_analysis[cluster_analysis['is_ood_cluster']].index
    testadata.obs['is_ood_cluster'] = testadata.obs['louvain'].isin(ood_clusters)
    testadata.obs['is_ood_cluster'] = testadata.obs['is_ood_cluster'].astype('category')
    
    print(f"Identified {len(ood_clusters)} OOD clusters: {list(ood_clusters)}")

    # Display detailed cluster analysis results
    print("\nCluster Analysis Results:")
    for cluster_id, row in cluster_analysis.iterrows():
        status = "OOD" if row['is_ood_cluster'] else "Normal"
        print(f"Cluster {cluster_id}: {row['cluster_size']} cells, "
            f"{row['bad_cell_count']} bad cells, "
            f"proportion: {row['bad_cell_proportion']:.3f} ({status})")

    # --- Step 5: Visualization ---
    print("\n--- Step 5: Generate Visualization ---")

    if 'X_umap' not in testadata.obsm_keys():
        sc.tl.umap(testadata)

    fig, axes = plt.subplots(1, 3, figsize=(12, 16))
    axes = axes.flatten()

    sc.pl.umap(testadata, color='louvain', title='Louvain Clusters',
            legend_loc='on data', ax=axes[0], show=False)

    testadata.obs['is_bad_cell_cat'] = testadata.obs['is_bad_cell'].map({True: 'Bad Cell', False: 'Good Cell'}).astype('category')
    sc.pl.umap(testadata, color='is_bad_cell_cat', title='Bad Cells (Voting Result)',
            palette={'Bad Cell': 'red', 'Good Cell': 'lightgray'}, ax=axes[1], show=False)

    testadata.obs['ood_status'] = testadata.obs['is_ood_cluster'].map({True: 'OOD Cluster', False: 'Normal Cluster'}).astype('category')
    sc.pl.umap(testadata, color='ood_status', title='OOD Clusters',
            palette={'OOD Cluster': 'darkred', 'Normal Cluster': 'lightblue'}, ax=axes[2], show=False)

    plt.tight_layout()
    plt.show()

    testadata.obs.drop(['is_bad_cell_cat', 'ood_status'], axis=1, inplace=True, errors='ignore')

    return cluster_analysis




def label_novel_cells_as_unknown(
    testadata,
    cell_type_key: str,
    ood_cluster_key: str = 'is_ood_cluster'
):
    """
    将已识别为 OOD 簇中所有细胞的细胞类型标记为 'Unknown'。

    参数:
    - testadata: AnnData 对象，其中包含 'is_ood_cluster' 列。
    - cell_type_key: 存储细胞类型注释的 obs 列名。
    - ood_cluster_key: 标识一个细胞是否属于 OOD 簇的布尔值 obs 列名。
    """
    print(f"\n--- 步骤 4: 将 OOD 簇中的细胞标记为 'unknown' (在 '{cell_type_key}' 列) ---")

    if ood_cluster_key not in testadata.obs:
        raise KeyError(f"在 obs 中未找到 OOD 簇标记列 '{ood_cluster_key}'。请先运行 identify_ood_clusters。")
    if cell_type_key not in testadata.obs:
        raise KeyError(f"在 obs 中未找到细胞类型列 '{cell_type_key}'。")

    # 确保细胞类型列是 category 类型，并且 'Unknown' 是一个合法的类别
    if pd.api.types.is_categorical_dtype(testadata.obs[cell_type_key]):
        if 'unknown' not in testadata.obs[cell_type_key].cat.categories:
            testadata.obs[cell_type_key] = testadata.obs[cell_type_key].cat.add_categories(['unknown'])
    
    # 找到 OOD 簇中的细胞
    ood_cells_mask = testadata.obs[ood_cluster_key] == True
    num_ood_cells = ood_cells_mask.sum()

    if num_ood_cells == 0:
        print("没有找到属于 OOD 簇的细胞，无需更新。")
        return testadata

    print(f"找到了 {num_ood_cells} 个细胞属于 OOD 簇。")

    # 使用 .loc 来确保安全赋值
    testadata.obs.loc[ood_cells_mask, cell_type_key] = 'unknown'
    
    print(f"已将 {num_ood_cells} 个细胞的类型更新为 'unknown'。")
    
    return testadata


def label_ood_cells_as_unknown(
    testadata,
    cell_type_key: str,
    ood_cluster_key: str = 'is_ood_cluster'
):
    """
    Label the cell type of all cells identified within OOD clusters as 'Unknown'.

    Args:
    - testadata: AnnData object containing the 'is_ood_cluster' column.
    - cell_type_key: Column name in obs storing cell type annotations.
    - ood_cluster_key: Column name in obs containing boolean values indicating if a cell belongs to an OOD cluster.
    """
    print(f"\n--- Step 4: Labeling cells in OOD clusters as 'unknown' (in '{cell_type_key}' column) ---")

    if ood_cluster_key not in testadata.obs:
        raise KeyError(f"OOD cluster marker column '{ood_cluster_key}' not found in obs. Please run identify_ood_clusters first.")
    if cell_type_key not in testadata.obs:
        raise KeyError(f"Cell type column '{cell_type_key}' not found in obs.")

    # Ensure the cell type column is categorical and 'unknown' is a valid category
    if pd.api.types.is_categorical_dtype(testadata.obs[cell_type_key]):
        if 'unknown' not in testadata.obs[cell_type_key].cat.categories:
            testadata.obs[cell_type_key] = testadata.obs[cell_type_key].cat.add_categories(['unknown'])

    # Find cells in OOD clusters
    ood_cells_mask = testadata.obs[ood_cluster_key] == True
    num_ood_cells = ood_cells_mask.sum()

    if num_ood_cells == 0:
        print("No cells found belonging to OOD clusters, no update needed.")
        return testadata

    # Use .loc for safe assignment
    testadata.obs.loc[ood_cells_mask, cell_type_key] = 'unknown'

    print(f"Updated cell type to 'unknown' for {num_ood_cells} cells.")

    return testadata