import numpy as np
from scipy.stats import entropy
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
    ood_proportion_threshold: float = 0.8
):
    # --- Step 1: Clustering ---
    sc.pp.neighbors(testadata, n_neighbors=n_neighbors, use_rep=emb_obsm_key)
    sc.tl.louvain(testadata, resolution=resolution)
    testadata.obs['louvain'] = testadata.obs['louvain'].astype('category')
    # --- Step 2: Calculate Uncertainty Scores ---
    scores_available = []
    if prob_obsm_key in testadata.obsm:
        prob_matrix = testadata.obsm[prob_obsm_key]
        testadata.obs['uncertainty_msp'] = 1 - prob_matrix.max(axis=1)
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
    # Initialize novel cell marker
    testadata.obs['is_novel_cell'] = False
    vote_details = {}

    # MSP Score Voting
    if 'uncertainty_msp' in scores_available:
        if msp_threshold is not None:
            msp_novel_mask = testadata.obs['uncertainty_msp'] > msp_threshold
            testadata.obs.loc[msp_novel_mask, 'is_novel_cell'] = True
            vote_details['MSP'] = msp_novel_mask.sum()
            print(f"MSP Voting: {msp_novel_mask.sum()} cells marked as novel cells (Threshold: {msp_threshold})")

    # Entropy Score Voting
    if 'uncertainty_entropy' in scores_available:
        if entropy_threshold is not None:
            entropy_threshold_val = entropy_threshold
            entropy_novel_mask = testadata.obs['uncertainty_entropy'] > entropy_threshold_val
        else:
            # Use quantile threshold
            entropy_threshold_val = testadata.obs['uncertainty_entropy'].quantile(entropy_quantile)
            entropy_novel_mask = testadata.obs['uncertainty_entropy'] > entropy_threshold_val
            print(f"Entropy Threshold (based on {entropy_quantile} quantile): {entropy_threshold_val:.4f}")

        testadata.obs.loc[entropy_novel_mask, 'is_novel_cell'] = True
        vote_details['Entropy'] = entropy_novel_mask.sum()
        print(f"Entropy Voting: {entropy_novel_mask.sum()} cells marked as novel cells")

    # ELBO Score Voting
    if elbo_obs_key in scores_available:
        if elbo_threshold is not None:
            elbo_threshold_val = elbo_threshold
            elbo_novel_mask = testadata.obs[elbo_obs_key] > elbo_threshold_val
        else:
            # Use quantile threshold
            elbo_threshold_val = testadata.obs[elbo_obs_key].quantile(elbo_quantile)
            elbo_novel_mask = testadata.obs[elbo_obs_key] > elbo_threshold_val
            print(f"ELBO Threshold (based on {elbo_quantile} quantile): {elbo_threshold_val:.4f}")

        testadata.obs.loc[elbo_novel_mask, 'is_novel_cell'] = True
        vote_details['ELBO'] = elbo_novel_mask.sum()
        print(f"ELBO Voting: {elbo_novel_mask.sum()} cells marked as novel cells")
    
    total_novel_cells = testadata.obs['is_novel_cell'].sum()
    print(f"Total novel cells: {total_novel_cells}/{testadata.n_obs} ({total_novel_cells/testadata.n_obs*100:.2f}%)")

    # --- Step 4: Identify novel clusters based on novel cell proportion ---
    print("\n--- Step 4: Identify novel Clusters ---")
    
    cluster_analysis = testadata.obs.groupby('louvain').agg({
        'is_novel_cell': ['size', 'sum']
    })
    cluster_analysis.columns = ['cluster_size', 'novel_cell_count']
    cluster_analysis['novel_cell_proportion'] = cluster_analysis['novel_cell_count'] / cluster_analysis['cluster_size']
    
    cluster_analysis['is_novel_cluster'] = cluster_analysis['novel_cell_proportion'] > ood_proportion_threshold
    
    novel_clusters = cluster_analysis[cluster_analysis['is_novel_cluster']].index
    testadata.obs['is_novel_cluster'] = testadata.obs['louvain'].isin(novel_clusters)
    testadata.obs['is_novel_cluster'] = testadata.obs['is_novel_cluster'].astype('category')
    
    print(f"Identified {len(novel_clusters)} novel clusters: {list(novel_clusters)}")

    # Display detailed cluster analysis results
    print("\nCluster Analysis Results:")
    for cluster_id, row in cluster_analysis.iterrows():
        status = "Novel" if row['is_novel_cluster'] else "Normal"
        print(f"Cluster {cluster_id}: {row['cluster_size']} cells, "
            f"{row['novel_cell_count']} novel cells, "
            f"proportion: {row['novel_cell_proportion']:.3f} ({status})")

    # --- Step 5: Visualization ---
    print("\n--- Step 5: Generate Visualization ---")

    if 'X_umap' not in testadata.obsm_keys():
        sc.tl.umap(testadata)

    fig, axes = plt.subplots(3, 1, figsize=(8, 16)) 
    axes = axes.flatten()

    sc.pl.umap(testadata, color='louvain', title='Louvain Clusters',
            legend_loc='on data', ax=axes[0], show=False)

    testadata.obs['is_novel_cell_cat'] = testadata.obs['is_novel_cell'].map({True: 'novel Cell', False: 'Good Cell'}).astype('category')
    sc.pl.umap(testadata, color='is_novel_cell_cat', title='novel Cells (Voting Result)',
            palette={'novel Cell': 'red', 'Good Cell': 'lightgray'}, ax=axes[1], show=False)

    testadata.obs['novel_status'] = testadata.obs['is_novel_cluster'].map({True: 'Novel Cluster', False: 'Normal Cluster'}).astype('category')
    sc.pl.umap(testadata, color='novel_status', title='Novel Clusters',
            palette={'Novel Cluster': 'darkred', 'Normal Cluster': 'lightblue'}, ax=axes[2], show=False)

    plt.tight_layout()
    plt.show()

    testadata.obs.drop(['is_novel_cell_cat', 'novel_status'], axis=1, inplace=True, errors='ignore')

    return cluster_analysis

def label_novel_cells_as_unknown(
    testadata,
    cell_type_key: str,
    novel_cluster_key: str = 'is_novel_cluster'
):
    """
    Label the cell type of all cells identified within novel clusters as 'Unknown'.

    Args:
    - testadata: AnnData object containing the 'is_novel_cluster' column.       
    - cell_type_key: Column name in obs storing cell type annotations.
    - novel_cluster_key: Column name in obs containing boolean values indicating if a cell belongs to a novel cluster.
    """
    print(f"\n--- Step 4: Labeling cells in novel clusters as 'unknown' (in '{cell_type_key}' column) ---")     

    if novel_cluster_key not in testadata.obs:
        raise KeyError(f"Novel cluster marker column '{novel_cluster_key}' not found in obs. Please run identify_novel_clusters first.")
    if cell_type_key not in testadata.obs:
        raise KeyError(f"Cell type column '{cell_type_key}' not found in obs.")

    # Ensure the cell type column is categorical and 'unknown' is a valid category
    if pd.api.types.is_categorical_dtype(testadata.obs[cell_type_key]):
        if 'unknown' not in testadata.obs[cell_type_key].cat.categories:
            testadata.obs[cell_type_key] = testadata.obs[cell_type_key].cat.add_categories(['unknown'])

    # Find cells in novel clusters
    novel_cells_mask = testadata.obs[novel_cluster_key] == True
    num_novel_cells = novel_cells_mask.sum()

    if num_novel_cells == 0:
        print("No cells found belonging to novel clusters, no update needed.")
        return testadata

    # Use .loc for safe assignment
    testadata.obs.loc[novel_cells_mask, cell_type_key] = 'unknown'

    print(f"Updated cell type to 'unknown' for {num_novel_cells} cells.")   

    return testadata