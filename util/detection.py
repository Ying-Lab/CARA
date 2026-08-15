import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd


def _compute_gap_threshold(bad_ratios):
    """
    Automatically infer ood_prop_threshold based on the maximum gap method.
    Args:
        bad_ratios: List of bad cell ratios for each cluster

    Returns:
        (threshold, max_gap, gap_above_indices, max_ratio_idx, possibly_ood_indices):
            threshold: Minimum value of clusters above max_gap (for existence judgment)
            max_gap: Maximum gap value
            gap_above_indices: List of original indices of all clusters above max_gap
            max_ratio_idx: Original index of the cluster with maximum bad_ratio
            possibly_ood_indices: Original indices of clusters above gap but not the maximum
        Returns (None, 0, [], -1, []) if cannot infer
    """
    if len(bad_ratios) < 2:
        return None, 0.0, [], -1, []

    # Sort
    indexed_ratios = sorted(enumerate(bad_ratios), key=lambda x: x[1])
    sorted_indices = [x[0] for x in indexed_ratios]
    sorted_ratios = [x[1] for x in indexed_ratios]

    gaps = [sorted_ratios[i + 1] - sorted_ratios[i] for i in range(len(sorted_ratios) - 1)]
    max_gap = max(gaps)
    max_gap_idx = int(np.argmax(gaps))

    # Original indices of all clusters above max_gap (possibly OOD clusters)
    gap_above_indices = sorted_indices[max_gap_idx + 1:]

    # Original index of cluster with maximum bad_ratio
    max_ratio_idx = int(np.argmax(bad_ratios))

    # Set threshold to maximum value (select the largest one)
    threshold = sorted_ratios[-1]

    # Clusters above gap but not the maximum (may contain a small amount of OOD)
    possibly_ood_indices = [i for i in gap_above_indices if i != max_ratio_idx]

    return threshold, max_gap, gap_above_indices, max_ratio_idx, possibly_ood_indices


def auto_select_voting_params(
    testadata,
    clusters,
    prob_obsm_key: str = 'prob',
    elbo_obsm_key: str = 'elbo_loss'
):
    """
    Automatically infer OOD voting parameters (keeping MSP+Entropy+ELBO three-vote framework).

    Core logic: Scan candidate quantile combinations, calculate cluster-level bad_ratio distribution
    for each combination, and select the parameter set that maximizes "maximum gap (max_gap)".
    The larger max_gap is, the more obvious the separation between ID and OOD clusters.

    Also performs existence judgment via max_gap: if the maximum gap is too small,
    consider that no significant OOD clusters exist in the data.

    Args:
        testadata: AnnData object
        clusters: Cluster label array
        prob_obsm_key: Key for probability matrix in obsm
        elbo_obsm_key: Key for ELBO loss in obsm

    Returns:
        (params_dict, max_gap):
            params_dict contains msp_threshold, entropy_quantile, elbo_quantile,
            ood_proportion_threshold, max_gap, bad_ratios;
            Returns None if no significant OOD clusters detected
    """
    cluster_ids = sorted(set(clusters), key=lambda x: int(x))

    # Ensure required scores are calculated (written to testadata.obs for subsequent voting reuse)
    if prob_obsm_key in testadata.obsm:
        prob_matrix = testadata.obsm[prob_obsm_key]
        if 'uncertainty_msp' not in testadata.obs:
            testadata.obs['uncertainty_msp'] = 1 - prob_matrix.max(axis=1)
        if 'uncertainty_entropy' not in testadata.obs:
            testadata.obs['uncertainty_entropy'] = entropy(prob_matrix.T)

    # Candidate parameter space
    msp_candidates = [0.6,0.65, 0.7,0.75, 0.8, 0.85,0.9]
    entropy_q_candidates = [0.6,0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    elbo_q_candidates = [0.6,0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

    best_gap = -1.0
    best_params = None
    best_bad_ratios = None

    for msp_t in msp_candidates:
        for eq in entropy_q_candidates:
            for elq in elbo_q_candidates:
                # Calculate bad_cell mask (three-vote voting logic)
                bad_cell = np.zeros(testadata.n_obs, dtype=bool)

                if msp_t is not None and 'uncertainty_msp' in testadata.obs:
                    bad_cell |= testadata.obs['uncertainty_msp'].values > msp_t

                if 'uncertainty_entropy' in testadata.obs:
                    entropy_thresh = testadata.obs['uncertainty_entropy'].quantile(eq)
                    bad_cell |= testadata.obs['uncertainty_entropy'].values > entropy_thresh

                if elbo_obsm_key in testadata.obsm:
                    elbo_vec = np.asarray(testadata.obsm[elbo_obsm_key]).ravel()
                    elbo_thresh = np.quantile(elbo_vec, elq)
                    bad_cell |= elbo_vec > elbo_thresh

                # Calculate cluster-level bad_ratio
                bad_ratios = []
                for c in cluster_ids:
                    mask = clusters == c
                    n = mask.sum()
                    bad_ratios.append(float(bad_cell[mask].sum()) / float(n) if n > 0 else 0.0)

                # Gap splitting
                threshold, max_gap, gap_above_indices, max_ratio_idx, possibly_ood_indices = _compute_gap_threshold(bad_ratios)
                if threshold is None:
                    continue

                # Scoring criteria: larger max_gap is better (more obvious polarization)
                if max_gap > best_gap:
                    best_gap = max_gap
                    best_params = {
                        'msp_threshold': msp_t,
                        'entropy_quantile': eq,
                        'elbo_quantile': elq,
                    }
                    best_bad_ratios = bad_ratios
                    best_gap_above_indices = gap_above_indices
                    best_max_ratio_idx = max_ratio_idx
                    best_possibly_ood_indices = possibly_ood_indices

    # Existence judgment: if gap is too small, consider no significant OOD clusters
    if best_gap < 0.10 or best_params is None:
        return None, best_gap

    # Recalculate ood_prop_threshold using best bad_ratios (use maximum value)
    ood_prop_threshold = max(best_bad_ratios)

    return {
        'msp_threshold': best_params['msp_threshold'],
        'entropy_quantile': best_params['entropy_quantile'],
        'elbo_quantile': best_params['elbo_quantile'],
        'ood_proportion_threshold': ood_prop_threshold,
        'max_gap': best_gap,
        'bad_ratios': best_bad_ratios,
        'gap_above_indices': best_gap_above_indices,
        'max_ratio_idx': best_max_ratio_idx,
        'possibly_ood_indices': best_possibly_ood_indices,
    }, best_gap

# old manual one
def novel_cluster_detection(
    testadata,
    emb_obsm_key: str = 'embedding',
    prob_obsm_key: str = 'prob',
    elbo_obsm_key: str = 'elbo_loss',
    n_neighbors: int = 10,
    resolution: float = 2,
    msp_threshold: float = 0.5,
    entropy_threshold: float = None,
    elbo_threshold: float = None,
    entropy_quantile: float = 0.9,
    elbo_quantile: float = 0.9,
    ood_proportion_threshold: float = 0.8,
    auto_params: bool = False
):
    sc.pp.neighbors(testadata, n_neighbors=n_neighbors, use_rep=emb_obsm_key)
    sc.tl.louvain(testadata, resolution=resolution)
    testadata.obs['louvain'] = testadata.obs['louvain'].astype('category')
    clusters = testadata.obs['louvain'].values

    # If auto parameter selection is enabled, infer optimal voting parameters
    if auto_params:
        auto_result, _ = auto_select_voting_params(
            testadata, clusters, prob_obsm_key=prob_obsm_key, elbo_obsm_key=elbo_obsm_key
        )
        if auto_result is None:
            testadata.obs['is_novel_cell'] = False
            testadata.obs['is_novel_cluster'] = pd.Categorical([False] * testadata.n_obs)
            empty_df = pd.DataFrame({
                'cluster_size': testadata.obs.groupby('louvain').size(),
                'novel_cell_count': 0,
                'novel_cell_proportion': 0.0,
                'is_novel_cluster': False
            })
            return empty_df

        # Override input values with inferred parameters
        msp_threshold = auto_result['msp_threshold']
        entropy_quantile = auto_result['entropy_quantile']
        elbo_quantile = auto_result['elbo_quantile']
        ood_proportion_threshold = auto_result['ood_proportion_threshold']

    # Calculate uncertainty scores (in auto mode, already written in auto_select_voting_params, here only supplement those not yet calculated)
    scores_available = []
    if prob_obsm_key in testadata.obsm:
        prob_matrix = testadata.obsm[prob_obsm_key]
        if 'uncertainty_msp' not in testadata.obs:
            testadata.obs['uncertainty_msp'] = 1 - prob_matrix.max(axis=1)
        if 'uncertainty_entropy' not in testadata.obs:
            testadata.obs['uncertainty_entropy'] = entropy(prob_matrix.T)
        scores_available.extend(['uncertainty_msp', 'uncertainty_entropy'])
    else:
        pass

    if elbo_obsm_key in testadata.obsm:
        scores_available.append(elbo_obsm_key)
    else:
        pass

    # Initialize novel cell marker
    testadata.obs['is_novel_cell'] = False
    vote_details = {}

    # MSP Score Voting
    if 'uncertainty_msp' in scores_available:
        if msp_threshold is not None:
            msp_novel_mask = testadata.obs['uncertainty_msp'] > msp_threshold
            testadata.obs.loc[msp_novel_mask, 'is_novel_cell'] = True
            vote_details['MSP'] = msp_novel_mask.sum()

    # Entropy Score Voting
    if 'uncertainty_entropy' in scores_available:
        if entropy_threshold is not None:
            entropy_threshold_val = entropy_threshold
            entropy_novel_mask = testadata.obs['uncertainty_entropy'] > entropy_threshold_val
        else:
            entropy_threshold_val = testadata.obs['uncertainty_entropy'].quantile(entropy_quantile)
            entropy_novel_mask = testadata.obs['uncertainty_entropy'] > entropy_threshold_val

        testadata.obs.loc[entropy_novel_mask, 'is_novel_cell'] = True
        vote_details['Entropy'] = entropy_novel_mask.sum()

    # ELBO Score Voting
    if elbo_obsm_key in scores_available:
        if elbo_threshold is not None:
            elbo_threshold_val = elbo_threshold
            elbo_novel_mask = testadata.obsm[elbo_obsm_key] > elbo_threshold_val
        else:
            elbo_vec = np.asarray(testadata.obsm[elbo_obsm_key]).ravel()
            elbo_threshold_val = np.quantile(elbo_vec, elbo_quantile)
            elbo_novel_mask = elbo_vec > elbo_threshold_val
        testadata.obs.loc[elbo_novel_mask, 'is_novel_cell'] = True
        vote_details['ELBO'] = elbo_novel_mask.sum()

    cluster_analysis = testadata.obs.groupby('louvain').agg({
        'is_novel_cell': ['size', 'sum']
    })
    cluster_analysis.columns = ['cluster_size', 'novel_cell_count']
    cluster_analysis['novel_cell_proportion'] = cluster_analysis['novel_cell_count'] / cluster_analysis['cluster_size']

    # If using automatic parameter selection, use gap-based logic to label clusters
    if auto_params and auto_result is not None:
        # Get cluster_ids order (corresponding to bad_ratios)
        cluster_ids_sorted = sorted(set(clusters), key=lambda x: int(x))

        # Initialize label column
        cluster_analysis['cluster_status'] = 'Normal'

        # Label clusters in possibly_ood_indices as "may contain a small amount of OOD"
        for idx in auto_result['possibly_ood_indices']:
            if idx < len(cluster_ids_sorted):
                cluster_id = cluster_ids_sorted[idx]
                if cluster_id in cluster_analysis.index:
                    cluster_analysis.loc[cluster_id, 'cluster_status'] = 'Possibly OOD'

        # Only label the largest cluster as "Novel"
        max_idx = auto_result['max_ratio_idx']
        if max_idx < len(cluster_ids_sorted):
            max_cluster_id = cluster_ids_sorted[max_idx]
            if max_cluster_id in cluster_analysis.index:
                cluster_analysis.loc[max_cluster_id, 'cluster_status'] = 'Novel'

        # Set is_novel_cluster column
        cluster_analysis['is_novel_cluster'] = cluster_analysis['cluster_status'] == 'Novel'

        novel_clusters = cluster_analysis[cluster_analysis['is_novel_cluster']].index
        possibly_ood_clusters = cluster_analysis[cluster_analysis['cluster_status'] == 'Possibly OOD'].index

        testadata.obs['is_novel_cluster'] = testadata.obs['louvain'].isin(novel_clusters)
        testadata.obs['is_novel_cluster'] = testadata.obs['is_novel_cluster'].astype('category')

        # Add possibly_ood label
        testadata.obs['possibly_ood'] = testadata.obs['louvain'].isin(possibly_ood_clusters)

    else:
        # Non-automatic mode, use original logic
        cluster_analysis['is_novel_cluster'] = cluster_analysis['novel_cell_proportion'] >= ood_proportion_threshold
        cluster_analysis['cluster_status'] = cluster_analysis['is_novel_cluster'].map({True: 'Novel', False: 'Normal'})

        novel_clusters = cluster_analysis[cluster_analysis['is_novel_cluster']].index
        testadata.obs['is_novel_cluster'] = testadata.obs['louvain'].isin(novel_clusters)
        testadata.obs['is_novel_cluster'] = testadata.obs['is_novel_cluster'].astype('category')
        testadata.obs['possibly_ood'] = False

    if 'X_umap' not in testadata.obsm_keys():
        sc.tl.umap(testadata)

    _, axes = plt.subplots(3, 1, figsize=(8, 16))
    axes = axes.flatten()

    sc.pl.umap(testadata, color='louvain', title='Louvain Clusters',
            legend_loc='on data', ax=axes[0], show=False)

    col = 'is_bad_cell' if 'is_bad_cell' in testadata.obs.columns else 'is_novel_cell'
    testadata.obs['bad_cell_proportion'] = testadata.obs.groupby('louvain')[col].transform('mean')
    sc.pl.umap(testadata, color='bad_cell_proportion', title='Proportion of Bad Cells per Cluster', cmap='viridis', ax=axes[1], show=False)

    # Visualize
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
    testadata.obs.loc[novel_cells_mask, cell_type_key] = 'unknown'

    print(f"Updated cell type to 'unknown' for {num_novel_cells} cells.")

    return testadata
