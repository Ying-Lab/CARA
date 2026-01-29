import os
import random
import numpy as np
import torch
import matplotlib
import scanpy as sc
import pandas as pd
import copy
from util.dataloader import *
from torch.utils.data import DataLoader
import anndata as ad
from sklearn.model_selection import train_test_split
def setup(seed = 233):
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.set_default_tensor_type(torch.FloatTensor)
    matplotlib.rcParams['figure.figsize'] = [8, 6]
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['svg.fonttype'] = 'none'
    
def map_indices_to_names(indices, index_to_category_map):
    return [index_to_category_map[index] for index in indices]
def map_names_to_indices(indices, category_to_index_map):
    return [category_to_index_map[index] for index in indices]

def map_batch_to_indices(indices, category_to_index_map):
    return [category_to_index_map[index] for index in indices]





from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,matthews_corrcoef
def get_accuracy(data_loader, classifier_fn):
        predictions, actuals = [], []
        # use the appropriate data loader
        for (xs, ys, omic,_,_) in data_loader:
            predictions.append(classifier_fn(xs,omic))  
            _, yshat=torch.topk(ys, 1)
            actuals.append(yshat)
        # compute the number of accurate predictions
        predictions = torch.cat(predictions, dim=0)
        actuals = torch.cat(actuals, dim=0)
        # _, y = torch.topk(actuals, 1)
        y=actuals
        _, yhat = torch.topk(predictions, 1)
        y = y.detach().cpu().numpy()
        yhat = yhat.detach().cpu().numpy()
        accuracy = accuracy_score(y, yhat)
        f1_macro = f1_score(y, yhat, average='macro')
        f1_weighted = f1_score(y, yhat, average='weighted')
        precision = precision_score(y, yhat, average='macro')
        recall = recall_score(y, yhat, average='macro')
        mcc = matthews_corrcoef(y, yhat)
        ARI = adjusted_rand_score(y.ravel(), yhat.ravel())
        NMI = normalized_mutual_info_score(y.ravel(), yhat.ravel())
        return accuracy, f1_macro, f1_weighted, precision, recall, mcc,ARI,NMI  


def mark_cell_types_unknown(
    adata,
    column: str = 'cell_type',
    to_unknown=None,                
    remove_unused: bool = True,     
    drop_existing_unknown: bool = False, 
    in_place: bool = True         
):
    """
    Mark specified cell types as 'unknown' and keep categories clean.
    """
    if to_unknown is None:
        to_unknown = []
    if column not in adata.obs.columns:
        raise KeyError(f"'{column}' is not in adata.obs")
    # If not in-place modification, copy first
    if not in_place:
        adata = adata.copy()
    # Optional: Filter out existing 'unknown' cells first
    if drop_existing_unknown:
        adata = adata[adata.obs[column] != 'unknown'].copy()
    # Ensure it is a categorical type
    series = adata.obs[column]
    if not pd.api.types.is_categorical_dtype(series):
        adata.obs[column] = pd.Categorical(series)
        series = adata.obs[column]
    # Add 'unknown' category (if it doesn't exist)
    if 'unknown' not in series.cat.categories:
        adata.obs[column] = series.cat.add_categories(['unknown'])
        series = adata.obs[column]
    # Optional: Remove unused categories (e.g. categories that were replaced)
    if remove_unused:
        adata.obs[column] = adata.obs[column].cat.remove_unused_categories()

    print("\nOperation completed!")
    print("Current categories:", list(adata.obs[column].cat.categories))
    return adata

def evaluate(data_loader, cara, loss_basic):
    predictions, scores, actuals, zs, zys, batchs, xs, barcodes,elbos = [], [], [], [], [], [], [], [],[]

    with torch.no_grad():
        for (xs, ys, omic, barcode, batch) in data_loader:
            yhats, yscores = cara.classifier_with_probability(xs, mode=omic)
            scores.append(yscores)

            _, yhats = torch.topk(yhats, 1)
            predictions.append(yhats.cpu().detach().numpy())

            _, yshat = torch.topk(ys, 1)
            actuals.append(yshat.cpu().detach().numpy())

            z = cara.latent_embedding(xs, mode=omic)
            zs.append(z.cpu().detach().numpy())

            zy = cara.latent_embedding_zy(xs, mode=omic)
            zys.append(zy.cpu().detach().numpy())
            barcodes.append(barcode)
            

            batchs.append(batch.cpu().detach().numpy())
            exps.append(xs.cpu().detach().numpy())
            with torch.no_grad():  
                elbo=loss_basic.evaluate_loss(xs, mode=omic)
                elbos.append(elbo.cpu().detach().numpy())

    batchs = np.concatenate(batchs, axis=0)
    barcodes = np.concatenate(barcodes, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    scores = torch.cat(scores, dim=0).cpu().detach().numpy()
    actuals = np.concatenate(actuals, axis=0)
    zs = np.concatenate(zs, axis=0)
    zys = np.concatenate(zys, axis=0)
    exps = np.concatenate(exps, axis=0)
    elbos = np.concatenate(elbos, axis=0)

    test_accuracy, test_f1_macro, test_f1_weighted, test_precision, test_recall, test_mcc, ARI, NMI = get_accuracy(
        data_loader, cara.classifier
    )
   
    str_print1 = " test accuracy {:.4f}".format(test_accuracy)
    str_print1 += " F1 {:.4f}(macro) {:.4f}(weighted)".format(test_f1_macro, test_f1_weighted)
    str_print1 += " precision {:.4f} recall {:.4f}".format(test_precision, test_recall)
    str_print1 += " mcc {:.4f}".format(test_mcc)
    str_print1 += " ARI {:.4f} NMI {:.4f}".format(ARI, NMI)
    print(str_print1)

    metrics = {
        "accuracy": test_accuracy,
        "f1_macro": test_f1_macro,
        "f1_weighted": test_f1_weighted,
        "precision": test_precision,
        "recall": test_recall,
        "mcc": test_mcc,
        "ARI": ARI,
        "NMI": NMI,
    }

    return predictions, scores, actuals, zs, zys, batchs, xs, barcodes, elbo_loss, metrics





from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    jaccard_score,
    fowlkes_mallows_score,
    v_measure_score,
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
)

def purity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm)

def compute_metrics(groundtruth, originalcell):
    y_true = np.ravel(groundtruth)
    y_pred = np.ravel(originalcell)

    ARI = adjusted_rand_score(y_true, y_pred)
    NMI = normalized_mutual_info_score(y_true, y_pred)
    jaccard = jaccard_score(y_true, y_pred, average='macro')
    fm = fowlkes_mallows_score(y_true, y_pred)
    v_measure = v_measure_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    purity = purity_score(y_true, y_pred)

    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)

    print(
        "accuracy:{:.4f}, precision_macro:{:.4f}, precision_weighted:{:.4f}, "
        "recall_macro:{:.4f}, recall_weighted:{:.4f}, f1_macro:{:.4f}, f1_weighted:{:.4f}, "
        "ARI:{:.4f}, NMI:{:.4f}, jaccard:{:.4f}, fm:{:.4f}, v_measure:{:.4f}, purity:{:.4f}".format(
            accuracy, precision_macro, precision_weighted,
            recall_macro, recall_weighted, f1_macro, f1_weighted,
            ARI, NMI, jaccard, fm, v_measure, purity
        )
    )

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "precision_weighted": precision_weighted,
        "recall_macro": recall_macro,
        "recall_weighted": recall_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "ARI": ARI,
        "NMI": NMI,
        "jaccard": jaccard,
        "fm": fm,
        "v_measure": v_measure,
        "purity": purity,
    }






def align_rna_atac_by_union_hvg(rna, atac, n_top_genes=3000):
    """
    Aligns RNA and ATAC datasets by the union of their highly variable genes (HVGs).
    Missing genes are filled with zeros.
    
    Args:
        rna (AnnData): The raw RNA AnnData object.
        atac (AnnData): The raw ATAC AnnData object (gene activity matrix).
        n_top_genes (int): Number of HVGs to select from each modality.
        
    Returns:
        tuple: (rna_final, atac_final, final_feature_list)
    """
    
    # Assume original data objects are named rna and atac
    # For clarity, we will name the final processed objects rna_final and atac_final
    
    # --- Step 1: Find Highly Variable Genes (HVGs) on copies ---
    
    # Using .copy() is good practice to avoid modifying the original data in place
    print("Finding HVGs for RNA data...")
    rna_hvg_search = rna.copy()
    sc.pp.highly_variable_genes(rna_hvg_search, flavor='seurat_v3', n_top_genes=n_top_genes, inplace=True)
    
    print("Finding HVGs for ATAC data...")
    atac_search = atac.copy()
    sc.pp.highly_variable_genes(atac_search, flavor='seurat_v3', n_top_genes=n_top_genes, inplace=True)
    
    # --- Step 2: Create a sorted 'template' list containing all HVGs ---
    
    hvg_rna = rna_hvg_search.var_names[rna_hvg_search.var['highly_variable']]
    hvg_atac = atac_search.var_names[atac_search.var['highly_variable']]
    
    # Calculate union and sort for deterministic ordering
    hvg_union_set = set(hvg_rna) | set(hvg_atac)
    final_feature_list = sorted(list(hvg_union_set))
    
    print(f"\nNumber of RNA HVGs: {len(hvg_rna)}")
    print(f"Number of ATAC HVGs: {len(hvg_atac)}")
    print(f"Total number of unique HVGs in union: {len(final_feature_list)}")
    
    # --- Step 3 (Key Modification): Safely align both datasets using pandas.reindex ---
    
    # Note: We use the original rna and atac objects here because they contain all genes,
    # not just the HVG-subsetted copies.
    
    print("\nConverting RNA data to DataFrame for reshaping...")
    # Convert sparse matrix to dense array if necessary
    rna_df = pd.DataFrame(
        rna.X.toarray() if hasattr(rna.X, 'toarray') else rna.X,
        index=rna.obs_names,
        columns=rna.var_names
    )
    
    print("Converting ATAC data to DataFrame...")
    atac_df = pd.DataFrame(
        atac.X.toarray() if hasattr(atac.X, 'toarray') else atac.X,
        index=atac.obs_names,
        columns=atac.var_names
    )
    
    # **Critical Step**: Use .reindex() to align columns.
    # This automatically handles missing genes (filling with 0) and reorders existing ones.
    
    print("\nAligning RNA data using reindex (filling missing genes with 0)...")
    rna_df_aligned = rna_df.reindex(columns=final_feature_list, fill_value=0)
    
    print("Aligning ATAC data using reindex (filling missing genes with 0)...")
    atac_df_aligned = atac_df.reindex(columns=final_feature_list, fill_value=0)
    
    # --- Step 4: Convert aligned DataFrames back to AnnData objects ---
    
    print("\nConverting aligned DataFrames back to AnnData objects...")
    rna_final = sc.AnnData(rna_df_aligned, obs=rna.obs)
    atac_final = sc.AnnData(atac_df_aligned, obs=atac.obs)
    
    # --- Step 5: Final Verification ---
    
    print("\n--- Verification ---")
    print(f"Final RNA data shape: {rna_final.shape}")
    print(f"Final ATAC data shape: {atac_final.shape}")
    
    # Check if .var_names are identical (including order)
    are_vars_identical = rna_final.var_names.equals(atac_final.var_names)
    
    if are_vars_identical:
        print("\nSuccess! 'rna_final' and 'atac_final' now have identical and ordered features.")
    else:
        print("\nWarning! Mismatch in variable ordering.")
        
    return rna_final, atac_final, final_feature_list



def  freeze_classifier(cara):
    for param in cara.decoder_z2zy.parameters():
        param.requires_grad = False
    for param in cara.encoder_z2y.parameters():
        param.requires_grad = False
        
def  unfreeze_classifier(cara):
    for param in cara.decoder_z2zy.parameters():
        param.requires_grad = True
    for param in cara.encoder_z2y.parameters():
        param.requires_grad = True


def process(seq, min_counts = 100, target_sum = 1e4):
    sc.pp.filter_genes(seq, min_counts=100)
    sc.pp.normalize_total(seq, target_sum=1e4)
    sc.pp.log1p(seq)
    return seq





def run_inference_for_epoch(losses, atac_sup_data_loader, unsup_atac_data_loader=None, use_cuda=True):
    num_losses = len(losses)
    epoch_losses_sup = [0.0] * num_losses
    epoch_losses_unsup = [0.0] * num_losses

    sup_iter = iter(atac_sup_data_loader)
    unsup_iter = iter(unsup_atac_data_loader) if unsup_atac_data_loader is not None else None
    
    # Simply iterate through the larger loader
    max_batches = max(len(atac_sup_data_loader), len(unsup_atac_data_loader) if unsup_atac_data_loader else 0)
    
    for _ in range(max_batches):
        # Supervised Step
        try:
            xs, ys, _, _, mode, _, batch = next(sup_iter)
        except StopIteration:
            sup_iter = iter(atac_sup_data_loader)
            xs, ys, _, _, mode, _, batch = next(sup_iter)
            
        if use_cuda:
            xs = xs.cuda(); batch = batch.cuda()
            if len(ys) > 0 and isinstance(ys[0], torch.Tensor): ys = ys.cuda()
        
        for loss_id in range(num_losses):
            epoch_losses_sup[loss_id] += losses[loss_id].step(xs, ys, mode, batch)
            
        # Unsupervised Step
        if unsup_iter:
            try:
                xs, _, _, _, mode, _, batch = next(unsup_iter)
            except StopIteration:
                unsup_iter = iter(unsup_atac_data_loader)
                xs, _, _, _, mode, _, batch = next(unsup_iter)
                
            if use_cuda:
                xs = xs.cuda(); batch = batch.cuda()
            
            for loss_id in range(num_losses):
                epoch_losses_unsup[loss_id] += losses[loss_id].step(xs, None, mode, batch)

    return epoch_losses_sup, epoch_losses_unsup

def run_inference_for_epoch_warmup(rna_sup_data_loader, losses, use_cuda=True):
    num_losses = len(losses)
    epoch_losses_sup = [0.0] * num_losses
    rna_labeled_train_iter = iter(rna_sup_data_loader)
    sup_batches = len(rna_sup_data_loader)
    
    for i in range(sup_batches):
        try:
            xs, ys, _, temp_x, mode, _, batch = next(rna_labeled_train_iter)
            if use_cuda:
                xs = xs.cuda()
                batch = batch.cuda()
                if len(ys) > 0 and isinstance(ys[0], torch.Tensor):
                    ys = ys.cuda()
            for loss_id in range(num_losses):
                new_loss = losses[loss_id].step(xs, ys, mode, batch)
                epoch_losses_sup[loss_id] += new_loss
        except StopIteration:
            pass
    return epoch_losses_sup

def load_and_process_data(config):
    """
    Load, preprocess, and prepare DataLoaders for RNA and ATAC data.
    
    Args:
        config: Configuration object containing paths and hyperparameters.
        
    Returns:
        dataloaders (dict): Dictionary containing the DataLoaders.
        datasets (dict): Dictionary containing the Datasets.
        dims (dict): Dictionary containing input dimensions (rna_input_size, atac_input_size).
        other_info (dict): Dictionary containing other info (batch_num, no_class).
    """
    
    # Load data 
    try: 
        rna = ad.read_h5ad(config.RNA_PATH) 
    except FileNotFoundError: 
        print(f"File not found: {config.RNA_PATH}. Please check your data directory.")
        return None
        
    print(f"Loading ATAC data from {config.ATAC_PATH}...") 
    try: 
        atac = ad.read_h5ad(config.ATAC_PATH) 
    except FileNotFoundError: 
        print(f"File not found: {config.ATAC_PATH}. Please check your data directory.") 
        return None

    # Basic preprocessing 
    rna = process(rna) 
    atac = process(atac) 
    rna.obs['omic'] = 1 
    rna.obs['batch'] = 0 
    atac.obs['batch'] = 0 

    # Align RNA and ATAC data by Union HVG 
    rna_final, atac_final, final_feature_list = align_rna_atac_by_union_hvg(rna, atac) 

    # Normalization adjustment 
    atac_final.X = atac_final.X * rna_final.X.mean() / atac_final.X.mean() 

    print("Data alignment complete.") 
    
    # RNA Splitting 
    rna_final.obs['omic'] = 1 
    train_idx, val_idx = train_test_split( 
        np.arange(len(rna_final)), 
        test_size=0.3, 
        stratify=rna_final.obs['cell_type'], 
        random_state=42 
    ) 
    RNACE_train = rna_final
    RNACE_test = rna_final[val_idx, :] 

    # ATAC Splitting 
    atac_final.obs['omic'] = 0 
    train_idx, val_idx = train_test_split( 
        np.arange(len(atac_final)), 
        test_size=0.8, 
        stratify=atac_final.obs['cell_type'], 
        random_state=42 
    ) 
    ATACCE_train = atac_final[train_idx, :] 
    ATACCE_test = atac_final[val_idx, :] 
    
    # Prepare label mappings 
    cell_types = ATACCE_test.obs['cell_type'].astype('category').cat.categories 
    category_to_index = {category: index for index, category in enumerate(sorted(set(cell_types)))} 
    index_to_category = {index: category for category, index in category_to_index.items()} 

    all_batches = atac_final.obs['batch'].unique() 
    batch_to_index = {name: i for i, name in enumerate(all_batches)} 
    batch_num = len(all_batches) 

    # Create Datasets 
    rna_train_labeled_dataset = RNA( 
        rna_final.X, 
        np.array(map_names_to_indices(rna_final.obs['cell_type'], category_to_index)), 
        rna_final.obs['omic'], 
        rna_final.obs_names, 
        map_batch_to_indices(rna_final.obs['batch'], batch_to_index), 
        transform=None, 
        train=True, 
        batch_num=batch_num, 
        no_class=config.NO_CLASS 
    ) 

    rna_test_rna_dataset = RNA_TEST( 
        RNACE_test.X, 
        np.array(map_names_to_indices(RNACE_test.obs['cell_type'], category_to_index)), 
        RNACE_test.obs['omic'], 
        RNACE_test.obs_names, 
        map_batch_to_indices(RNACE_test.obs['batch'], batch_to_index), 
        train=False, 
        transform=None, 
        batch_num=batch_num, 
        no_class=config.NO_CLASS 
    ) 

    atac_train_labeled_dataset = ATAC( 
        ATACCE_train.X, 
        np.array(map_names_to_indices(ATACCE_train.obs['cell_type'], category_to_index)), 
        ATACCE_train.obs['omic'], 
        ATACCE_train.obs_names, 
        map_batch_to_indices(ATACCE_train.obs['batch'], batch_to_index), 
        transform=None, 
        train=True, 
        batch_num=batch_num, 
        no_class=config.NO_CLASS 
    ) 

    atac_train_unlabeled_dataset = ATAC( 
        ATACCE_test.X, 
        np.array(map_names_to_indices(ATACCE_test.obs['cell_type'], category_to_index)), 
        ATACCE_test.obs['omic'], 
        ATACCE_test.obs_names, 
        map_batch_to_indices(ATACCE_test.obs['batch'], batch_to_index), 
        transform=None, 
        train=True, 
        batch_num=batch_num, 
        no_class=config.NO_CLASS 
    ) 

    atac_test_dataset = ATAC_TEST( 
        ATACCE_test.X, 
        np.array(map_names_to_indices(ATACCE_test.obs['cell_type'], category_to_index)), 
        ATACCE_test.obs['omic'], 
        ATACCE_test.obs_names, 
        map_batch_to_indices(ATACCE_test.obs['batch'], batch_to_index), 
        train=False, 
        transform=None, 
        batch_num=batch_num, 
        no_class=config.NO_CLASS 
    ) 

    # Create DataLoaders 
    labeled_rna_trainloader = DataLoader(rna_train_labeled_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, drop_last=True) 
    data_loader = DataLoader(rna_test_rna_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, drop_last=True) 
    labeled_atac_trainloader = DataLoader(atac_train_labeled_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, drop_last=True) 
    unlabeled_atac_trainloader = DataLoader(atac_train_unlabeled_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, drop_last=True) 
    atac_test_loader = DataLoader(atac_test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, drop_last=True) 
    
    print("DataLoaders created.")
    
    return labeled_rna_trainloader, data_loader, labeled_atac_trainloader, unlabeled_atac_trainloader, atac_test_loader, index_to_category, batch_num, {
        'rna_train_labeled_dataset': rna_train_labeled_dataset,
        'rna_test_rna_dataset': rna_test_rna_dataset,
        'atac_train_labeled_dataset': atac_train_labeled_dataset,
        'atac_train_unlabeled_dataset': atac_train_unlabeled_dataset,
        'atac_test_dataset': atac_test_dataset
    }, {
        "ATACCE_train" : ATACCE_train,
        "ATACCE_test" : ATACCE_test,
        "RNACE_train" : RNACE_train,
        "RNACE_test" : RNACE_test 
    }
