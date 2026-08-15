library(Signac)
library(Seurat)
library(EnsDb.Hsapiens.v75)

# Load 10x ATAC counts
counts <- Read10X_h5("pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5")

# Create chromatin assay
chrom_assay <- CreateChromatinAssay(
  counts = counts$Peaks,
  sep = c(":", "-"),
  genome = "hg19",
  fragments = "pbmc_granulocyte_sorted_10k_atac_fragments.tsv.gz",
  min.cells = 10,
  min.features = 200
)

# Create Seurat object for ATAC
pbmc <- CreateSeuratObject(
  counts = chrom_assay,
  assay = "peaks"
)

# Load hg19 gene annotations
annotations <- GetGRangesFromEnsDb(EnsDb.Hsapiens.v75)
seqlevelsStyle(annotations) <- "UCSC"
Annotation(pbmc) <- annotations

# Compute raw gene activity counts
gene.activities <- GeneActivity(pbmc)

# Add as new assay
pbmc[["act"]] <- CreateAssayObject(counts = gene.activities)

# Optional: set default assay to gene activity
DefaultAssay(pbmc) <- "act"

# Save the raw gene activity matrix for downstream use
raw_gene_activity <- GetAssayData(pbmc, assay = "act", slot = "counts")