# CARA
CARA: Robust annotation and discovery of novel cell types in single-cell ATAC-seq data through cross-modal reference alignment

Prerequisites
-----
- Python 3.9
- PyTorch 1.12.1+cu116 (GPU recommended)


Installation
-----
```bash

git clone https://github.com/Ying-Lab/CARA
cd CARA
conda create -n cara python=3.9
conda activate cara
pip install --index-url https://download.pytorch.org/whl/cu116 \
  "torch==1.12.1+cu116" "torchvision==0.13.1+cu116" "torchaudio==0.12.1+cu116"
pip install -r requirements.txt
```

Example
-----
 Refer to the demo : Cara kidney demo.ipynb