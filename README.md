
# Urban Garden Ground Cover  Semantic Segmentation


**Code for the Paper**: Afrasiabian et al., Annotated centimeter resolution imagery dataset for Deep-learning based Semantic Segmentation in heterogeneous urban gardens to support biodiversity surveys.

Data are avaible on Zonedo
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15261988.svg)](https://doi.org/10.5281/zenodo.15261988)

## 1. Overview  
This repository provides high-resolution Uncrewed Aerial Vehicle (UAV) orthomosaic RGB imagery and corresponding ground-truth masks for semantic segmentation of ground cover types in urban community gardens. Collected from five community gardens in Munich, Germany, during 2021 and 2022, the dataset supports research in urban ecology, remote sensing, and machine learning.

## 2. Dataset  
- **Imagery**: 24 RGB GeoTIFFs with a spatial resolution of 3.2–7.9 mm, projected in EPSG:25832.
- **Masks**: Single-band GeoTIFFs saved in the same folder as the RGB images, with filenames appended with `_Labelled`. Each mask includes eight classes: grass, herb, litter, soil, stone, straw, wood, and woodchip.
- **Split**: Images are divided into 14 `train/`, 5 `val/`, and 5 `test/` sets.
- **Patches**: Each orthomosaic image is cropped into 512 × 512 px patches with a 256 px stride, suitable for deep learning applications.
- **Metadata**: `dataset_metadata.csv` detailing image geometry, Metashape settings, and flight parameters.

## 3. Code
For each model (e.g., Random Forest, XGBoost, UNet, DeepLabV3+), there is a specific script containing preprocessing, training, validation, and evaluation on the test subset.

## 4. Results
Each model directory contains: 
- **confusion_matrix.text**:	Visual representation of the test set confusion matrix.
- **metrics.txt**:	Overall and per-class performance metrics and timing.
- **loss.text**:	Epoch-wise training and validation loss, accuracy, and Cohen's Kappa.

## 5. License
- **Code**: Released under the MIT License.
- **Data**: © 2025 Yasamin Afrasiabian, Anirudh Belwalkar. Distributed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

## 6. Citation
If you use this dataset or code in your research, please cite:

@dataset{afrasiabian_2025_ugc,
  author    = {Yasamin Afrasiabian and others},
  title     = {Annotated centimeter resolution imagery dataset for Deep-learning based Semantic Segmentation in heterogeneous urban gardens to support biodiversity surveys},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.15496503}
}

@software{afrasiabian_2025_ugc_code,
  author    = {Yasamin Afrasiabian and others},
  title     = {UGC-Mapping – Code and Benchmark},
  year      = {2025},
  url       = {https://github.com/paglab/ugc-mapping},
  version   = {1.0.0},
  license   = {MIT}
}
