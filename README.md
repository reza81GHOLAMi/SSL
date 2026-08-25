# UaVirBASE: A Public-Access Unmanned Aerial Vehicle Sound Source Localization Dataset

## Overview

Welcome to **UaVirBASE**, a comprehensive dataset for Unmanned Aerial Vehicle (UAV) sound source localization. This repository provides the dataset, code, and resources to reproduce the experiments and results presented in our paper. The dataset is designed to support research in UAV sound source localization.

## Getting Started

The source code includes all configurations, model processing operations, and training procedures necessary to replicate our results.

### Clone the Repository

To clone this repository, use the following command:

```bash
git clone https://gitlab.com/g.jekaterynczuk/uavirbase_ssl.git
```

## Dataset

The UaVirBASE dataset consists of high-quality UAV audio recordings and accompanying metadata for sound source localization. It includes:

* Audio recordings: UAV sounds captured under various conditions, including different distances, angles, and ambient noise levels.
* Metadata: Detailed information such as UAV position (distance, height, angle, and orientation relative to the microphone array), environmental conditions, and exact microphone locations in WGS84 coordinates.

Dataset is available under the following link: https://zenodo.org/records/15391924

## Usage

* Set dataset path - 
after downloading the dataset, open postprocess.py and update the base_folder_path variable to match the location where your dataset is stored.

* Run preprocessing - 
execute postprocess.py and select option 3 to preprocess the raw data.
Once this step completes, run the script again and select option 4 to divide the dataset.

* Start training - 
with the processed dataset ready, you can start training by running main.py.
If you're using Weights & Biases for logging, make sure to fill in your own WandB project and entity details (or disable WandB if you don't plan to use it).

## Citation

If you use this dataset or code in your work, please cite the following publication:
> Jekateryńczuk, G.; Szadkowski, R.; Piotrowski, Z. UaVirBASE: A Public-Access Unmanned Aerial Vehicle Sound Source Localization Dataset. Appl. Sci. 2025, 15, 5378. https://doi.org/10.3390/app15105378