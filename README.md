# ENSO-GTC 1.0.0

This is the code for this [paper](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022MS003132) `https://doi.org/10.1029/2022MS003132`

This project can be built and trained on Ubuntu 18.04.3 LTS, with python3.7 and CUDA 10.0/cudnn 7.6.5.

### 0. Environment
```
conda create -n enso python=3.7
source activate enso

pip install [some torch-related libs](https://drive.google.com/drive/folders/1hHQC0Ku1Vm4pLd2F3wVb2f5wnxx9ZyH6?usp=sharing)
pip install netCDF4==1.5.3
pip install progress==1.5
pip install loguru==0.3.2
pip install cmaps
pip install pyproj
pip install h5py
conda install -c conda-forge cartopy
```

### 1. Download climate dataset
[Met Office Hadley Centre observations datasets](https://www.metoffice.gov.uk/hadobs/hadisst/data/download.html) (HadISST) is used for this model. Download it and put it in `./file/`.

The archieved dataset is also in [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6426550.svg)](https://doi.org/10.5281/zenodo.6426550) (not the latest!)

### 2. For independent training and forecasting processes
Firstly, use the following commands to parse and parpare training data.
```
python -m data.prepare_data
```
The output training data files are also in `./file/`

Then, train the model:
```
python -m train_multi_gpus
```

### 3. Monthly ENSO forecasting
Firstly, download the latest HadISST from the above wetsites and replace the new data for data preprocessing.

Secondly, fine-tune the trained model:
```
python workflow.py
```

Finally, make forecasts for the future 18 months:
```
python forecast.py
```
The forecast results will be recorded in `./result-{year}-{month}.csv`
