## Description

### Preprocessing

Each subject and each signal is pre-processed using: 
* 3-97% winsorization, which removes extreme values from the signal data;
* Butterworth low-pass filter with 10 Hz cut-off, removes components above the threshold frequency of 10 Hz;
* downsampling, which  reduces the dimensionality of the inputs; it consequently decreases the number of learning parameters in the DL models;
* min-max normalization. 

### Used architectures

Code includes the following implementations:
* FCN, Resnet, Encoder, MCDCNN, Time-CNN and MLP were based on: https://github.com/hfawaz/dl-4-tsc,
* Inception based on: https://github.com/hfawaz/InceptionTime,
* MLP-LSTM was developed by us based on the above MLP implementation,
* CNN-LSTM was implemented based on description provided by Kanjo at al. (https://doi.org/10.1016/j.inffus.2018.09.001),
* Stresnet was used from the previous work of Gjoreski at al. (https://doi.org/10.1109/ACCESS.2020.2986810).

All architectures, except for MCDCNN, were adjusted for multi-source data - separate inputs for each signal. 

## How to run?

### Environment preparation

Prepare Anaconda / virtual environment with Python. Install packages from `utils/requirements.txt`.

### Data preparation

1. Create `archive` folder and put there WESAD, AMIGOS, ASCERTAIN and DECAF datasets in separate folders.
2. Create file `config.ini` with the following content:
    ```
    [Paths]
    root_dir = ... # directory path of project
    mts_out_dir = ..  # output directory for preprocessed datasets
    wesad_dir = archives/WESAD
    decaf_dir = archives/DECAF
    ascertain_dir = archives/ASCERTAIN
    amigos_dir = archives/Amigos
    ```
3. Run `ar_datasets_preprocessing.py`. Datasets should be successfully preprocessed.

### Tuning and results collection

In order to train models you need to run `.\tuning.sh X` where X is the id of GPU on which you want to run training.
If you want to gather all results, run `results.py`.

## Reference

If you re-use this work, please cite:

```
@article{.....
}
```

