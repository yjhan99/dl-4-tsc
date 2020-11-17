## Description

This is a supplementary repository for article titled [Can We Ditch Feature Engineering? End-to-End Deep Learning for Affect Recognition from Physiological Sensor Data](https://doi.org/10.3390/s20226535).

### Preprocessing

Each subject and each signal is pre-processed using: 
* 3-97% winsorization, which removes extreme values from the signal data;
* Butterworth low-pass filter with 10 Hz cut-off, removes components above the threshold frequency of 10 Hz;
* downsampling, which  reduces the dimensionality of the inputs; it consequently decreases the number of learning parameters in the DL models;
* min-max normalization. 

### Used architectures
* Implementation of FCN, Resnet, Encoder, MCDCNN, Time-CNN, and MLP was based on code provided at https://github.com/hfawaz/dl-4-tsc,
* Implementation of Inception was based on code provided at https://github.com/hfawaz/InceptionTime,
* Implementation of MLP-LSTM was based on the above MLP implementation
* Implementation of CNN-LSTM was based on the description provided by [Kanjo at al.](https://doi.org/10.1016/j.inffus.2018.09.001),
* Implementation of Stresnet code was used from the previous work by [Gjoreski at al.](https://doi.org/10.1109/ACCESS.2020.2986810).

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
Dzieżyc, M.; Gjoreski, M.; Kazienko, P.; Saganowski, S.; Gams, M. Can We Ditch Feature Engineering? End-to-End Deep Learning for Affect Recognition from Physiological Sensor Data. Sensors 2020, 20, 6535. 
```

```
@Article{s20226535,
AUTHOR = {Dzieżyc, Maciej and Gjoreski, Martin and Kazienko, Przemysław and Saganowski, Stanisław and Gams, Matjaž},
TITLE = {Can We Ditch Feature Engineering? End-to-End Deep Learning for Affect Recognition from Physiological Sensor Data},
JOURNAL = {Sensors},
VOLUME = {20},
YEAR = {2020},
NUMBER = {22},
ARTICLE-NUMBER = {6535},
URL = {https://www.mdpi.com/1424-8220/20/22/6535},
ISSN = {1424-8220},
ABSTRACT = {To further extend the applicability of wearable sensors in various domains such as mobile health systems and the automotive industry, new methods for accurately extracting subtle physiological information from these wearable sensors are required. However, the extraction of valuable information from physiological signals is still challenging&mdash;smartphones can count steps and compute heart rate, but they cannot recognize emotions and related affective states. This study analyzes the possibility of using end-to-end multimodal deep learning (DL) methods for affect recognition. Ten end-to-end DL architectures are compared on four different datasets with diverse raw physiological signals used for affect recognition, including emotional and stress states. The DL architectures specialized for time-series classification were enhanced to simultaneously facilitate learning from multiple sensors, each having their own sampling frequency. To enable fair comparison among the different DL architectures, Bayesian optimization was used for hyperparameter tuning. The experimental results showed that the performance of the models depends on the intensity of the physiological response induced by the affective stimuli, i.e., the DL models recognize stress induced by the Trier Social Stress Test more successfully than they recognize emotional changes induced by watching affective content, e.g., funny videos. Additionally, the results showed that the CNN-based architectures might be more suitable than LSTM-based architectures for affect recognition from physiological sensors.},
DOI = {10.3390/s20226535}
}
```

