<img src="resources/logo.png" align="right" width="250px" height="250px">

# Elastic Malware Benchmark for Empowering Researchers

The EMBER dataset is a collection of features from PE files that serve as a benchmark dataset for researchers. The EMBER2017 dataset contained features from 1.1 million PE files scanned in or before 2017 and the EMBER2018 dataset contains features from 1 million PE files scanned in or before 2018. This repository makes it easy to reproducibly train the benchmark models, extend the provided feature set, or classify new PE files with the benchmark models.

This paper describes many more details about the dataset: [https://arxiv.org/abs/1804.04637](https://arxiv.org/abs/1804.04637)

## Features

The [LIEF](https://lief.quarkslab.com/) project is used to extract features from PE files included in the EMBER dataset. Raw features are extracted to JSON format and included in the publicly available dataset. Vectorized features can be produced from these raw features and saved in binary format from which they can be converted to CSV, dataframe, or any other format. This repository makes it easy to generate raw features and/or vectorized features from any PE file. Researchers can implement their own features, or even vectorize the existing features differently from the existing implementations.

The feature calculation is versioned. Feature version 1 is calculated with the LIEF library version 0.8.3. Feature version 2 includes the additional data directory feature, updated ordinal import processing, and is calculated with LIEF library version 0.9.0.  We have verified under Windows and Linux that LIEF provides consistent feature representation for version 2 features using LIEF version 0.10.1 and that it does not on a Mac.

## Years

The first EMBER dataset consisted of version 1 features calculated over samples collected in or before 2017. The second EMBER dataset release consisted of version 2 features calculated over samples collected in or before 2018. In conjunction with the second release, we also included the version 2 features from the samples collected in 2017. Combining the data from 2017 and 2018 will allow longer longitudinal studies of the evolution of features and PE file types. But different selection criteria were applied when choosing samples from 2017 and 2018. Specifically, the samples from 2018 were chosen so that the resultant training and test sets would be harder for machine learning algorithms to classify. Please beware of this inconsistancy while constructing your multi-year studies. The original paper only describes Ember 2017 (featur version 1). For a detailed information about the Ember 2018 dataset, please refer to https://www.camlis.org/2019/talks/roth where you can find both [slides](https://docs.google.com/presentation/d/1A13tsUkgWeujTy9SD-vDFfQp9fnIqbSE_tCihNPlArQ/edit#slide=id.g476bf81b41_0_446) and a [video talk](https://youtu.be/MsZmnUO5lkY).

## Download

Download the data here:

| Year | Feature Version | Filename                     | URL                                                                                                            | sha256                                                             |
|------|-----------------|------------------------------|----------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| 2017 | 1               | ember_dataset.tar.bz2        | [https://ember.elastic.co/ember_dataset.tar.bz2](https://ember.elastic.co/ember_dataset.tar.bz2)               | `a5603de2f34f02ab6e21df7a0f97ec4ac84ddc65caee33fb610093dd6f9e1df9` |
| 2017 | 2               | ember_dataset_2017_2.tar.bz2 | [https://ember.elastic.co/ember_dataset_2017_2.tar.bz2](https://ember.elastic.co/ember_dataset_2017_2.tar.bz2) | `60142493c44c11bc3fef292b216a293841283d86ff58384b5dc2d88194c87a6d` |
| 2018 | 2               | ember_dataset_2018_2.tar.bz2 | [https://ember.elastic.co/ember_dataset_2018_2.tar.bz2](https://ember.elastic.co/ember_dataset_2018_2.tar.bz2) | `b6052eb8d350a49a8d5a5396fbe7d16cf42848b86ff969b77464434cf2997812` |


## Installation
### Instrall directly from git
Use `pip` to install the `ember` and required files

```
pip install git+https://github.com/elastic/ember.git
```

This provides access to EMBER feature extaction for example.  However, to use the scripts to train the model, one would instead clone the repository.


### Install after cloning the EMBER repository
Use `pip` or `conda` to install the required packages before installing `ember` itself:

```
pip install -r requirements.txt
python setup.py install
```

```
conda config --add channels conda-forge
conda install --file requirements_conda.txt
python setup.py install
```

### Notes on LIEF versions

LIEF is now pinned to version 0.9.0 in the provided requirements files. This default behavior will allow new users to immediately reproduce EMBER version 2 features. LIEF 0.9.0 will not install on an M1 Mac, though. For those users, a Dockerfile is now included that installs the dependencies using conda.

EMBER will work with more recent releases of LIEF, but keep in mind that models trained on features generated with one version of LIEF will have unpredictable results when evaluating on features generated with another.

## Scripts

The `train_ember.py` script simplifies the model training process. It will vectorize the ember features if necessary and then train the LightGBM model.

```
python train_ember.py [/path/to/dataset]
```

The `classify_binaries.py` script will return model predictions on PE files.

```
python classify_binaries.py -m [/path/to/model] BINARIES
```

## Import Usage

The raw feature data can be expanded into vectorized form on disk for model training and into metadata form. These two functions create those extra files:

```
import ember
ember.create_vectorized_features("/data/ember2018/")
ember.create_metadata("/data/ember2018/")
```

Once created, that data can be read in using convenience functions:

```
import ember
X_train, y_train, X_test, y_test = ember.read_vectorized_features("/data/ember2018/")
metadata_dataframe = ember.read_metadata("/data/ember2018/")
```

Once the data is downloaded and the ember module is installed, this simple code should reproduce the benchmark ember model:

```
import ember
ember.create_vectorized_features("/data/ember2018/")
lgbm_model = ember.train_model("/data/ember2018/")
```

Once the model is trained, the ember module can be used to make a prediction on any input PE file:

```
import ember
import lightgbm as lgb
lgbm_model = lgb.Booster(model_file="/data/ember2018/ember_model_2018.txt")
putty_data = open("~/putty.exe", "rb").read()
print(ember.predict_sample(lgbm_model, putty_data))
```

## Citing

If you use this data in a publication please cite the following [paper](https://arxiv.org/abs/1804.04637):

```
H. Anderson and P. Roth, "EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models”, in ArXiv e-prints. Apr. 2018.

@ARTICLE{2018arXiv180404637A,
  author = {{Anderson}, H.~S. and {Roth}, P.},
  title = "{EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models}",
  journal = {ArXiv e-prints},
  archivePrefix = "arXiv",
  eprint = {1804.04637},
  primaryClass = "cs.CR",
  keywords = {Computer Science - Cryptography and Security},
  year = 2018,
  month = apr,
  adsurl = {http://adsabs.harvard.edu/abs/2018arXiv180404637A},
}
```
