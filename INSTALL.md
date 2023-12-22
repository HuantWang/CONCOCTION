# Installation

Concoction was tested with Python 3.6 and Ubuntu 18.04.

#### 1) Clone this repo

``` console
$ git clone git@github.com:Anonymization
```

#### 2) Install Prerequisites

``` console
$ conda env create -f environment.yml
```
[environment.yml](https://github.com/HuantWang/CONCOCTION/blob/main/environment.yml)

#### 3) train the detection model 

``` console
$ cd src/detection_model
$ python evaluation_bug.py --model_name_or_path princeton-nlp/sup-simcse-bert-base-uncased --mode test
```
