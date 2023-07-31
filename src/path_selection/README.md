## On Deep Unsupervised Active Learning

The code of the IJCAI-2020 paper ["On Deep Unsupervised Active Learning"](https://www.ijcai.org/proceedings/2020/0364.pdf)

### Environments
- pytorch >= 1.5.0
- numpy >= 1.14.3
- sklearn >= 0.23.1
- json
- configparser
- argparse

### How to use
To train the model, run:

`python train.py`

The results will be obtained in ./log/$DATASET/

If you want to train your own dataset, please modify the config file in ./config/$DATASET/.
