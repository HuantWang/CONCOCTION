# Concoction

## train
[path_to_data]:Dataset for detection
[mode]:Used for model training
```shell
conda activate pytorch1.7.1
cd ./concoction/detectionModel
python evaluation_bug.py --path_to_data ../data/data/train --mode train
```
[path_to_data]:Dataset for detection
[mode]:Used for model training
[model_to_load]:Load the model for prediction
## test
```shell
conda activate concoction
cd ./concoction/detectionModel
python evaluation_bug.py --path_to_data ../data/data/test --mode test --model_to_load ./trained_model/github.h5
```
