# path select
[data_path]:Dataset for detection
[stored_path]:Storage path of preprocessed data
## 1.preprocess
```
conda activate pytorch1.7.1
cd ./concoction/pathSelection
python preprocess.py --data_path ../data/dataset0 --stored_path ../data/dataset0_pathselect

```
## 2.path select

[data_path]:Dataset for execution path selection
[stored_path]:Storage path of results
```
python train.py --data_path ../data/dataset0_pathselect --stored_path ../data/dataset0_pathselect_result
```