# Pretrain static Representation Models
## 1.preprocess
```
conda activate pytorch1.7.1
cd ./concoction/pretrainedModel/staticRepresentation
python preprocess.py --data_path ../data/dataset --output_path ../data/output_static.txt
```

## 2.pretrain
```
conda activate pytorch1.7.1
cd ./concoction/pretrainedModel/staticRepresentation
python train.py --model_name_or_path graphcodebert-base --train_data_file ../data/output_static.txt --per_device_train_batch_size 4 --do_train --output_dir ./trainedModel --mlm --overwrite_output_dir --line_by_line
```