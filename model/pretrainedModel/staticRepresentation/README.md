# Pretrain static Representation Models
## 1.preprocess
```
python preprocess.py --data_path [data path] --output_path [output path]
```
## 2.pretrain
```
python train.py --model_name_or_path graphcodebert-base --train_data_file [output_path] --per_device_train_batch_size 8 --do_train --output_dir [the path to save the model] --mlm --overwrite_output_dir --line_by_line
```