# Pretrain Dynamic representation model
## 1.data preprocess
```
conda activate pytorch1.7.1
cd ./concoction/pretrainedModel/dynamicRepresentation
python preprocess.py --data_path ../data/dataset  --output_path ../data/output_dynamic.txt
```

## 2.train
```
conda activate pytorch1.7.1
cd ./concoction/pretrainedModel/dynamicRepresentation
python train.py --model_name_or_path bert-base-uncased     --train_file ../data/output_dynamic.txt   --output_dir ./result    --num_train_epochs 1     --per_device_train_batch_size 4     --learning_rate 3e-5     --max_seq_length 32      --metric_for_best_model stsb_spearman  --load_best_model_at_end     --eval_steps 2     --pooler_type cls     --mlp_only_train     --overwrite_output_dir     --temp 0.05     --do_train
```