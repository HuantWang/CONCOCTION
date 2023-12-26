# Pretrain Dynamic representation model
## 1.data preprocess
```
python preprocess.py --data_path /home/CONCOCTION/model/data/Ours/BUG --output_path /home/CONCOCTION/model/data/Ours/output_simcse.txt
```

## 2.train
```
python train.py --model_name_or_path bert-base-uncased     --train_file /home/CONCOCTION/model/data/Ours/output_simcse.txt   --output_dir ./result    --num_train_epochs 1     --per_device_train_batch_size 32     --learning_rate 3e-5     --max_seq_length 32      --metric_for_best_model stsb_spearman  --load_best_model_at_end     --eval_steps 2     --pooler_type cls     --mlp_only_train     --overwrite_output_dir     --temp 0.05     --do_train
```