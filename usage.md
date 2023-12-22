# Usage Instructions: Combining Static and Dynamic Code Information for Software Vulnerability Prediction

## Preliminaries

This interactive markdown file provides a small-scale demo to showcase the program representation, vulnerability detection, and prediction of vulnerability detection discussed in the paper.

The main results of our ICSE 2024 paper involve comparing the performance of our vulnerability detection with prior machine learning-based approaches. The evaluation presented in our paper was conducted on a much larger dataset and for a longer duration. The intention of this instruction is to provide minimal working examples that can be evaluated within a reasonable time frame.

# The Concoction Model Architecture

Note that This is a small-scale demo for vulnerability detection. The full-scale evaluation used in the paper takes over 24 hours to run.

## Step 1. Program representation

The program representation component maps the input source code and dynamic symbolic execution traces of the target function into a numerical embedding vector.

#### *Static representation model  training*:

```
$ conda activate pytorch1.7.1
$ cd src/pretrain_model
$ python train.py --model_name_or_path bert-base-uncased     --train_file data/Ours/BUG/output_static.txt   --output_dir result/my-unsup-simcse-bert-base-uncased     --num_train_epochs 1     --per_device_train_batch_size 32     --learning_rate 3e-5     --max_seq_length 32      --metric_for_best_model stsb_spearman     --load_best_model_at_end     --eval_steps 2     --pooler_type cls     --mlp_only_train     --overwrite_output_dir     --temp 0.05     --do_train
```

#### *Dynamic representation model* training:

```
$ python train.py --model_name_or_path bert-base-uncased     --train_file data/Ours/BUG/output_dynamic.txt   --output_dir result/my-unsup-simcse-bert-base-uncased     --num_train_epochs 1     --per_device_train_batch_size 32     --learning_rate 3e-5     --max_seq_length 32      --metric_for_best_model stsb_spearman     --load_best_model_at_end     --eval_steps 2     --pooler_type cls     --mlp_only_train     --overwrite_output_dir     --temp 0.05     --do_train
```

## Step 2. Vulnerability Detection

Concoction’s detection component takes the joint embedding as input to predict the presence of vulnerabilities. Our current implementation only identifies whether a function may contain a vulnerability or bug and does not specify the type of vulnerability. Here we use SARD benchmarks.

#### *Vulnerability Detection model training*:

```
$ cd src/detection_model
$ python evaluation_bug.py --model_name_or_path princeton-nlp/sup-simcse-bert-base-uncased --mode test
```

## Step 3. Deployment

This demo shows how to deploy our trained model on a real world project. Here we apply the xx as our test project.

#### *Path Selection for Symbolic Execution*:

After training the end-to-end model, we develop a path selection component to automatically select a subset of important paths whose dynamic traces are likely to improve prediction accuracy during deployment.

*approximate runtime ~ 30 minutes*

```
$ cd src/path_selection
$ python train.py
```

#### *Fuzzing for Test Case Generation*:

We use fuzzing techniques to generate test cases for functions predicted to contain potential vulnerabilities, aiming to automate the testing process and minimize the need for manual inspection.