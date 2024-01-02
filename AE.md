# Combining Static and Dynamic Code Information for Software Vulnerability Prediction: Artifact Instructions for Docker Image

The recommended approach for AE is to use the pre-configured, interactive Jupyter notebook with the instructions given in the AE submission.

The following step-by-step instructions are provided for using a Docker Image running on a local host. Our Docker image (50 GB uncompressed) contains the entire execution environment (including Python and system dependencies), benchmarks, and source code, which includes 16 state-of-the-art bug detection frameworks. All of our code and data are open-sourced and have been developed with extensibility as a primary goal.

Please check [the notebook](http://43.129.205.177:8080/) for a small-scale demo showcasing vulnerability detection and evaluation as described in the paper.

*Disclaim: Although we have worked hard to ensure that our AE scripts are robust, our tool remains a \*research prototype\*. It may still have glitches when used in complex, real-life settings. If you discover any bugs, please raise an issue, describing how you ran the program and the problem you encountered. We will get back to you ASAP. Thank you.*

# Step-by-Step Instructions 

## ★ Main Results 

The main results of our work are presented in Figures 7 to 8 and Tables 3 to 5 in the submitted paper, which compare Concoction against 16 alternative techniques for bug detection.

## ★ Docker Image 

We prepare our artifact within a Docker image to run "out of the box". 
Our docker image was tested on a host machine running Ubuntu 18.04.

## ★ Artifact Evaluation  

Follow the instructions below to use our AE evaluation scripts.

### 1. Setup

Install Docker by following the instructions [here](https://docs.docker.com/install/linux/docker-ce/ubuntu/). The following instructions assume the host OS runs Linux.

#### 1.1  Fetch the Docker Image

Fetch the docker image from docker hub.

```
$ sudo docker pull nwu/concoction:v1
```

To check the list of images, run:

```
$ sudo docker images

```

Run the docker image.

```
$ docker run -itd --gpus all  -p 10052:22 10051:8888 --name concoction nwu/concoction:v1 /bin/bash
$ docker start concoction 
$ docker exec -it concoction /bin/bash
```


#### 1.2 Setup the Environment

After importing the docker container **and getting into bash** in the container, run the following command to select the conda environment, before using any of the AE scripts:

`````` shell
$ conda activate pytorch1.7.1
``````

Then, go to the root directory of our tool:

```
(docker) $ cd /homee/
```

### 2. Evaluation

The following steps describe how to evaluate our technique for bug detection. First, we explain how to train the detection model using the Concoction dataset. Then, we demonstrate how to evaluate the performance of our detection model and compare the results against the baselines.

#### 2.1. **Detect Vulnerabilities in Large-scale Testing**

This section provides a quantified summary of Concoction's performance in detecting function-level code vulnerabilities across the 20 projects listed in Table 3 in our paper. Please follow this part(https://github.com/HuantWang/CONCOCTION/tree/main/vul_info) for detailed information.

#### 2.2. **Comparison on Open Datasets**

***Note:*** **Make sure the environment is chosen correctly as [this table](https://github.com/HuantWang/CONCOCTION/blob/AE/INSTALL.md#alternative-using-different-environments-to-evaluate-other-sota-approaches)**. 

##### 2.2.1.  *SARD dataset*

The results presented here correspond to Figure 7 in the submitted manuscript. This evaluation reports the "higher-is-better" metrics (accuracy, precision, recall, and F1 score) achieved by Concoction and 11 baseline methods on the SARD dataset.

(*approximate runtime:  **~ 300 minutes**, ~ 30 minutes for each baseline model)

```shell
#Client RL search
(docker) $ python /homee/Evaluation/demo2/demo2.py --dataset CWE-416 --model vuldeepecker
```

You can change the following parameters:

```--dataset ``` Perform comparative experiments using different types of datasets.
Note: Eight different public vulnerability datasets can be selected here, namely: CWE-416, CWE-789, CWE-78, CWE-124, CWE-122, CWE-190, CWE-191, CWE-126.

```--model``` Perform experiments with various comparative tasks.


##### 2.2.2. **CVE dataset**

The results presented here correspond to Figure 8 in the submitted manuscript. This evaluation reports the "higher-is-better" metrics (accuracy, precision, recall, and F1 score) achieved by Concoction and 11 baseline methods on the CVE dataset.

(*approximate runtime:  **~ 300 minutes**, ~ 30 minutes for each baseline model)

```shell
#Client RL search
(docker) $ python /homee/Evaluation/demo2/demo2.py --dataset Github --model reveal
```

You can change the following parameters:

```--dataset ``` Perform comparative experiments using different types of datasets.

```--model``` Perform experiments with various comparative tasks.


#### 2.3. **Comparison on Known CVEs**

The results presented here correspond to Table 5 in the submitted manuscript. We compare Concoction to the baselines on three open-source projects: SQLite, Libtiff, Libpng. This contain 35 CVEs reported by independent users. We  train all methods on the same training dataset and test on these three projects.

(*approximate runtime:  **~ 300 minutes**, ~ 30 minutes for each baseline model)

```shell
(docker) $ python tasks/halide/run.py
```



# The Concoction Tutorial 

## Step 1. Program representation

The program representation component maps the input source code and dynamic symbolic execution traces of the target function into a numerical embedding vector.

#### *Static representation model  training*:

## 1.preprocess
```
$ conda activate pytorch1.7.1
$ cd ./src/concoction/pretrainedModel/staticRepresentation
$ python preprocess.py --data_path ../data/dataset --output_path ../data/output_static.txt
```

## 2.pretrain
```
$ conda activate pytorch1.7.1
$ cd ./src/concoction/pretrainedModel/staticRepresentation
$ python train.py --model_name_or_path graphcodebert-base --train_data_file ../data/output_static.txt --per_device_train_batch_size 8 --do_train --output_dir ./trainedModel --mlm --overwrite_output_dir --line_by_line
```

#### *Dynamic representation model* training:

## 1.data preprocess
```
$ conda activate pytorch1.7.1
$ cd ./src/concoction/pretrainedModel/dynamicRepresentation
$ python preprocess.py --data_path ../data/dataset  --output_path ../data/output_dynamic.txt
```

## 2.train
```
$ conda activate pytorch1.7.1
$ cd ./src/concoction/pretrainedModel/dynamicRepresentation
$ python train.py --model_name_or_path bert-base-uncased     --train_file ../data/output_dynamic.txt   --output_dir ./result    --num_train_epochs 1     --per_device_train_batch_size 32     --learning_rate 3e-5     --max_seq_length 32      --metric_for_best_model stsb_spearman  --load_best_model_at_end     --eval_steps 2     --pooler_type cls     --mlp_only_train     --overwrite_output_dir     --temp 0.05     --do_train
```


## Step 2. Vulnerability Detection

Concoction’s detection component takes the joint embedding as input to predict the presence of vulnerabilities. Our current implementation only identifies whether a function may contain a vulnerability or bug and does not specify the type of vulnerability. Here we use SARD benchmarks.

#### *Vulnerability Detection model training*:

## train
```
$ conda activate pytorch1.7.1
$ cd ./src/concoction/detectionModel
$ python evaluation_bug.py --path_to_data ../data/data/train --mode train
```

## test
```
$ conda activate concoction
$ cd ./src/concoction/detectionModel
$ python evaluation_bug.py --path_to_data ../data/data/test --mode test --model_to_load ./trained_model/github.h5
```



## Step 3. Deployment

This demo shows how to deploy our trained model on a real world project. Here we apply the xx as our test project.

#### *Path Selection for Symbolic Execution*:

After training the end-to-end model, we develop a path selection component to automatically select a subset of important paths whose dynamic traces are likely to improve prediction accuracy during deployment.

*approximate runtime ~ 30 minutes*

## 1.preprocess
```
$ conda activate pytorch1.7.1
$ cd ./src/concoction/pathSelection
$ python preprocess.py --data_path ../data/dataset0 --stored_path ../data/dataset0_pathselect

```
## 2.path select

```
$ python train.py --data_path ../data/dataset0_pathselect --stored_path ../data/dataset0_pathselect_result
```

#### *Fuzzing for Test Case Generation*:

We use fuzzing techniques to generate test cases for functions predicted to contain potential vulnerabilities, aiming to automate the testing process and minimize the need for manual inspection.
