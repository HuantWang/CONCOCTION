# Combining Static and Dynamic Code Information for Software Vulnerability Prediction: Artifact Instructions for Docker Image

The recommended approach for AE is to use the pre-configured, interactive Jupyter notebook with the instructions given in the AE submission.

The following step-by-step instructions are provided for using a Docker Image running on a local host. Our Docker image (80 GB uncompressed) contains the entire execution environment (including Python and system dependencies), benchmarks, and source code, which includes 16 state-of-the-art bug detection frameworks. All of our code and data are open-sourced and have been developed with extensibility as a primary goal.

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
$ sudo docker pull concoctionnwu/concoction:v3
```

To check the list of images, run:

```
$ sudo docker images
#output
#REPOSITORY                                                               TAG                                 IMAGE ID       CREATED         SIZE
#concoctionnwu/concoction                                                 v2                                  cc84e8929fe1   15 hours ago    82.4GB

```

Run the docker image in a GPU-enabled environment

```
$ docker run -itd --gpus all  -p 10054:22 -p 10053:8888 --name Concoction concoctionnwu/concoction:v2 /bin/bash
$ docker start Concoction 
$ docker exec -it Concoction /bin/bash
```


#### 1.2 Setup the Environment

After importing the docker container **and getting into bash** in the container, run the following command to select the conda environment, before using any of the AE scripts:

`````` shell
$ conda activate base
``````

Then, go to the root directory of our tool:

```
(docker) $ cd /homee/
```

# The Concoction Tutorial 

## Step 1. Program representation

The program representation component maps the input source code and dynamic symbolic execution traces of the target function into a numerical embedding vector.

### *Static representation model*

#### Extract static information

```
# Execute the script to extract static information, 
# passing the program path (path) and the script path (ScriptPath) as arguments
# bash [ScriptPath]  [path]
$ bash /homee/Evaluation/demo1/getStatic.sh /homee/Evaluation/exampleProject/test1
```

#### Training

```
# Execute the script for pretraining reoresentation model
# [--dataset] dataset path for training 
$ python /homee/Evaluation/demo1/concoction_1.py --dataset /homee/concoction/data/dataset
```

#### Using trained model to represent programs

```
# Execute the script with a trained model to represent programs
# [--program] program you can choose with test_a.c,test_b.c or test_c.c
$ python /homee/Evaluation/demo1/concoction_2.py --program test_a.c
```

### *Dynamic representation model*

#### Extract dynamic information

```
# Execute the script to extract dynamic information, 
# bash [ScriptPath]  [project_path] [Description of Dependencies to Install] [Compilation Instructions for the Project] [Execution Instructions for the Project]
$ cd /homee/feature/dynamic
$ bash main.sh /homee/Evaluation/exampleProject/jasper-version-1.900.1 ./before_insert.txt ./compile.txt ./do.txt
```

#### Training

```
# Execute the script for pretraining reoresentation model
# [--dataset] dataset path for training 
$ python /homee/Evaluation/demo1/concoction_3.py --dataset /homee/concoction/data/dataset
```

#### Using trained model to represent programs dynamic information

```
# Execute the script with a trained model to represent programs
# [--program] program you can choose with test_a.c,test_b.c or test_c.c
$ python /homee/Evaluation/demo1/concoction_4.py --program test_a.c
```


## Step 2. Vulnerability Detection

Concoction’s detection component takes the joint embedding as input to predict the presence of vulnerabilities. Our current implementation only identifies whether a function may contain a vulnerability or bug and does not specify the type of vulnerability. Here we use SARD benchmarks.

### *Vulnerability Detection model training*:

#### Train

```
# Execute the script for training the detection model
# [--dataset] dataset path for training [--epochs] training epoch [--batch_size] training batch_size 
$ python /homee/Evaluation/demo1/concoction_5.py  --dataset /homee/Evaluation/ExperimentalEvaluation/data/github_0.6_new/train --epochs 200 --batch_size 64

```

#### Testing

```
# Execute the script  for testing the detection model
# [--dataset] dataset path for testing [--model_to_load] model loaded to test 
$ python /homee/Evaluation/demo1/concoction_6.py  --dataset /homee/Evaluation/ExperimentalEvaluation/data/github_0.6_new/test --model_to_load /homee/Evaluation/ExperimentalEvaluation/Concoction/saved_models/github.h5
```

## Step 3. Deployment

This demo shows how to deploy our trained model on a real world project. Here we apply the jasper as our test project.

#### *Path Selection for Symbolic Execution*:

After training the end-to-end model, we develop a path selection component to automatically select a subset of important paths whose dynamic traces are likely to improve prediction accuracy during deployment.

```
# Path collection and Active learning for path selection
$ python /homee/Evaluation/demo1/concoction.py
```

#### Vulnerability detection

```
# Vulnerability detection
# [--dataset] dataset path for testing [--model_to_load] model loaded to test
$ python /homee/Evaluation/demo1/concoction_7.py  --dataset /homee/concoction/data/dataset0 --model_to_load /homee/Evaluation/ExperimentalEvaluation/Concoction/saved_models/github.h5
```





# ***Reference***

| Source code                                                  | Title                                                        | Authors                                                   |
| :----------------------------------------------------------- | :----------------------------------------------------------- | :-------------------------------------------------------- |
| [Funded](https://github.com/HuantWang/FUNDED_NISL)[1]        | Combining Graph-Based Learning With Automated Data Collection for Code Vulnerability Detection | H. Wang, G. Ye, Z. Tang, et al.                           |
| [Vuldeepecker](https://github.com/CGCL-codes/VulDeePecker)[2] | VulDeePecker: A Deep Learning-Based System for Vulnerability Detection | Zhen Li, Deqing Zou, Shouhuai Xu, et al.                  |
| [ReVeal](https://github.com/VulDetProject/ReVeal)[3]         | Deep learning based vulnerability detection: Are we there yet | Saikat Chakraborty, Rahul Krishna, Yangruibo Ding, et al. |
| [Devign](https://github.com/epicosy/devign)[4]               | Devign: Effective vulnerability identification by learning comprehensive program semantics via graph neural networks | Yaqin Zhou, Shangqing Liu, Jingkai Siow, et al.           |
| [ReGVD](https://github.com/daiquocnguyen/GNN-ReGVD)[5]       | ReGVD: Revisiting Graph Neural Networks for Vulnerability Detection | Van-Anh Nguyen, Dai Quoc Nguyen, Van Nguyen, et al.       |
| [LineVul](https://github.com/awsm-research/LineVul)[6]       | Linevul: A transformer-based line-level vulnerability prediction | Michael Fu and Chakkrit Tantithamthavorn.                 |
| [LineVD](https://github.com/davidhin/linevd)[7]              | LineVD: Statement-level vulnerability detection using graph neural networks | David Hin, Andrey Kan, Huaming Chen, et al.               |
| [CodeXGLUE](https://github.com/microsoft/CodeXGLUE)[8]       | CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation | Shuai Lu, Daya Guo, Shuo Ren, et al.                      |
| [GraphcodeBERT](https://github.com/microsoft/CodeBERT/tree/master/GraphCodeBERT)[9] | GraphCodeBERT: Pre-training Code Representations with Data Flow | Daya Guo, Shuo Ren, Shuai Lu, et al.                      |
| [Contraflow](https://dl.acm.org/doi/10.1145/3533767.3534371)[10] | Path-Sensitive Code Embedding via Contrastive Learning for Software Vulnerability Detection | Xiao Cheng, Guanqin Zhang, Haoyu Wang, et al.             |
| [LIGER](https://github.com/keowang/dynamic-program-embedding)[11] | Blended, Precise Semantic Program Embeddings                 | Ke Wang and Zhendong Su.                                  |
| [CodeQL](https://github.com/github/codeql)[12]               | CodeQL, discover vulnerabilities with semantic code analysis engine | /                                                         |
| [Infer](https://github.com/facebook/infer)[13]               | Infer, a static program analyzer                             | /                                                         |
| [KLEE](https://klee.github.io/)[14]                          | Klee: unassisted and automatic generation of high-coverage tests for complex systems programs | Cristian Cadar, Daniel Dunbar, Dawson R Engler, et al.    |
| [MoKLEE](https://srg.doc.ic.ac.uk/projects/moklee/)[15]      | Running symbolic execution forever                           | Frank Busse, Martin Nowack, and Cristian Cadar.           |
| [AFL++](https://github.com/AFLplusplus/AFLplusplus)[16]      | AFL++:Combining Incremental Steps of Fuzzing Research        | Andrea Fioraldi, Dominik Maier, Heiko Eißfeldt, et al.    |

#### 