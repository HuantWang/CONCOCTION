# Installation

CONCOCTION was tested with Python 3.6 and Ubuntu 18.04.

Our Docker images are one of the fastest ways to get started.We highly recommend using a Docker environment.

## DOCKER

Install docker engine by following the instructions [here](https://docs.docker.com/install/linux/docker-ce/ubuntu/).

Fetch the docker image from docker hub.

```
$ sudo docker pull concoctionnwu/concoction:v2
```

To check the list of images, run:

```
$ sudo docker images
#output
#REPOSITORY                                                               TAG                                 IMAGE ID       CREATED         SIZE
#concoctionnwu/concoction                                                 v1                                  cc84e8929fe1   15 hours ago    82.4G

```

Run the docker image  in a GPU-enabled environment

```
$ docker run -itd --gpus all  -p 10054:22 -p 10053:8888 --name Concoction concoctionnwu/concoction:v2 /bin/bash
$ docker start Concoction 
$ docker exec -it Concoction /bin/bash
```

## Building from Source

## 1.1. Dependences

### 1.1.1 Toolchain and Python

We recommend using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to manage the remaining build dependencies. First create a conda environment with the required dependencies:

```shell
# add concoction python environment, 
$ conda env create -f environment.yml
```

#### ***Alternative: using different environments to evaluate other SOTA approaches***

|  Approaches   |  Conda name   |
| :-----------: | :-----------: |
|    LineVul    |   sotaEnv1    |
| VulDeepecker  |   sotaEnv2    |
|    Devign     | sotaEnv2 |
| GraphCodeBERT | sotaEnv2 |
|    REVEAL     | sotaEnv2 |
|   CodeXGLUE   | sotaEnv2 |
|    Funded     |    sotaEnv3     |
|    LineVD     |    sotaEnv1    |
|     ReGVD     | sotaEnv2 |
|  ContraFlow   |  pytorch1.7.1 |
|     LIGER     |     sotaEnv4     |
|  Concoction   |  pytorch1.7.1 |

Using following command to control your anaconda version：

```shell
# add different environments to evaluate other SOTA 
$ conda env create -f environment_sotaEnv1.yml
$ conda env create -f environment_sotaEnv3.yml
$ conda env create -f environment_sotaEnv4.yml
$ conda env create -f environment_sotaEnv2.yml
```

## 1.2. KLEE

We use KLEE to extract dynamic information, so we have to build the KLEE first. The version of KLEE we used is **v2.1**. [Download](https://klee.github.io/build-llvm13/) with **LLVM**.
If you just want to quickly get started with the dynamic feature extraction function, it is recommended to use the [container that comes with the KLEE tool](https://klee.github.io/docker/).

## 1.3 JDK11
When we extract static features, we will use **JDK11**.[Download](https://www.oracle.com/cn/java/technologies/javase/jdk11-archive-downloads.html)
## Have a Test

### Test demo
```shell
$ conda activate pytorch1.7.1
$ cd ./concoction/detectionModel
$ python evaluation_bug.py --path_to_data ../data/data/train --mode train 
```
