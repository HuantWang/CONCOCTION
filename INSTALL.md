# Installation

CONCOCTION was tested with Python 3.6 and Ubuntu 18.04.

Our Docker images are one of the fastest ways to get started.We highly recommend using a Docker environment.

## DOCKER

Install docker engine by following the instructions [here](https://docs.docker.com/install/linux/docker-ce/ubuntu/).

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
|    LineVul    |    LineVul    |
| VulDeepecker  | vuldeepecker1 |
|    Devign     | vuldeepecker1 |
| GraphCodeBERT | vuldeepecker1 |
|    REVEAL     | vuldeepecker1 |
|   CodeXGLUE   | vuldeepecker1 |
|    Funded     |    funded     |
|    LineVD     |    Linevul    |
|     ReGVD     | vuldeepecker1 |
|  ContraFlow   |  pytorch1.7.1 |
|     LIGER     |     LIGER     |
|  Concoction   |  pytorch1.7.1 |

Using following command to control your anaconda version：

```shell
# add different environments to evaluate other SOTA 
$ conda env create -f environment_linevul.yml
$ conda env create -f environment_funded.yml
$ conda env create -f environment_liger.yml
$ conda env create -f environment_vuldeepecker1.yml
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
