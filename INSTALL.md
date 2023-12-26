# Installation

CONCOCTION was tested with Python 3.6 and Ubuntu 18.04.

## DOCKER

Install docker engine by following the instructions [here](https://docs.docker.com/install/linux/docker-ce/ubuntu/).

1. Fetch the docker image from docker hub.

```
$ sudo docker pull nwussimage/concoction_0.1
```

To check the list of images, run:

```
$ sudo docker images
REPOSITORY                                   TAG                 IMAGE ID            CREATED             SIZE
nwussimage/concoction_0.1		     latest              ac6b624d06de        2 hours ago         41.8GB
```

1. Run the docker image.

```
$ docker run -dit -P --name=supersonic nwussimage/concoction_0.1 /bin/bash
$ docker start concoction 
$ docker exec -it concoction /bin/bash
```

## Building from Source

## 1.1. Dependences

### 1.1.1 Toolchain and Python

We recommend using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to manage the remaining build dependencies. First create a conda environment with the required dependencies:

```shell
# add python environment, 
$ conda create -y -n concoction python=3.8
$ pip install -r requirements.txt
$ conda activate concoction
```

#### ***Alternative: using different environments to evaluate other SOTA approaches***

|  Approaches   | Conda name |
| :-----------: | :--------: |
|    LineVul    |            |
| VulDeepecker  |            |
|    Devign     |            |
| GraphCodeBERT |            |
|    REVEAL     |            |
|   CodeXGLUE   |            |
|    Funded     |            |
|    LineVD     |            |
|     ReGVD     |            |
|  ContraFlow   |            |
|     LIGER     |            |
|  Concoction   |            |

Using following command to control your anaconda version：

```shell
# add python environment
$ conda create -y -n concoction_1 python=3.8
$ pip install -r requirements_1.txt
$ conda activate concoction
```

## 1.2. KLEE

We use KLEE to extract dynamic information, so we have to build the KLEE first. The version of KLEE we used is **v3.0**. [Download]<https://klee.github.io/build-llvm13/> with **LLVM**.
If you just want to quickly get started with the dynamic feature extraction function, it is recommended to use the [container that comes with the KLEE tool](https://klee.github.io/docker/).

## Have a Test

### Test demo

```shell 
$ python supersonic_main.py --task CSR  --mode test
```
