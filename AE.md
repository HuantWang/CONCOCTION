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
# passing the program path (path) and the script path (ScriptPath) as arguments
# bash [ScriptPath]  [path]
$ bash /homee/Evaluation/demo1/getDynamic.sh /homee/Evaluation/exampleProject/jasper-version-1.900.1
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



# ***Data***
This is our introduction to the training and validation dataset used in the paper
You can download the dataset from [here](https://drive.google.com/file/d/1ObjOKKWl0cS81ZF5KPMoSPprThVav5aA/view?usp=drive_link)

## Open datasets used in training and evaluation
| Source |    Projects     |       Versions       | Samples | Vulnerability samples |
| ------ | :-------------: | :------------------: | :-----: | :-------------------: |
| SARD   |        /        |          /           | 30,954  |         5,477         |
| Github |     Jasper      |  v1.900.1-5,v2.0.12  | 24,996  |          666          |
|        |     Libtiff     |       v4.0.3-9       |  6,336  |          724          |
|        |     Libzip      |     v0.10,v1.2.0     |  5,686  |          66           |
|        |     Libyaml     |        v0.1.4        | 27,625  |          42           |
|        |     Sqlite      |        v3.8.2        |  1,825  |          31           |
|        | Ok-file-formats |       203defd        |  1,014  |          17           |
|        |     libpng      | v1.2.7,v1.5.4,v1.6.0 |   954   |          12           |
|        |     libming     |       v0.4.7-8       |  1,104  |          16           |
|        |    libexpat     |        v2.0.1        |  1,051  |          14           |

This folder contains all the datasets used in our paper.

`github:` This folder contains more than 68K C functions from 9 large C-language open-source projects.

`sard:` This folder contains more than 30K C functions from the SARD standard vulnerability dataset.

## Data structure

All the data is stored in `.zip` files. After decompression, you will find `.txt` files, 
each of which represents a C function feature file.
Each feature file(eg.2ok_jpg.c-ok_jpg_convert_data_unit_grayscale.c.txt) 
includes `static features` (AST,CFG,DFG and other seven edges) and `dynamic features` (input variable values and execution traces).

#### Description of text example
|          Items              |        Labels        |                Values               |
|:---------------------------:|:--------------------:|:----------------------------------:|
| Vulnerability or not        | -----label-----      |                0/1                 |
| Source code                 | -----code-----       | static void ok_jpg_convert_d...    |
| Code relationship flow edges| -----children-----   | 1,2<br/>1,3<br/>...<br />1,4       |
| Code relationship flow edges| -----nextToken-----  | 2,4,7,9,10,13,15,                  |
| Code relationship flow edges| -----computeFrom-----| 42,43<br/>42,44<br/>69,70<br />...  |
| Code relationship flow edges| -----guardedBy-----  | 90,92<br/>101,102<br/>101,103<br/>...|
| Code relationship flow edges| -----guardedByNegation----- | 124,125<br/>125,126<br/>125,127<br />... |
| Code relationship flow edges| -----lastLexicalUse----- | 42,44<br/>43,44<br/>47,48<br />... |
| Code relationship flow edges| -----jump-----       | 21,22<br/>21,23<br/>23,24<br />...  |
| Node tokens                 | -----ast_node-----   | const uint8_t *y<br/>const uint8_t<br/>uint8_t<br />... |
| ...                         | ...                  | ...                               |
| Input variable values       | =======testcase======== | y_inc:0x00000000<br/>x_inc:0x00000000<br />... |
| Execution traces            | =========trace========= | for(int x = 0;x < max_width;x++)<br/>out[0] = y[x];<br/>out[1] = y[x];<br />... |


# ***Vulnerability Info***
Code vulnerabilities discovered by Concoction
You can download the project source code from [here](https://drive.google.com/file/d/146fkOLlfjxBxh_bSXwFu0HLmWb61A8ji/view?usp=drive_link)

## Bug statistics for each tested project

| NO.  | Project                                          | Description                                                  | Stars | Release Data | The number of vulnerabilities we found |
| ---- | ------------------------------------------------ | ------------------------------------------------------------ | ----- | ------------------------------ | ---------------------------------- |
| 1    | [ok-file-formats](#1)                            | C functions for reading a few different file formats.        | 100   | Sep.2021                       | 7                                  |
| 2    | [OpenEXR](#2)                                    | OpenEXR is a high dynamic-range (HDR) image                  | 1.3k  | Nov.2022                       | 1                                  |
| 3    | [ImageMagick](#3)                                | Use [ImageMagick®](https://imagemagick.org/) to create, edit, compose, or convert bitmap images. It can read and write images in a variety of formats (over 200) Use ImageMagick to resize, flip, mirror, rotate, distort, shear and transform images, adjust image colors, apply various special effects, or draw text, lines, polygons, ellipses and Bézier curves. | 8.8k  | Apr.2023                       | 1                                  |
| 4    | [openjpeg](#4)                                   | OpenJPEG is an open-source JPEG 2000 codec written in C language. | 1.4k  | Apr.2023                       | 1                                  |
| 5    | [packJPG](#5)                                    | packJPG is a compression program specially designed for further compression of JPEG images without causing any further loss. Typically it reduces the file size of a JPEG file by 20%. | 144   | Apr.2020                       | 7                                  |
| 6    | [mediancut-posterizer](#6)                       | The goal of this tool is to make RGB/RGBA PNG images more compressible, assuming that lower number of unique byte values increses chance of finding repetition and improves efficiency of Huffman coding. | 228   | Feb.2023                       | 1                                  |
| 7    | [astc-encoder](#7)                               | a command-line tool for compressing and decompressing images using the ASTC texture compression standard. | 780   | Apr.2023                       | 3                                  |
| 8    | [assimp](#8)                                     | A library to import and export various 3d-model-formats including scene-post-processing to generate missing render data. | 8.9k  | Apr.2023                       | 6                                  |
| 9    | [epub2txt2](#9)                                  | `epub2txt` is a simple command-line utility for extracting text from EPUB documents and, optionally, re-flowing it to fit a text display of a particular number of columns. | 129   | Jun.2022                       | 1                                  |
| 10   | [Leanify](#11)                                   | Leanify is a lightweight lossless file minifier/optimizer. It removes unnecessary data (debug information, comments, metadata, etc.) and recompress the file to reduce file size. | 767   | 2022.06.09                     | 2                                  |
| 11   | [AudioFile](#12)                                 | A simple header-only C++ library for reading and writing audio files. | 752   | Apr.2023                       | 1                                  |
| 12   | [zydis](#13)                                     | Fast and lightweight x86/x86-64 disassembler and code generation library. | 2.7k  | Apr.2023                       | 4                                  |
| 13   | [lepton](#14)                                    | Lepton is a tool and file format for losslessly compressing JPEGs by an average of 22%.This can be used to archive large photo collections, or to serve images live and save 22% bandwidth. | 5k    | Feb.2023                       | 1                                  |
| 14   | [pdftojson](#15)                                 | using XPDF, pdftojson extracts text from PDF files as JSON, including word bounding boxes. | 137   | Oct.2017                       | 3                                  |
| 15   | [xlsxio](#16)                                    | Cross-platform C library for reading values from and writing values to .xlsx files. | 268   | 2022.07.05                     | 1                                  |
| 17   | [ELFLoader](#17)                                 | This is a ELF object in memory loader/runner. The goal is to create a single elf loader that can be used to run follow on capabilities across all x86_64 and x86 nix operating systems. | 180   | 2022.05.17                     | 4                                  |
| 18   | [deark](#18)                                     | Deark is a command-line utility that can decode certain types of files, and either:convert them to a more-modern or more-readable format; or extract embedded files from them | 125   | Apr.2023                       | 3                                  |
| 19   | [Kernel](#19)                                    | Linux is a clone of the operating system Unix. It aims towards POSIX and Single UNIX Specification compliance. |    141K   | Apr.2023                               | 2                                  |
| 20   | [sqlcheck](https://github.com/jarulraj/sqlcheck) | `sqlcheck` automatically detects common SQL anti-patterns.  Such anti-patterns often slow down queries. Addressing them will,  therefore, help accelerate queries. | 2.3k  | Mar.2022                       | 4                                  |



## Reported category for each group

| Reported Category      | Submitted | Confirmed | Fixed  | Dyn-related |
| ---------------------- | --------- | --------- | ------ | ----------- |
| buffer-overflow        | 33        | 33        | 20     | 23          |
| segmentation-violation | 6         | 6         | 1      | 5           |
| memory-leaks           | 4         | 4         | 1      | 3           |
| other types            | 10        | 10        | 4      | 5           |
| **Total**              | **53**    | **52**    | **26** | **36**      |



## Detailed information of vulnerabilities

### <span id="1">1.[ok-file-formats](https://github.com/brackeen/ok-file-formats)</span>

| NO.  | Version                                                      | <span style="display:inline-block;width: 100px"> Issue link </span> | <span style="display:inline-block;width: 120px"> Poc</span>  | <span style="display:inline-block;width: 100px">IssueState</span> | <span style="display:inline-block;width: 100px"> CVEState</span> |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | [01be744](https://github.com/brackeen/ok-file-formats/commit/01be744dbb62672e3df283b73541fca228546da8) | [heap-buffer-overflow](https://github.com/brackeen/ok-file-formats/issues/11) | [heap-buffer-overflow-1](https://github.com/NISL-SecurityGroup/NISL-BugDetection/blob/main/project/ok-file-formats/heap-buffer-overflow-1/heap-buffer-overflow-1.jpg) | Fixed                                                        | [CVE-2021-44340](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2021-44340) |
| 2    | [97f78ca](https://github.com/brackeen/ok-file-formats/commit/97f78ca229067f23bca4bc3f303147bd09da7d79) | [heap-buffer-overflow](https://github.com/brackeen/ok-file-formats/issues/12) | [heap-buffer-overflow-2](https://github.com/NISL-SecurityGroup/NISL-BugDetection/blob/main/project/ok-file-formats/heap-buffer-overflow-2/heap-buffer-overflow-2.jpg) | Fixed                                                        | [CVE-2021-44334](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2021-44334) |
| 3    | [203defd](https://github.com/brackeen/ok-file-formats/commit/203defdfb2c8b1207a392493a09145c1b54bb070) | [heap-buffer-overflow](https://github.com/brackeen/ok-file-formats/issues/15) | [heap-buffer-overflow-3](https://github.com/nisl-bugTest/Pocfiles/blob/main/fuzzing/ok-file-formats/heap-buffer-overflow-3/poc) | Fixed                                                        | [CVE-2021-44339](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2021-44339) |
| 4    | [203defd](https://github.com/brackeen/ok-file-formats/commit/203defdfb2c8b1207a392493a09145c1b54bb070) | [heap-buffer-overflow](https://github.com/brackeen/ok-file-formats/issues/16) | [heap-buffer-overflow-4](https://github.com/nisl-bugTest/Pocfiles/blob/main/fuzzing/ok-file-formats/heap-buffer-overflow-4/poc) | Fixed                                                        | [CVE-2021-44339](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2021-44339) |
| 5    | [203defd](https://github.com/brackeen/ok-file-formats/commit/203defdfb2c8b1207a392493a09145c1b54bb070) | [heap-buffer-overflow](https://github.com/brackeen/ok-file-formats/issues/17) | [heap-buffer-overflow-5](https://github.com/nisl-bugTest/Pocfiles/blob/main/fuzzing/ok-file-formats/heap-buffer-overflow-5/poc) | Fixed                                                        | [CVE-2021-44335](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2021-44335) |
| 6    | [203defd](https://github.com/brackeen/ok-file-formats/commit/203defdfb2c8b1207a392493a09145c1b54bb070) | [heap-buffer-overflow](https://github.com/brackeen/ok-file-formats/issues/18) | [heap-buffer-overflow-6](https://github.com/nisl-bugTest/Pocfiles/blob/main/fuzzing/ok-file-formats/heap-buffer-overflow-6/poc) | Fixed                                                        | [CVE-2021-44343](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2021-44343) |
| 7    | [203defd](https://github.com/brackeen/ok-file-formats/commit/203defdfb2c8b1207a392493a09145c1b54bb070) | [heap-buffer-overflow](https://github.com/brackeen/ok-file-formats/issues/19) | [heap-buffer-overflow-7](https://github.com/nisl-bugTest/Pocfiles/blob/main/fuzzing/ok-file-formats/heap-buffer-overflow-7/poc) | Fixed                                                        | [CVE-2021-44342](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2021-44342) |

### <span id="2">2.[OpenEXR](https://github.com/AcademySoftwareFoundation/openexr)</span>

| NO.  | Version                                                      | <span style="display:inline-block;width: 100px"> Issue link </span> | <span style="display:inline-block;width: 120px"> Poc</span>  | <span style="display:inline-block;width: 100px">IssueState</span> | <span style="display:inline-block;width: 100px"> CVEState</span> |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | [2.2.0](https://github.com/AcademySoftwareFoundation/openexr/tree/v2.2.0) | [allocation-size-too-big](https://github.com/AcademySoftwareFoundation/openexr/issues/996) | [allocation-size-too-big](https://github.com/NISL-SecurityGroup/NISL-BugDetection/blob/main/project/OpenEXR/allocation-size-too-big/allocation-size-too-big) | Fixed                                                        | [CVE-2017-14988](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2017-14988) |

### <span id="3">3.[ImageMagick](https://github.com/ImageMagick/ImageMagick)</span>

| NO.  | Version                                                      | <span style="display:inline-block;width: 100px"> Issue link </span> | <span style="display:inline-block;width: 120px"> Poc</span>  | <span style="display:inline-block;width: 100px">IssueState</span> | <span style="display:inline-block;width: 100px"> CVEState</span> |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | [7.0.11-5](https://github.com/ImageMagick/ImageMagick/tree/7.0.11-5) | [memory_leaks](https://github.com/ImageMagick/ImageMagick/issues/3540) | [memory_leaks](https://github.com/NISL-SecurityGroup/NISL-BugDetection/blob/main/project/ImageMagick/memory_leaks/memory_leaks) | Fixed                                                        | [CVE-2021-3574](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2021-3574) |

### <span id="4">4.[openjpeg](https://github.com/uclouvain/openjpeg)</span>

| NO.  | Version                                                      | <span style="display:inline-block;width: 100px"> Issue link </span> | <span style="display:inline-block;width: 120px"> Poc</span>  | <span style="display:inline-block;width: 100px">IssueState</span> | <span style="display:inline-block;width: 100px"> CVEState</span> |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | [a44547d](https://github.com/google/oss-fuzz/commit/a44547d8d6f78ad7ce02323ecc33382a1d628e39) | [heap-buffer-overflow](https://github.com/uclouvain/openjpeg/issues/1363) | [heap-buffer-overflow-1](https://github.com/NISL-SecurityGroup/NISL-BugDetection/blob/main/project/openjpeg/heap-buffer-overflow/heap-buffer-overflow-1) | Fixed                                                        | [CVE-2021-3575](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2021-3575) |

### <span id="5">5.[packJPG](https://github.com/packjpg/packJPG)</span>


| NO.  | Version                                               | <span style="display:inline-block;width: 100px"> Issue link </span> | <span style="display:inline-block;width: 120px"> Poc</span>  | <span style="display:inline-block;width: 100px">IssueState</span> | <span style="display:inline-block;width: 100px"> CVEState</span> |
| ---- | ----------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | [v2.5k](https://github.com/packjpg/packJPG/tree/2.5k) | [memory_leaks                ](https://github.com/packjpg/packJPG/issues/29) | [memory_leaks    ](https://github.com/nisl-bugTest/Pocfiles/tree/main/fuzzing/packJPG/memory_leaks) | Pending                                                      | CVE-2022-45738                                               |
| 2    | [v2.5k](https://github.com/packjpg/packJPG/tree/2.5k) | [global-buffer-overflow](https://github.com/packjpg/packJPG/issues/30) | [global-buffer-overflow](https://github.com/nisl-bugTest/Pocfiles/tree/main/fuzzing/packJPG/global-buffer-overfllow) | Pending                                                      | CVE-2022-45740                                               |
| 3    | [v2.5k](https://github.com/packjpg/packJPG/tree/2.5k) | [alloc-dealloc-mismatch](https://github.com/packjpg/packJPG/issues/31) | [alloc-dealloc-mismatch](https://github.com/nisl-bugTest/Pocfiles/tree/main/fuzzing/packJPG/alloc-dealloc-mismatch) | Pending                                                      | CVE-2022-45739                                               |
| 4    | [v2.5k](https://github.com/packjpg/packJPG/tree/2.5k) | [global-buffer-overflow](https://github.com/packjpg/packJPG/issues/32) | [global-buffer-overflow](https://github.com/nisl-bugTest/Pocfiles/tree/main/fuzzing/packJPG/global-buffer-overflow-2) | Pending                                                      | CVE-2022-45742                                               |
| 5    | [v2.5k](https://github.com/packjpg/packJPG/tree/2.5k) | [heap-buffer-overflow](https://github.com/packjpg/packJPG/issues/33) | [heap-buffer-overflow](https://github.com/nisl-bugTest/Pocfiles/tree/main/fuzzing/packJPG/heap-buffer-overflow) | Pending                                                      | CVE-2022-45741                                               |
| 6    | [v2.5k](https://github.com/packjpg/packJPG/tree/2.5k) | [memory leaks](https://github.com/packjpg/packJPG/issues/34) | [memory leaks](https://github.com/nisl-bugTest/Pocfiles/tree/main/fuzzing/packJPG/memory_leaks-2) | Pending                                                      | CVE-2022-45737                                               |
| 7    | [v2.5k](https://github.com/packjpg/packJPG/tree/2.5k) | [SEGV](https://github.com/packjpg/packJPG/issues/35)         | [SEGV](https://github.com/nisl-bugTest/Pocfiles/tree/main/fuzzing/packJPG/SEGV) | Pending                                                      | CVE-2022-45736                                               |

### <span id="6">6.[mediancut-posterizer](https://github.com/kornelski/mediancut-posterizer)</span>

| NO.  | Version                                                      | <span style="display:inline-block;width: 100px"> Issue link </span> | <span style="display:inline-block;width: 120px"> Poc</span>  | <span style="display:inline-block;width: 100px">IssueState</span> | <span style="display:inline-block;width: 100px"> CVEState</span> |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | [2.1](https://github.com/kornelski/mediancut-posterizer/tree/2.1) | [SEGV](https://github.com/kornelski/mediancut-posterizer/issues/15) | [SEGV](https://github.com/nisl-bugTest/Pocfiles/blob/main/fuzzing/mediancut-posterizer/SEGV) | Confirmed                                                    | [CVE-2021-44333](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2021-44333) |

### <span id="7">7.[astc-encoder](https://github.com/ARM-software/astc-encoder)</span>


| NO.  | Version                                                      | <span style="display:inline-block;width: 100px"> Issue link </span> | <span style="display:inline-block;width: 120px"> Poc</span>  | <span style="display:inline-block;width: 100px">IssueState</span> | <span style="display:inline-block;width: 100px"> CVEState</span> |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | [v3.2k](https://github.com/ARM-software/astc-encoder/tree/3.2) | [stack-buffer-overflow](https://github.com/ARM-software/astc-encoder/issues/294) | [stack-buffer-overflow](https://github.com/nisl-bugTest/Pocfiles/blob/main/fuzzing/astc-encoder/stack-buffer-overflow-1/crash01.png) | Fixed                                                        | [CVE-2021-44331](https://cve.mitre.org/cgi-bin/cvename.cgi?name=2021-44331) |
| 2    | [v3.2k](https://github.com/ARM-software/astc-encoder/tree/3.2) | [stack-buffer-overflow](https://github.com/ARM-software/astc-encoder/issues/295) | [stack-buffer-overflow](https://github.com/nisl-bugTest/Pocfiles/blob/main/fuzzing/astc-encoder/stack-buffer-overflow-2/crash02) | Fixed                                                        | [CVE-2021-43086](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2021-43086) |
| 3    | [v3.2k](https://github.com/ARM-software/astc-encoder/tree/3.2) | [stack-buffer-overflow](https://github.com/ARM-software/astc-encoder/issues/296) | [stack-buffer-overflow](https://github.com/nisl-bugTest/Pocfiles/blob/main/fuzzing/astc-encoder/stack-buffer-overflow-3/crash04) | Fixed                                                        | [CVE-2021-43086](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2021-43086) |

### <span id="8">8.[assimp](https://github.com/assimp/assimp)</span>

| NO.  | Version                                                      | <span style="display:inline-block;width: 100px"> Issue link </span> | <span style="display:inline-block;width: 120px"> Poc</span>  | <span style="display:inline-block;width: 100px">IssueState</span> | <span style="display:inline-block;width: 100px"> CVEState</span> |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | [5.1.4](https://github.com/assimp/assimp/releases/tag/v5.1.4) | [heap-buffer-overflow](https://github.com/assimp/assimp/issues/4285) | [heap-buffer-overflow](https://github.com/NISL-SecurityGroup/NISL-BugDetection/blob/main/project/assimp/heap-buffer-overflow-1/Assimp__ColladaLoader__CreateMesh_heap-buffer-overflow) | Confirmed                                                    | CVE-2022-45744                                               |
| 2    | [5.1.4](https://github.com/assimp/assimp/releases/tag/v5.1.4) | [heap-use-after-free](https://github.com/assimp/assimp/issues/4286) | [heap-use-after-free](https://github.com/NISL-SecurityGroup/NISL-BugDetection/blob/main/project/assimp/heap-use-after-free/Assimp__ColladaParser__ExtractDataObjectFromChannel_heap-use-after-free) | Confirmed                                                    | CVE-2022-45747                                               |
| 3    | [5.1.4](https://github.com/assimp/assimp/releases/tag/v5.1.4) | [SEGV](https://github.com/assimp/assimp/issues/4287)         | [SEGV](https://github.com/NISL-SecurityGroup/NISL-BugDetection/blob/main/project/assimp/SEGV-1/Assimp__ColladaParser__ReadPrimitives_SEGV) | Confirmed                                                    | CVE-2022-45745                                               |
| 4    | [5.1.4](https://github.com/assimp/assimp/releases/tag/v5.1.4) | [SEGV](https://github.com/assimp/assimp/issues/4288)         | [SEGV](https://github.com/NISL-SecurityGroup/NISL-BugDetection/blob/main/project/assimp/SEGV-2/Assimp__MDLImporter__InternReadFile_3DGS_MDL345_SEGV) | Confirmed                                                    | CVE-2022-45743                                               |
| 5    | [5.1.4](https://github.com/assimp/assimp/releases/tag/v5.1.4) | [memcpy-param-overlap](https://github.com/assimp/assimp/issues/4289) | [memcpy-param-overlap](https://github.com/NISL-SecurityGroup/NISL-BugDetection/blob/main/project/assimp/memcpy-param-ocverlap/memcpy-param-overlap) | Confirmed                                                    | CVE-2022-45748                                               |
| 6    | [5.1.4](https://github.com/assimp/assimp/releases/tag/v5.1.4) | [heap-buffer-overflow](https://github.com/assimp/assimp/issues/4290) | [heap-buffer-overflow](https://github.com/NISL-SecurityGroup/NISL-BugDetection/blob/main/project/assimp/heap-buffer-overflow-2/std__vector_aiVertexWeight%2C%20std__allocator_aiVertexWeight_%20___push_back_heap-buffer-overflow) | Confirmed                                                    | CVE-2022-45746                                               |

### <span id="9">9. [epub2txt2](https://github.com/kevinboone/epub2txt2)     </span>

| NO.  | Version                                                      | <span style="display:inline-block;width: 100px"> Issue link </span> | <span style="display:inline-block;width: 120px"> Poc</span>  | <span style="display:inline-block;width: 100px">IssueState</span> | <span style="display:inline-block;width: 100px"> CVEState</span> |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | [71dc41](https://github.com/kevinboone/epub2txt2/commit/71dc4199e8715d7b8b44b24b83247f199ce0a0a2) | [stack-buffer-overflow](https://github.com/kevinboone/epub2txt2/issues/17) | [stack-buffer-overflow](https://github.com/NISL-SecurityGroup/NISL-BugDetection/blob/main/project/epub2txt2/epub2txt.poc) | Fixed                                                        | [CVE-2022-23850](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2022-23850) |

### <span id="11">10. [Leanify](https://github.com/JayXon/Leanify)</span>

| NO.  | Version                                                      | <span style="display:inline-block;width: 100px"> Issue link </span> | <span style="display:inline-block;width: 120px"> Poc</span>  | <span style="display:inline-block;width: 100px">IssueState</span> | <span style="display:inline-block;width: 100px"> CVEState</span> |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | [b5f2efc](https://github.com/JayXon/Leanify/commit/b5f2efccde6f57c4dfecc2006e75444621763fd4) | [heap-buffer-overflow](https://github.com/JayXon/Leanify/issues/80) | [heap-buffer-overflow](https://github.com/NISL-SecurityGroup/NISL-BugDetection/blob/main/project/Leanify/leanify_poc) | Fixed                                                        | -                                                            |
| 2    | [b5f2efc](https://github.com/JayXon/Leanify/commit/b5f2efccde6f57c4dfecc2006e75444621763fd4) | [out-of-memory](https://github.com/JayXon/Leanify/issues/82) | [out-of-memory](https://github.com/NISL-SecurityGroup/NISL-BugDetection/blob/main/project/Leanify/leanify_poc_2) | Fixed                                                        | -                                                            |

### <span id="12">11. [AudioFile](https://github.com/adamstark/AudioFile)</span>

| NO.  | Version                                                      | <span style="display:inline-block;width: 100px"> Issue link </span> | <span style="display:inline-block;width: 120px"> Poc</span>  | <span style="display:inline-block;width: 100px">IssueState</span> | <span style="display:inline-block;width: 100px"> CVEState</span> |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | [004065d](https://github.com/adamstark/AudioFile/commit/004065d01e9b7338580390d4fdbfbaa46adede4e) | [heap-buffer-overflow](https://github.com/adamstark/AudioFile/issues/58) | [heap-buffer-overflow](https://github.com/NISL-SecurityGroup/NISL-BugDetection/blob/main/project/Audiofile/poc3) | Fixed                                                        | [CVE-2022-25023](https://cve.mitre.org/cgi-bin/cvename.cgi?name=2022-25023) |

### <span id="13">12. [zydis](https://github.com/zyantific/zydis)</span>


| NO.  | Version                                                      | <span style="display:inline-block;width: 100px"> Issue link </span> | <span style="display:inline-block;width: 120px"> Poc</span>  | <span style="display:inline-block;width: 100px">IssueState</span> | <span style="display:inline-block;width: 100px"> CVEState</span> |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | [770c320](https://github.com/zyantific/zydis/commit/770c3203ef81040af892e3ae6ca42252edbab43c) | [stack-buffer-overflow](https://github.com/zyantific/zydis/issues/315) | [stack-buffer-overflow](https://github.com/nisl-bugTest/Pocfiles/blob/main/fuzzing/zydis/main) | Fixed                                                        | -                                                            |
| 2    | [770c320](https://github.com/zyantific/zydis/commit/770c3203ef81040af892e3ae6ca42252edbab43c) | [stack-buffer-overflow](https://github.com/zyantific/zydis/issues/316) | [stack-buffer-overflow](https://github.com/nisl-bugTest/Pocfiles/blob/main/fuzzing/zydis/zydisinputnext) | Fixed                                                        | -                                                            |
| 3    | [770c320](https://github.com/zyantific/zydis/commit/770c3203ef81040af892e3ae6ca42252edbab43c) | [stack-buffer-overflow](https://github.com/zyantific/zydis/issues/317) | [stack-buffer-overflow](https://github.com/nisl-bugTest/Pocfiles/blob/main/fuzzing/zydis/interceptor) | Fixed                                                        | -                                                            |
| 4    | [770c320](https://github.com/zyantific/zydis/commit/770c3203ef81040af892e3ae6ca42252edbab43c) | [stack-buffer-overflow](https://github.com/zyantific/zydis/issues/318) | [stack-buffer-overflow](https://github.com/nisl-bugTest/Pocfiles/blob/main/fuzzing/zydis/zydisinputpeek) | Fixed                                                        | -                                                            |

### <span id="14">13.[lepton](https://github.com/dropbox/lepton)</span>

| NO.  | Version                                                      | <span style="display:inline-block;width: 100px"> Issue link </span> | <span style="display:inline-block;width: 120px"> Poc</span>  | <span style="display:inline-block;width: 100px">IssueState</span> | <span style="display:inline-block;width: 100px"> CVEState</span> |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | [v1.0-1.2.1-185-g2a08b77](https://github.com/dropbox/lepton) | [heap-buffer-overflow](https://github.com/dropbox/lepton/issues/154) | [heap-buffer-overflow](https://github.com/NISL-SecurityGroup/NISL-BugDetection/blob/main/project/lepton/lepton_poc) | Fixed                                                        | [CVE-2022-26181](https://ubuntu.com/security/CVE-2022-26181) |

### <span id="15">14. [pdftojson](https://github.com/ldenoue/pdftojson)</span>

| NO.  | Version                                                      | <span style="display:inline-block;width: 100px"> Issue link </span> | <span style="display:inline-block;width: 120px"> Poc</span>  | <span style="display:inline-block;width: 100px">IssueState</span> | <span style="display:inline-block;width: 100px"> CVEState</span> |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | [94204bb](https://github.com/ldenoue/pdftojson/commit/94204bbfc523730db7c634c2c3952b9025cd7762) | [stack-buffer-overflow](https://github.com/ldenoue/pdftojson/issues/4) | [stack-buffer-overflow](https://github.com/NISL-SecurityGroup/NISL-BugDetection/blob/main) | Confirmed                                                    | [CVE-2022-44109](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2022-44109) |
| 2    | [94204bb](https://github.com/ldenoue/pdftojson/commit/94204bbfc523730db7c634c2c3952b9025cd7762) | [SEGV](https://github.com/ldenoue/pdftojson/issues/5)        | [SEGV](https://github.com/NISL-SecurityGroup/NISL-BugDetection/blob/main) | Pending                                                      | -                                                            |
| 3    | [94204bb](https://github.com/ldenoue/pdftojson/commit/94204bbfc523730db7c634c2c3952b9025cd7762) | [stack-overflow](https://github.com/ldenoue/pdftojson/issues/3) | [stack-overflow](https://github.com/NISL-SecurityGroup/NISL-BugDetection/blob/main) | Confirmed                                                    | [CVE-2022-44108](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2022-44108) |

### <span id="16">15.[xlsxio](https://github.com/brechtsanders/xlsxio)</span>

| NO.  | Version                                                      | <span style="display:inline-block;width: 100px"> Issue link </span> | <span style="display:inline-block;width: 120px"> Poc</span>  | <span style="display:inline-block;width: 100px">IssueState</span> | <span style="display:inline-block;width: 100px"> CVEState</span> |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | [af485eb](https://github.com/brechtsanders/xlsxio/commit/af485ebf4bb26f5dba39c91a8a06ac0f8c8d8016) | [illegal-memory-access](https://github.com/brechtsanders/xlsxio/issues/109) | [illegal-memory-access](https://github.com/NISL-SecurityGroup/NISL-BugDetection/blob/main) | Fixed                                                        | -                                                            |

### <span id="17">17.[ELFLoader](https://github.com/trustedsec/ELFLoader)</span>

| NO.  | Version                                                      | <span style="display:inline-block;width: 100px"> Issue link </span> | <span style="display:inline-block;width: 120px"> Poc</span>  | <span style="display:inline-block;width: 100px">IssueState</span> | <span style="display:inline-block;width: 100px"> CVEState</span> |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | [34fd7ba](https://github.com/trustedsec/ELFLoader/commit/34fd7ba6ccc4368f87bcfc8e4ef702496e4c5655) | [SEGV-crash](https://github.com/trustedsec/ELFLoader/issues/7) | [SEGV-crash](https://github.com/trustedsec/ELFLoader/files/9412984/000000.zip) | Confirmed                                                    | -                                                            |
| 2    | [34fd7ba](https://github.com/trustedsec/ELFLoader/commit/34fd7ba6ccc4368f87bcfc8e4ef702496e4c5655) | [heap-buffer-overflow](https://github.com/trustedsec/ELFLoader/issues/6) | [heap-buffer-overflow](https://github.com/trustedsec/ELFLoader/files/9412810/000012.zip) | Confirmed                                                    | -                                                            |
| 3    | [34fd7ba](https://github.com/trustedsec/ELFLoader/commit/34fd7ba6ccc4368f87bcfc8e4ef702496e4c5655) | [heap-buffer-overflow](https://github.com/trustedsec/ELFLoader/issues/5) | [heap-buffer-overflow](https://github.com/trustedsec/ELFLoader/files/9412776/000009.zip) | Confirmed                                                    | -                                                            |
| 4    | [34fd7ba](https://github.com/trustedsec/ELFLoader/commit/34fd7ba6ccc4368f87bcfc8e4ef702496e4c5655) | [heap-buffer-overflow](https://github.com/trustedsec/ELFLoader/issues/4) | [heap-buffer-overflow](https://github.com/trustedsec/ELFLoader/files/9412761/000001.zip) | Confirmed                                                    | -                                                            |

### <span id="18">18.[deark](https://github.com/jsummers/deark)</span>

| NO.  | Version        | <span style="display:inline-block;width: 100px"> Issue link </span> | <span style="display:inline-block;width: 120px"> Poc</span>  | <span style="display:inline-block;width: 100px">IssueState</span> | <span style="display:inline-block;width: 100px"> CVEState</span> |
| ---- | -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | Version v1.6.2 | [SEGV-crash](https://github.com/jsummers/deark/issues/51)    | [SEGV-crash](https://github.com/jsummers/deark/files/9594067/id_000037.sig_11.src_007860%2B004032.time_79856072.execs_177982905.op_splice.rep_4.zip) | Fixed                                                        | -                                                            |
| 2    | Version v1.6.2 | [stack-buffer-overflow](https://github.com/jsummers/deark/issues/52) | [stack-buffer-overflow](https://github.com/jsummers/deark/files/9594078/id_000027.sig_11.src_013544%2B002505.time_31840218.execs_68965869.op_splice.rep_16.zip) | Fixed                                                        | [CVE-2022-43289](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2022-43289) |
| 3    | Version v1.6.2 | [DEADLYSIGNAL](https://github.com/jsummers/deark/issues/50)  | [DEADLYSIGNAL](https://github.com/jsummers/deark/files/9594060/id_000006.sig_08.src_007594.time_3689202.execs_8229562.op_havoc.rep_2.zip) | Fixed                                                        | -                                                            |

### <span id="19">19.[Kernel](http://git.kernel.org/)</span>

| NO.  | Version                                                      | <span style="display:inline-block;width: 100px"> Issue link </span> | <span style="display:inline-block;width: 120px"> Poc</span> | <span style="display:inline-block;width: 100px">IssueState</span> | <span style="display:inline-block;width: 100px"> CVEState</span> |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | [Version 6.0](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/security/selinux/hooks.c?id=9bd572ec7a66b56e1aed896217ff311d981cf575#n4535) | [use-after-free](https://syzkaller.appspot.com/bug?extid=04b20e641c99a5d99ac2) | use-after-free                                              | confirmed                                                    | -                                                            |
| 2    | [Version 6.0](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/security/selinux/hooks.c?id=9bd572ec7a66b56e1aed896217ff311d981cf575#n4899) | [use-after-free](https://syzkaller.appspot.com/bug?extid=04b20e641c99a5d99ac2) | use-after-free                                              | confirmed                                                    | -                                                            |

### <span id="22">20.[sqlcheck](https://github.com/jarulraj/sqlcheck)</span>

| NO.  | Version                                                      | <span style="display:inline-block;width: 100px"> Issue link </span> | <span style="display:inline-block;width: 120px"> Poc</span>  | <span style="display:inline-block;width: 100px">IssueState</span> | <span style="display:inline-block;width: 100px"> CVEState</span> |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | [391ae84](https://github.com/jarulraj/sqlcheck/commit/391ae8434c089021e1a114d6663b0a384011e2bc) | [heap-buffer-overflow](https://github.com/jarulraj/sqlcheck/issues/57) | [heap-buffer-overflow](https://github.com/jarulraj/sqlcheck/issues/57) | Pending                                                      | CVE-2023-23057                                               |
| 2    | [391ae84](https://github.com/jarulraj/sqlcheck/commit/391ae8434c089021e1a114d6663b0a384011e2bc) | [heap-buffer-overflow](https://github.com/jarulraj/sqlcheck/issues/59) | [heap-buffer-overflow](https://github.com/jarulraj/sqlcheck/issues/57) | Pending                                                      | CVE-2023-23055                                               |
| 3    | [391ae84](https://github.com/jarulraj/sqlcheck/commit/391ae8434c089021e1a114d6663b0a384011e2bc) | [stack-buffer-overflow](https://github.com/jarulraj/sqlcheck/issues/58) | [heap-buffer-overflow](https://github.com/jarulraj/sqlcheck/issues/57) | Pending                                                      | CVE-2023-23058                                               |
| 4    | [391ae84](https://github.com/jarulraj/sqlcheck/commit/391ae8434c089021e1a114d6663b0a384011e2bc) | [stack-buffer-overflow](https://github.com/jarulraj/sqlcheck/issues/56) | [heap-buffer-overflow](https://github.com/jarulraj/sqlcheck/issues/57) | Pending                                                      | CVE-2023-23056                                               |





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
