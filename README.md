[![Maintenance](https://img.shields.io/badge/Maintained%3F-YES-green.svg)](https://github.com/HuantWang/SUPERSONIC/graphs/commit-activity)
[![License CC-BY-4.0](https://img.shields.io/badge/License-CC%20BY%204.0-blue.svg)](https://github.com/HuantWang/SUPERSONIC/blob/master/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/supersonic/badge/?version=latest)](https://supersonic.readthedocs.io/en/latest/?badge=latest)

<div align="center">
 <img src="./logo.png" alt="1683381967744" width=25% height=20%>
</div>
<p align="center" >
  <i>Deep learning based vulnerability detection model</i>
</p>



## Introduction

CONCOCTION is an automated machine learning based vulnerability detection 
framework. This is the first DL system to learn program presentations by 
combining static source code information and dynamic program execution 
traces.

Check [our paper]([https://github.com/HuantWang/HuantWang.github.io/blob/main/ICSE_24.pdf](https://eprints.whiterose.ac.uk/208077/1/ICSE_24___Camera_Ready__Copy_.pdf)) for detailed information.

## Installation

Concoction builds upon:

-	Python v3.6
-	KLEE v3.0
-	LLVM v13.0

The system was tested on the following operating systems:

- Ubuntu 18.04

See [INSTALL.md](INSTALL.md) for further details.

## Usage

See [usage.md](./AE.md) for a step-by-step demo of Concoction.

## Data

This is our introduction to the training and validation dataset used in the paper.
You can download the dataset from [here](https://drive.google.com/file/d/1ObjOKKWl0cS81ZF5KPMoSPprThVav5aA/view?usp=drive_link)

### Open datasets used in training and evaluation

This folder contains the datasets used in our paper.

`github:` This folder contains overall 6.8K C functions from C-language open-source projects.

`sard:` This folder contains overall 10K C functions from the SARD standard vulnerability dataset.

### Data structure

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


## Main Results

A full list of code vulnerabilities discovered by Concoction can be found [here](./vul_info/README.md).

## Contributing

We welcome contributions to Concoction. If you are interested in contributing please see
[this document](./CONTRIBUTING.md).

## Citation

If you use CONCOCTION in any of your work, please cite our paper:

~~~
@inproceedings{Concoction,
      title={Combining Structured Static Code Information and Dynamic Symbolic Traces for Software Vulnerability Prediction},
      author={Huanting Wang, Zhanyong Tang, Shin Hwei Tan, Jie Wang, Yuzhe Liu, Hejun Fang, Chunwei Xia, Zheng Wang},
      booktitle={The IEEE/ACM 46th International Conference on Software Engineering (ICSE)},
      year={2024},
}
~~~
