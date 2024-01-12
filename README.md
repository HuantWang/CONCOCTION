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

Check [our paper](https://github.com/HuantWang/HuantWang.github.io/blob/main/ICSE_24.pdf) for detailed information.

## Installation

Concoction builds upon:

-	Python v3.6
-	KLEE v3.0
-	LLVM v13.0

The system was tested on the following operating systems:

- Ubuntu 18.04

See [INSTALL.md](INSTALL.md) for further details.

## Usage

See [AE.md](./AE.md) for a step-by-step demo of Concoction.

## Data

Data are available at [here](./dataset/README.md).

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
