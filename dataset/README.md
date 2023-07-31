# This is our introduction to the training and validation dataset used in the paper

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
