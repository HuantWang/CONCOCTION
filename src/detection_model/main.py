from evaluation_bug import Eval
import sys

if __name__ == "__main__":
    path_to_data = "/home/model/data/sard/122_124_126_191"
    model_to_load = (
        "/home/model/BUGCSE/f1_0.9918699265156729_2023-07-04_416_789_78_190.h5"
    )
    mode = "test"
    Eval(path_to_data, model_to_load, mode)
