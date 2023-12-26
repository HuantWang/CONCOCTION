import os,argparse
import shutil
from tqdm import tqdm as tqdm
import numpy as np

import logging
from operator import itemgetter
import pickle
import torch
from transformers import AutoTokenizer, AutoModel


def preprocess(fpath,storedDir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    map_dict={}
    path_num=16
    security = "safe word " * 64
    cache=os.path.join(os.path.dirname(__file__),'princeton-nlp/sup-simcse-bert-base-uncased')
    model = AutoModel.from_pretrained(cache).to(device)
    tokenizer = AutoTokenizer.from_pretrained(cache)
    def save_dict_to_pickle(dict_data, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(dict_data, f)
    def findAllFile(dir):
        for root, ds, fs in os.walk(dir):
            for f in fs:
                yield root, f

    if os.path.exists(os.path.join(fpath, "select_feature.pickle")):
        print("Load data from exist model")
    else:
        print("preprocess data........")
        data_path = fpath
        y = []
        gap = " "
        Graph_length = 50
        X_feature = {
            "name": {},
            "X_Code": {},
            "label": {},
            "cfg": {},
            "feature": {},
        }
        X_feature["name"] = []
        X_feature["X_Code"] = []
        X_feature["label"] = []
        X_feature["cfg"] = []
        X_feature["feature"] = []

        map_feature = {
            "name": {},
            "path": {},
            "feature": {},
        }
        map_feature["name"] = []
        map_feature["path"] = []
        map_feature["feature"] = []
        select_feature = {
            "feature": {},
            "label": {},
            "cluster": {},
            "sub_mask": {},
            "sub_adj": {},
            "vir_adj": {},
            "num_subg": {},
            "subg_size": {},
            "label_idx": {}
        }
        select_feature["feature"] = []
        select_feature["label"] = []
        select_feature["cluster"] = []
        select_feature["sub_mask"] = []
        select_feature["sub_adj"] = []
        select_feature["vir_adj"] = []
        select_feature["num_subg"] = []
        select_feature["subg_size"] = []
        select_feature["label_idx"] = []

        map_toc_dict = {}
        for root, file in tqdm(findAllFile(data_path), desc='dirs'):
            if file.endswith(".txt"):
                flag = "none"
                file_path = os.path.join(root, file)
                X_Code_Single = []
                X_CFG_Single = np.zeros([Graph_length, Graph_length])
                X_CFG = []
                X_trace_Single = []
                X_testcase_single = []
                X_Node_Singe = []
                X_dynamic_single = []
                path_single = []
                path_all = []
                path_embeddings = []
                Dic_map = {}
                Dic_single = {}
                Dic_code_single = {}
                name_path={}
                path_vec={}

                f = open(file_path)
                for line in f:
                    if line == "-----label-----\n":
                        flag = "label"
                        continue
                    if line == "-----code-----\n":
                        flag = "code"
                        continue
                    if line == "-----path-----\n":
                        flag = "path"
                        continue
                    if line == "-----cfgNode-----\n":
                        flag = "cfgNode"
                        continue
                    if line == "-----children-----\n":
                        flag = "none"
                        continue
                    if flag == "none":
                        continue
                    if flag == "label":
                        y = line.split()
                        flag = "none"
                        continue
                    if flag == "code":
                        X_Code_line = line.split("\n")[0]
                        X_Code_Single = X_Code_Single + [X_Code_line]
                        continue
                    if flag == "path":
                        path_single = line.split()[0].split(',')
                        path_all.append(path_single)
                        continue
                    if flag == "cfgNode":
                        if line == "=====================================\n":
                            for path_single in path_all:
                                path_node=[]
                                for node in path_single:
                                    path_node.append(Dic_code_single[int(node)])
                                path_node_save=path_node
                                path_node = ' '.join(path_node)

                                X_path_embedding = tokenizer.batch_encode_plus(
                                    [path_node],
                                    return_tensors="pt",
                                    max_length=64,
                                    padding="max_length",
                                    truncation="longest_first",
                                ).to(device)

                                with torch.no_grad():
                                    dynamic_last_output = model(
                                        **X_path_embedding, output_hidden_states=True, return_dict=True
                                    ).last_hidden_state
                                X_path_embedding = torch.reshape(
                                    torch.nn.AdaptiveMaxPool2d((1, 128))(dynamic_last_output), (-1, 128)
                                )
                                path_node = path_node_save
                                name_path.setdefault(file_path, {}).setdefault(tuple(X_Code_Single), {})[X_path_embedding] = path_node

                            file_path = storedDir+"/" + file + ".pickle"
                            save_dict_to_pickle(name_path, file_path)

                            break
                        num_single = int(line.split(",")[0].split('(')[-1])
                        X_Code_line = ''.join(line.split("\n")[0].split(",")[1:])
                        Dic_code_single[num_single] = X_Code_line
                        continue
                    f.close()

                    while len(X_CFG) < path_num:
                        X_CFG.append(np.zeros((50, 50)))
                    if len(X_CFG) > path_num:
                        X_CFG = X_CFG[:path_num]
                    X_CFG = np.array(X_CFG).reshape(len(X_CFG), Graph_length, Graph_length)

            else:
                continue

        print(f"Execution path representation stored in {storedDir}")



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, help="data path"
    )
    parser.add_argument(
        "--stored_path", type=str, help="stored data path"
    )
    args = parser.parse_args()
    fpath= args.data_path
    storedDir=args.stored_path
    if not os.path.exists(storedDir):
        os.mkdir(storedDir)
    preprocess(fpath,storedDir)













