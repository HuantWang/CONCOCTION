from transformers import AutoModel, AutoTokenizer
import torch
import os
import io
import logging
import numpy as np
import os
from tqdm import tqdm

import random
from operator import itemgetter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class LoadFile:
    def loadFile(self, fpath):
        if os.path.exists(os.path.join(fpath, "feature.npy")):
            logger.info("Load data from exist npy")
            X_feature = np.load(
                os.path.join(fpath, "feature.npy"), allow_pickle=True
            ).item()

            index = list(range(len(X_feature["name"])))
            random.Random(123456).shuffle(index)

            train_idx = len(X_feature["name"])
            dev_idx = int(len(X_feature["name"]) * 0.8)

            index_train = index[:train_idx]
            index_dev = index[train_idx + 1 : dev_idx + 1]
            index_test = index[dev_idx + 2 :]

            train = {
                "name": {},
                "X_Code": {},
                "X_trace": {},
                "X_testcase": {},
                "X_Graph": {},
                "X_Node": {},
                "X_dynamic": {},
                "label": {},
            }
            train["name"] = itemgetter(*index_train)(X_feature["name"])
            train["X_Code"] = itemgetter(*index_train)(X_feature["X_Code"])
            train["X_trace"] = itemgetter(*index_train)(X_feature["X_trace"])
            train["X_testcase"] = itemgetter(*index_train)(X_feature["X_testcase"])
            train["X_Graph"] = itemgetter(*index_train)(X_feature["X_Graph"])
            train["X_Node"] = itemgetter(*index_train)(X_feature["X_Node"])
            train["X_dynamic"] = itemgetter(*index_train)(X_feature["X_dynamic"])
            train["label"] = itemgetter(*index_train)(X_feature["label"])

            dev = {
                "name": {},
                "X_Code": {},
                "X_trace": {},
                "X_testcase": {},
                "X_Graph": {},
                "X_Node": {},
                "X_dynamic": {},
                "label": {},
            }
            test = {
                "name": {},
                "X_Code": {},
                "X_trace": {},
                "X_testcase": {},
                "X_Graph": {},
                "X_Node": {},
                "X_dynamic": {},
                "label": {},
            }

            return train, dev, test
        else:
            logger.info("preprocess data........")

            def findAllFile(dir):
                for root, ds, fs in os.walk(dir):
                    for f in fs:
                        yield root, f

            data_path = fpath
            y = []
            gap = " "
            Graph_length = 50
            X_feature = {
                "name": {},
                "X_Code": {},
                "X_trace": {},
                "X_testcase": {},
                "X_Graph": {},
                "X_Node": {},
                "X_dynamic": {},
                "label": {},
            }
            X_feature["name"] = []
            X_feature["X_Code"] = []
            X_feature["X_trace"] = []
            X_feature["X_testcase"] = []
            X_feature["X_Graph"] = []
            X_feature["X_Node"] = []
            X_feature["X_dynamic"] = []
            X_feature["label"] = []

            # for root, file in tqdm(findAllFile(data_path), desc='dirs'):
            for root, file in tqdm(findAllFile(data_path), desc="dirs"):
                if file.endswith(".txt"):
                    flag = "none"
                    file_path = os.path.join(root, file)

                    X_Code_Single = []
                    X_Graph_Single = np.zeros([Graph_length, Graph_length])
                    X_trace_Single = []
                    X_testcase_single = []
                    X_Node_Singe = []
                    X_dynamic_single = []
                    f = open(file_path)
                    try:
                        for line in f:
                            if line == "-----label-----\n":
                                flag = "label"
                                continue
                            if line == "-----code-----\n":
                                flag = "code"
                                continue
                            if line == "-----children-----\n":
                                flag = "children"
                                continue
                            if line == "-----nextToken-----\n":
                                flag = "nextToken"
                                continue
                            if line == "-----computeFrom-----\n":
                                flag = "computeFrom"
                                continue
                            if line == "-----guardedBy-----\n":
                                flag = "guardedBy"
                                continue
                            if line == "-----guardedByNegation-----\n":
                                flag = "guardedByNegation"
                                continue
                            if line == "-----lastLexicalUse-----\n":
                                flag = "lastLexicalUse"
                                continue
                            if line == "-----jump-----\n":
                                flag = "jump"
                                continue
                            if line == "=======testcase========\n":
                                flag = "testcase"
                                continue
                            if line == "=========trace=========\n":
                                flag = "trace"
                                continue
                            if (
                                line == "-----attribute-----\n"
                                or line == "----------------dynamic----------------\n"
                            ):
                                flag = "next"
                                continue
                            if line == "-----ast_node-----\n":
                                flag = "ast_node"
                                continue
                            if line == "=======================\n":
                                break
                            if flag == "next":
                                continue
                            if flag == "label":
                                y = line.split()
                                continue
                            if flag == "code":
                                X_Code_line = line.split("\n")[0]
                                X_Code_Single = X_Code_Single + [X_Code_line]
                                continue
                            if flag == "children":
                                num_1 = int(line.split()[0].split(",")[0])
                                num_2 = int(line.split()[0].split(",")[1])
                                if num_2 < Graph_length and num_1 < Graph_length:
                                    X_Graph_Single[num_1 - 1, num_2 - 1] = 1
                                else:
                                    continue
                                continue
                            if flag == "nextToken":
                                num_1 = int(line.split()[0].split(",")[0])
                                num_2 = int(line.split()[0].split(",")[1])
                                X_Graph_Single[num_1 - 1, num_2 - 1] = 1
                                continue
                            if flag == "computeFrom":
                                num_1 = int(line.split()[0].split(",")[0])
                                num_2 = int(line.split()[0].split(",")[1])
                                if num_2 < Graph_length and num_1 < Graph_length:
                                    X_Graph_Single[num_1 - 1, num_2 - 1] = 1
                                else:
                                    continue
                                continue
                            if flag == "guardedBy":
                                num_1 = int(line.split()[0].split(",")[0])
                                num_2 = int(line.split()[0].split(",")[1])
                                if num_2 < Graph_length and num_1 < Graph_length:
                                    X_Graph_Single[num_1 - 1, num_2 - 1] = 1
                                else:
                                    continue
                                continue
                            if flag == "guardedByNegation":
                                num_1 = int(line.split()[0].split(",")[0])
                                num_2 = int(line.split()[0].split(",")[1])
                                if num_2 < Graph_length and num_1 < Graph_length:
                                    X_Graph_Single[num_1 - 1, num_2 - 1] = 1
                                else:
                                    continue
                                continue
                            if flag == "lastLexicalUse":
                                num_1 = int(line.split()[0].split(",")[0])
                                num_2 = int(line.split()[0].split(",")[1])
                                if num_2 < Graph_length and num_1 < Graph_length:
                                    X_Graph_Single[num_1 - 1, num_2 - 1] = 1
                                else:
                                    continue
                                continue
                            if flag == "jump":
                                num_1 = int(line.split()[0].split(",")[0])
                                num_2 = int(line.split()[0].split(",")[1])
                                if num_2 < Graph_length and num_1 < Graph_length:
                                    X_Graph_Single[num_1 - 1, num_2 - 1] = 1
                                else:
                                    continue
                                continue
                            if flag == "ast_node":
                                X_Code_line = line.split("\n")[0]
                                X_Node_Singe = X_Node_Singe + [X_Code_line]
                                continue
                            if flag == "testcase":
                                X_Code_line = line.split("\n")[0]
                                X_testcase_single = X_testcase_single + [X_Code_line]
                                X_dynamic_single = X_dynamic_single + [X_Code_line]
                            if flag == "trace":
                                X_Code_line = line.split("\n")[0]
                                X_trace_Single = X_trace_Single + [X_Code_line]
                                X_dynamic_single = X_dynamic_single + [X_Code_line]
                        f.close()
                    except:
                        logger.info("please delete the file " + file)

                    X_feature["name"].append(file_path)
                    X_feature["X_Code"].append(gap.join(X_Code_Single))
                    X_feature["X_trace"].append(gap.join(X_trace_Single).split())
                    X_feature["X_testcase"].append(gap.join(X_testcase_single).split())
                    X_feature["X_Graph"].append(X_Graph_Single)
                    X_feature["X_Node"].append(gap.join(X_Node_Singe))
                    X_feature["X_dynamic"].append(gap.join(X_dynamic_single).split())
                    X_feature["label"].append(int(y[0]))

            index = list(range(len(X_feature["name"])))
            random.Random(123456).shuffle(index)

            train_idx = len(X_feature["name"])
            dev_idx = int(len(X_feature["name"]) * 0.8)

            index_train = index[:train_idx]
            index_dev = index[train_idx + 1 : dev_idx + 1]
            index_test = index[dev_idx + 2 :]

            logger.info("Saving embedding...")
            np.save(os.path.join(fpath, "feature.npy"), X_feature)
            logger.info("Saving success")

            train = {
                "name": {},
                "X_Code": {},
                "X_trace": {},
                "X_testcase": {},
                "X_Graph": {},
                "X_Node": {},
                "X_dynamic": {},
                "label": {},
            }
            train["name"] = itemgetter(*index_train)(X_feature["name"])
            train["X_Code"] = itemgetter(*index_train)(X_feature["X_Code"])
            train["X_trace"] = itemgetter(*index_train)(X_feature["X_trace"])
            train["X_testcase"] = itemgetter(*index_train)(X_feature["X_testcase"])
            train["X_Graph"] = itemgetter(*index_train)(X_feature["X_Graph"])
            train["X_Node"] = itemgetter(*index_train)(X_feature["X_Node"])
            train["X_dynamic"] = itemgetter(*index_train)(X_feature["X_dynamic"])
            train["label"] = itemgetter(*index_train)(X_feature["label"])

            return train, dev, test


def embedding(model_name_or_path, PATH_TO_DATA):
    """
    model_name_or_path:model地址
    PATH_TO_DATA:特征地址
    """
    model = AutoModel.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model = model.to(device)

    def batcher(batch, max_length=None):
        # Handle rare token encoding issues in the dataset
        Graph_length = 50
        security = "safe word " * 64
        batch_dynamic = batch[0]
        batch_Node = batch[1]
        batch_Graph = batch[2]

        if (
            len(batch_dynamic) >= 1
            and len(batch_dynamic[0]) >= 1
            and isinstance(batch_dynamic[0][0], bytes)
        ):
            batch_dynamic = [
                [word.decode("utf-8") for word in s] for s in batch_dynamic
            ]

        sentences = [" ".join(s) for s in batch_dynamic]

        # Tokenization
        batch_dynamic = tokenizer.batch_encode_plus(
            sentences,
            return_tensors="pt",
            max_length=64,
            padding="max_length",
            truncation="longest_first",
        )

        # Move to the correct device
        for k in batch_dynamic:
            batch_dynamic[k] = batch_dynamic[k].to(device)

        mask = torch.tensor(batch_Graph).to(torch.int64)
        tokens_ids = []
        # for i in batch_Node:
        # code_tokens = tokenizer_g.tokenize(i + security)
        # token_ids = tokenizer_g.convert_tokens_to_ids(code_tokens[:Graph_length])
        # tokens_ids.append(token_ids)
        tokens_ids = torch.tensor(tokens_ids).to(torch.int64)

        # Get raw embeddings
        with torch.no_grad():
            dynamic_last_output = model(
                **batch_dynamic, output_hidden_states=True, return_dict=True
            ).last_hidden_state
            # graph_last_output = model_g.roberta(
            #     inputs_embeds=model_g.roberta.embeddings.word_embeddings(tokens_ids),
            #     attention_mask=mask,
            #     return_dict=True,
            # ).last_hidden_state

        # feature = torch.cat((dynamic_last_output, graph_last_output), dim=1)
        # embeddings = torch.reshape(
        #     torch.nn.AdaptiveMaxPool2d((1, 128))(feature), (-1, 128)
        # )

        return dynamic_last_output

    train, dev, test = LoadFile().loadFile(PATH_TO_DATA)
    sst_data = {"train": train, "dev": dev, "test": test}
    sst_embed = {"train": {}, "dev": {}, "test": {}}
    for key in sst_data:
        if key == "dev" or key == "test":
            continue
        logger.info("Computing embedding for {0}".format(key))
        # Sort to reduce padding
        sorted_data = sorted(
            zip(
                sst_data[key]["name"],
                sst_data[key]["X_Code"],
                sst_data[key]["X_trace"],
                sst_data[key]["X_testcase"],
                sst_data[key]["X_Graph"],
                sst_data[key]["X_Node"],
                sst_data[key]["X_dynamic"],
                sst_data[key]["label"],
            ),
            key=lambda z: (len(z[0]), z[-1]),
        )

        (
            sst_data[key]["name"],
            sst_data[key]["X_Code"],
            sst_data[key]["X_trace"],
            sst_data[key]["X_testcase"],
            sst_data[key]["X_Graph"],
            sst_data[key]["X_Node"],
            sst_data[key]["X_dynamic"],
            sst_data[key]["label"],
        ) = map(list, zip(*sorted_data))

        sst_embed[key]["X"] = []
        bsize = 64
        for ii in tqdm(range(0, len(sst_data[key]["label"]), bsize)):
            embeddings = []
            batch = (
                sst_data[key]["X_dynamic"][ii : ii + bsize],
                sst_data[key]["X_Node"][ii : ii + bsize],
                sst_data[key]["X_Graph"][ii : ii + bsize],
            )
            embeddings = batcher(batch)
            sst_embed[key]["X"].append(embeddings)

        sst_embed[key]["X"] = torch.cat(sst_embed[key]["X"]).detach().numpy()
        # sst_embed[key]['X'] = np.vstack(sst_embed[key]['X'])
        sst_embed[key]["y"] = np.array(sst_data[key]["label"])
        logger.info("Computed {0} embeddings".format(key))
        logger.info("This is embeddings:{0}".format(sst_embed[key]["X"]))
