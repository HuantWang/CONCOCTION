import sys
import io, os
import numpy as np
import logging
logging.basicConfig(format="%(asctime)s : %(message)s", level=logging.DEBUG)
import argparse
from prettytable import PrettyTable
import torch
import torchmetrics
import transformers
import warnings
import nni
from transformers import AutoModel, AutoTokenizer,logging
logging.set_verbosity_error()

warnings.filterwarnings("ignore")
from sys import getrefcount
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    get_linear_schedule_with_warmup,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
)




# Set PATHs
PATH_TO_SENTEVAL=os.path.join(os.path.dirname(__file__),'SentEval')
# PATH_TO_SENTEVAL = "./SentEval"
# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

def main(args):
    cache1=os.path.join(os.path.dirname(__file__),"transformer/bert-base-uncased")
    cache2=os.path.join(os.path.dirname(__file__), "transformer/graphcodebert-base")
    model = AutoModel.from_pretrained(cache1)
    tokenizer = AutoTokenizer.from_pretrained(cache1)
    model_g = RobertaForSequenceClassification.from_pretrained(cache2)
    tokenizer_g = RobertaTokenizer.from_pretrained(cache2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model_g = model_g.to(device)

    # Set params for SentEval

    if args.mode == "train":
        # Full mode
        params = {"task_path": args.path_to_data, "usepytorch": True, "kfold": 10}
        params["classifier"] = {
            "nhid": 2,
            "optim": "adam",
            "batch_size": 64,
            "tenacity": 5,
            "epoch_size": 60,
        }
        params["classifier"] = nni.get_next_parameter()  # 获得下一组搜索空间中的参数
    else:
        raise NotImplementedError

    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    def batcher(params, batch, max_length=None):
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
        mask=mask.to(device)
        tokens_ids = []
        for i in batch_Node:
            code_tokens = tokenizer_g.tokenize(i + security)
            token_ids = tokenizer_g.convert_tokens_to_ids(code_tokens[:Graph_length])
            tokens_ids.append(token_ids)
        tokens_ids = torch.tensor(tokens_ids).to(torch.int64)
        tokens_ids=tokens_ids.to(device)

        # Get raw embeddings
        with torch.no_grad():
            dynamic_last_output = model(
                **batch_dynamic, output_hidden_states=True, return_dict=True
            ).last_hidden_state
            graph_last_output = model_g.roberta(
                inputs_embeds=model_g.roberta.embeddings.word_embeddings(tokens_ids),
                attention_mask=mask,
                return_dict=True,
            ).last_hidden_state

        feature = torch.cat((dynamic_last_output, graph_last_output), dim=1)

        # feature = dynamic_last_output

        embeddings = torch.reshape(
            torch.nn.AdaptiveMaxPool2d((1, 128))(feature), (-1, 128)
        )

        return embeddings

    se = senteval.engine.SE(params, batcher, prepare)
    se.eval("BUG")
    print(f"this is the data path: {args.path_to_data}")
    # results['BUG'] = result

def mainPred(args):
    cache1=os.path.join(os.path.dirname(__file__),"transformer/bert-base-uncased")
    cache2=os.path.join(os.path.dirname(__file__), "transformer/graphcodebert-base")
    model = AutoModel.from_pretrained(cache1)
    tokenizer = AutoTokenizer.from_pretrained(cache1)
    model_g = RobertaForSequenceClassification.from_pretrained(cache2)
    tokenizer_g = RobertaTokenizer.from_pretrained(cache2)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model = model.to(device)
    model_g = model_g.to(device)

    # Set params for SentEval

    if args.mode == "test":
        # Full mode
        params = {"task_path": args.path_to_data, "usepytorch": True, "kfold": 10}
        params["classifier"] = {
            "nhid": 2,
            "optim": "adam",
            "batch_size": 64,
            "tenacity": 5,
            "epoch_size": 200,
        }

        print(params["classifier"])
    else:
        raise NotImplementedError

    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    def batcher(params, batch, max_length=None):
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
        for i in batch_Node:
            code_tokens = tokenizer_g.tokenize(i + security)
            token_ids = tokenizer_g.convert_tokens_to_ids(code_tokens[:Graph_length])
            tokens_ids.append(token_ids)
        tokens_ids = torch.tensor(tokens_ids).to(torch.int64)

        # Get raw embeddings
        with torch.no_grad():
            dynamic_last_output = model(
                **batch_dynamic, output_hidden_states=True, return_dict=True
            ).last_hidden_state
            graph_last_output = model_g.roberta(
                inputs_embeds=model_g.roberta.embeddings.word_embeddings(tokens_ids),
                attention_mask=mask,
                return_dict=True,
            ).last_hidden_state

        feature = torch.cat((dynamic_last_output, graph_last_output), dim=1)

        # feature = dynamic_last_output

        embeddings = torch.reshape(
            torch.nn.AdaptiveMaxPool2d((1, 128))(feature), (-1, 128)
        )

        return embeddings

    se = senteval.engine.SE(params, batcher, prepare)
    se.predict("BUG",args.model_to_load)
    print(f"this is the data path: {args.path_to_data}")
    # results['BUG'] = result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, help="Transformers' model name or path"
    )
    parser.add_argument(
        "--path_to_data", type=str, help="data path"
    )
    parser.add_argument(
        "--model_to_load", type=str, help="the pretrained detection model's path"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        default="test",
        help="What evaluation mode to use (train:train the model; test:predict the specific testcase",
    )
    args = parser.parse_args()
    if args.mode=="train":
        main(args)
    elif args.mode=="test":
        mainPred(args)

def Eval(path_to_data,model_to_load,mode):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, help="Transformers' model name or path"
    )
    parser.add_argument(
        "--path_to_data", type=str, help="data path"
    )
    parser.add_argument(
        "--model_to_load", type=str, help="the pretrained detection model's path"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        help="What evaluation mode to use (train:train the model; test:predict the specific testcase",
    )
    args = parser.parse_args()
    if args.model_to_load==None:
        args.model_to_load=model_to_load
    if args.path_to_data==None:
        args.path_to_data = path_to_data
    if args.mode==None:
        args.mode=mode

    if args.mode=="train":
        main(args)
    elif args.mode=="test":
        mainPred(args)

