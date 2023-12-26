from transformers import AutoTokenizer, AutoModel
import torch,os,re,logging

from .setRootpath import getRootpath

ROOT_PATH=getRootpath()

def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)
set_global_logging_level(logging.ERROR, ["transformers", "nlp", "torch", "tensorflow", "tensorboard", "wandb"])


storedPath=os.path.join(ROOT_PATH,'concoction/pretrainedModel/staticRepresentation/trainedModel')
tokenizer = AutoTokenizer.from_pretrained(storedPath)
model = AutoModel.from_pretrained(storedPath)
nl_tokens=tokenizer.tokenize("return maximum value")
code_tokens=tokenizer.tokenize("def max(a,b): if a>b: return a else return b")
tokens=[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]+code_tokens+[tokenizer.eos_token]
tokens_ids=tokenizer.convert_tokens_to_ids(tokens)
context_embeddings=model(torch.tensor(tokens_ids)[None,:])[0]
print("code:")
print("return maximum value"+" "+"def max(a,b): if a>b: return a else return b")
print("embeddings:")
print(context_embeddings)