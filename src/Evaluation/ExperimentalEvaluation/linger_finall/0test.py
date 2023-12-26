import numpy as np
import torch
label_tuple = (1, 1, 1, 0)
def convert(label_tuple):
    label_np = np.array(label_tuple)
    # 将标签转换为 one-hot 编码
    label_onehot = np.eye(2)[label_np]
    label_tensor = torch.from_numpy(label_onehot)
    return label_tensor


import tensorflow as tf

# 创建一些张量
tensor1 = tf.constant([1, 2, 3])
tensor2 = tf.constant([4, 5, 6])
tensor3 = tf.constant([7, 8, 9])

# 在新的轴（维度）上堆叠这些张量
stacked_tensor = tf.stack([tensor1, tensor2, tensor3])

print(stacked_tensor)
