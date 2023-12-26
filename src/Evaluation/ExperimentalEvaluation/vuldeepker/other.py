# from tensorboard.backend.event_processing import event_accumulator
#
# # 加载日志数据
# ea = event_accumulator.EventAccumulator('logs/log6/events.out.tfevents.1600745700.LAPTOP-16HRI9IS')
# ea.Reload()
# x = ea.scalars.Keys()
# y = ea.scalars
# print(ea.scalars.Keys())
#
# val_psnr = ea.scalars.Items('loss')
# print(len(val_psnr))
# print([(i.step, i.value) for i in val_psnr])
import numpy as np
# x = [[0,1,1],[2,3,4]]
# x = np.array(x)
# y = x[:,0]
# x = y.size
# print(x)
train_scores = [1,2,3,4,5,6,7]
train_scores = np.array(train_scores)
train_scores = train_scores.reshape(train_scores.size,1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
print()

