from sklearn.metrics import *

y_true = [0, 1, 2, 2, 2]
y_pred = [0, 0, 2, 2, 1]
target_names = ['class 0', 'class 1', 'class 2']
data = classification_report(y_true, y_pred, target_names=target_names)
pre = precision_score(y_true, y_pred,average="micro")
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred,average="micro")
recall = recall_score(y_true, y_pred,average="micro")

print()