from sklearn.utils import class_weight
import torch
import numpy as np

# x = torch.randn(20, 5)
# y = torch.randint(0, 2, (20,))  # classes
# print("-->y", y)
# label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
# y = torch.tensor(label)
#
# labels = torch.tensor([[(l + 1) % 2, (l + 2) % 2] for l in label])
# y = labels.float()
# # class_weights = class_weight.compute_class_weight('balanced', np.unique(y), y.numpy())
# class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y.numpy())
#
# class_weights = torch.tensor(class_weights, dtype=torch.float)
#
# print(class_weights)

list = [1, 2, 3, 4, 5]
def fun(list):
    list[0] = 0

fun(list)
print(list)