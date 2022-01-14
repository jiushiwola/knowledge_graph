import numpy as np
import torch
a = np.array([[1,2,3], [4,5,6]])
a[:,1:] = [[7,8],[9,10]]
b = torch.tensor(a)
c = b[[1,0]]
print(c)
print(b)

d = [2] * 5
print(d)
e = np.empty((2,4))
print(e)