import numpy as np
import torch
import sys

if __name__=="__main__":
	path= sys.argv[1]
	if 'pth' in path:
		dict= torch.load(path)
	else:
		dict= np.load(path, allow_pickle=True)
	print(dict)
