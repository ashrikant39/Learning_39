import numpy as np
import torch
import sys
from collections import OrderedDict
from augmentations import aug_ohl_list, aug_name_ls
import pandas as pd

if __name__=="__main__":
	path= sys.argv[1]
	if 'pth' in path:
		dict= torch.load(path)
	else:
		dict= np.load(path, allow_pickle=True)
	prob_array= dict.item()['dis_ps'][-1]
	aug_list=[]

	for aug_name, aug_function, aug_prob in zip(aug_name_ls, aug_ohl_list, prob_array):
		data={}
		data['name']= aug_name
		data['function']= aug_function
		data['prob']= aug_prob
		aug_list.append(data)


	sorted_aug_list= sorted(aug_list, key= lambda i: i['prob'], reverse=True)
	df= pd.DataFrame(sorted_aug_list)
	df.to_csv('Augmentations.csv')
