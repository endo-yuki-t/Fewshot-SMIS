from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import numpy as np
import torch
import os

class InferenceDataset(Dataset):

	def __init__(self, root, opts, transform=None, data_num=100000000):
		#self.paths = sorted(data_utils.make_dataset(root))[:data_num]
		self.paths = sorted(data_utils.make_dataset(root))[-data_num:]
		self.transform = transform
		self.opts = opts

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, index):
		from_path = self.paths[index]
		from_im = Image.open(from_path)
		from_im = from_im.convert('RGB') if self.opts.label_nc == 0 else from_im.convert('L')
		#from_im = Image.fromarray(np.uint8(np.zeros((256,256))))
		if self.transform:
			from_im = self.transform(from_im)
			
		from_feat = []
		for i in range(from_im.shape[0]):
			ys,xs = np.where(from_im[i]==1)
			y = ys.mean() if len(ys)!=0 else 0
			x = xs.mean() if len(xs)!=0 else 0
			from_feat.extend([y,x])
		from_feat = torch.from_numpy(np.array(from_feat,dtype=np.float32))
		
		return from_im, os.path.basename(self.paths[index])
