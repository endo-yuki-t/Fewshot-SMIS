import os
from argparse import Namespace

from tqdm import tqdm
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import sys

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im, log_input_image
from options.train_options import TrainOptions
from models.psp import pSp


def run():
	opts = TrainOptions().parse()
	os.makedirs(opts.exp_dir, exist_ok=True)
	checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
	os.makedirs(checkpoint_dir, exist_ok=True)
	data_dir = os.path.join(opts.exp_dir, 'data')
	os.makedirs(data_dir, exist_ok=True)
	os.makedirs(data_dir+'/labels', exist_ok=True)
	os.makedirs(data_dir+'/images', exist_ok=True)
	
	opts.device =  'cuda:0'
	net = pSp(opts)
	net.eval()
	net.cuda()

	input_im = torch.zeros((1,opts.label_nc,256,256))
	
	print("Generating images in %s"%(opts.exp_dir))
	for i in range(1):
		with torch.no_grad():
			input_cuda = input_im.cuda().float()
			result_batch, codes_batch = run_on_batch(input_cuda, net, opts)
	
		result = tensor2im(result_batch[0])
		codes = codes_batch[0]
		Image.fromarray(np.array(result.resize((256, 256)))).save(data_dir+'/images/%d.png'%(i))
		save_dict = {'codes':codes}
		torch.save(save_dict, checkpoint_dir+'/%d.pt'%(i))
	print("Done")

def run_on_batch(inputs, net, opts):

	latent_mask = list(range(opts.style_num))
	result_batch = []
	codes_batch = []
	for image_idx, input_image in enumerate(inputs):
		# get latent vector to inject into our input image
		vec_to_inject = np.random.randn(1, 512).astype('float32')
		_, latent_to_inject = net(torch.from_numpy(vec_to_inject).to("cuda"),
		                          input_code=True,
		                          return_latents=True)

		# get output image with injected style vector
		res, codes = net(input_image.unsqueeze(0).to("cuda").float(),
		          latent_mask=latent_mask,
		          inject_latent=latent_to_inject,
		          return_latents=True)
		
		codes_batch.append(codes)
		result_batch.append(res)
	result_batch = torch.cat(result_batch, dim=0)
	return result_batch, codes_batch


if __name__ == '__main__':
	run()
