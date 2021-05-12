"""
This file defines the core research contribution
"""
import matplotlib

matplotlib.use('Agg')
import torch
from torch import nn
from models.encoders import psp_encoders
from models.stylegan2.model import Generator
from configs.paths_config import model_paths
import numpy as np

def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt


class pSp(nn.Module):

	def __init__(self, opts):
		super(pSp, self).__init__()
		self.set_opts(opts)
		# Define architecture
		self.encoder = self.set_encoder()
		if opts.style_num==18:
			self.decoder = Generator(1024, 512, 8, channel_multiplier=opts.channel_multiplier)
		elif opts.style_num==16:
			self.decoder = Generator(512, 512, 8, channel_multiplier=opts.channel_multiplier)
		elif opts.style_num==14:
			self.decoder = Generator(256, 512, 8, channel_multiplier=opts.channel_multiplier)
		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
		# Load weights if needed
		self.load_weights()
		
		self.id2prototype = []

	def set_encoder(self):
		if self.opts.encoder_type == 'GradualStyleEncoder':
			encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoW':
			encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoWPlus':
			encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoWPlus(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'GradualStyleEncoderV2':
			encoder = psp_encoders.GradualStyleEncoderV2(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'PointNetEncoder':
			encoder = psp_encoders.PointNetEncoder(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'PointNetEncoderV2':
			encoder = psp_encoders.PointNetEncoderV2(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'TransformerEncoder':
			encoder = psp_encoders.TransformerEncoder(50, 'ir_se', self.opts)
		else:
			raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
		return encoder

	def load_weights(self):
		if self.opts.checkpoint_path is not None:
			print('Loading pSp from checkpoint: {}'.format(self.opts.checkpoint_path))
			ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
			self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
			self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
			self.__load_latent_avg(ckpt)
		else:
			if self.opts.use_pretrained_encoder:
				print('Loading encoders weights from irse50!')
				encoder_ckpt = torch.load(model_paths['ir_se50'])
				# if input to encoder is not an RGB image, do not load the input layer weights
				if self.opts.label_nc != 0:
					encoder_ckpt = {k: v for k, v in encoder_ckpt.items() if "input_layer" not in k}
				self.encoder.load_state_dict(encoder_ckpt, strict=False)
			print('Loading decoder weights from pretrained!')
			ckpt = torch.load(self.opts.stylegan_weights)
			self.decoder.load_state_dict(ckpt['g_ema'], strict=True)
			if self.opts.learn_in_w:
				self.__load_latent_avg(ckpt, repeat=1)
			else:
				self.__load_latent_avg(ckpt, repeat=self.opts.style_num)

	def forward(self, x, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
	            inject_latent=None, return_latents=False, alpha=None):
		if input_code:
			codes = x
		else:
			codes = self.encoder(x)
			# normalize with respect to the center of an average face
			if self.opts.start_from_latent_avg:
				if self.opts.learn_in_w:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
				else:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)


		if latent_mask is not None:
			for i in latent_mask:
				if inject_latent is not None:
					if alpha is not None:
						codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
					else:
						codes[:, i] = inject_latent[:, i]
				else:
					codes[:, i] = 0

		input_is_latent = not input_code

		images, result_latent = self.decoder([codes],
		                                     input_is_latent=input_is_latent,
		                                     randomize_noise=randomize_noise,
		                                     return_latents=return_latents)
				
		if resize:
			images = self.face_pool(images)

		if return_latents:
			return images, result_latent
		else:
			return images
		
	def compute_rep_vec(self, x_list, codes_list, input_code=False, randomize_noise=True,
	            return_latents=False):
		
		input_is_latent = not input_code
		self.id2prototype = []
		denom = torch.zeros(x_list[0].shape[1]).cuda()
		for i in range(len(x_list)):
			x, codes = x_list[i], codes_list[i]
			images, result_latent, result_feature_map = self.decoder([codes],
			                                     input_is_latent=input_is_latent,
			                                     randomize_noise=randomize_noise,
			                                     return_latents=return_latents,
			                                     return_feature_map=True)
			
			small_x = nn.functional.interpolate(x, size=result_feature_map.shape[2:])

			id2prototype = []
			for cid in range(small_x.shape[1]):
				vec_list = result_feature_map.permute(0,2,3,1)[small_x[:,cid]==1]
				if vec_list.shape[0] == 0:
					id2prototype.append(torch.zeros(1,vec_list.shape[1]).cuda())
				else:
					id2prototype.append(vec_list.mean(axis=0).unsqueeze(0))
					denom[cid] += 1.
	
			id2prototype = torch.cat(id2prototype)
			self.id2prototype.append(id2prototype)
		
		self.id2prototype = sum(self.id2prototype)/(denom[:,None]+1e-8)
		self.id2prototype = self.id2prototype/(self.id2prototype.norm(dim=1)[:,None]+1e-8)

	def dense_pseudo_labeling(self, x, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
	            inject_latent=None, return_latents=False, alpha=None, return_labelmap=False):

		if input_code:
			codes = x
		else:
			codes = self.encoder(x)
			# normalize with respect to the center of an average face
			if self.opts.start_from_latent_avg:
				if self.opts.learn_in_w:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
				else:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)


		if latent_mask is not None:
			for i in latent_mask:
				if inject_latent is not None:
					if alpha is not None:
						codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
					else:
						codes[:, i] = inject_latent[:, i]
				else:
					codes[:, i] = 0

		input_is_latent = not input_code
		images, result_latent, result_feature_map = self.decoder([codes],
		                                     input_is_latent=input_is_latent,
		                                     randomize_noise=randomize_noise,
		                                     return_latents=return_latents,
		                                     return_feature_map=True)
		
		b, c, w, h = result_feature_map.shape
		wh = w*h
		result_feature_map = result_feature_map/(result_feature_map.norm(dim=1)[:,None]+1e-8)
		cos = torch.matmul(self.id2prototype,result_feature_map.reshape(b,c,wh))
		cos = cos.reshape((b,self.opts.label_nc,w,h))
		label_map = torch.argmax(cos, dim=1).unsqueeze(1).float()
		label_map = nn.functional.interpolate(label_map, size=(256,256)).long()

		if resize:
			images = self.face_pool(images)
		
		if return_labelmap and return_latents:
			return images, result_latent, label_map
		elif return_latents:
			return images, result_latent
		elif return_labelmap:
			return images, label_map
		else:
			return images
		
	def compute_rep_vecs(self, x_list, codes_list, input_code=False, randomize_noise=True,
	            return_latents=False):

		input_is_latent = not input_code
		
		self.id2prototype = [[] for i in range(x_list[0].shape[1])]
		for i in range(len(x_list)):
			x, codes = x_list[i], codes_list[i]
			images, result_latent, result_feature_map = self.decoder([codes],
			                                     input_is_latent=input_is_latent,
			                                     randomize_noise=randomize_noise,
			                                     return_latents=return_latents,
			                                     return_feature_map=True)
			
			small_x = nn.functional.interpolate(x, size=result_feature_map.shape[2:])
			for cid in range(small_x.shape[1]):
				vec_list = result_feature_map.permute(0,2,3,1)[small_x[:,cid]==1]
				if vec_list.shape[0] == 0:
					self.id2prototype[cid].append(torch.zeros(0,vec_list.shape[1]).cuda())
				else:
					self.id2prototype[cid].append(vec_list/(vec_list.norm(dim=1)[:,None]+1e-8))
					
		for cid in range(small_x.shape[1]):
			self.id2prototype[cid] = torch.cat(self.id2prototype[cid])

	def sparse_pseudo_labeling(self, x, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
	            inject_latent=None, return_latents=False, alpha=None, return_labelmap=False, topk=3, cos_th=0.5):

		if input_code:
			codes = x
		else:
			codes = self.encoder(x)
			# normalize with respect to the center of an average face
			if self.opts.start_from_latent_avg:
				if self.opts.learn_in_w:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
				else:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)


		if latent_mask is not None:
			for i in latent_mask:
				if inject_latent is not None:
					if alpha is not None:
						codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
					else:
						codes[:, i] = inject_latent[:, i]
				else:
					codes[:, i] = 0

		input_is_latent = not input_code
		images, result_latent, result_feature_map = self.decoder([codes],
		                                     input_is_latent=input_is_latent,
		                                     randomize_noise=randomize_noise,
		                                     return_latents=return_latents,
		                                     return_feature_map=True)
		b, c, w, h = result_feature_map.shape
		wh = w*h
		result_feature_map = result_feature_map/(result_feature_map.norm(dim=1)[:,None]+1e-8)
		label_map_list = []
		for bid in range(b):
			label_map = torch.zeros(wh).cuda()
			min_dist_map = torch.zeros(wh).cuda()
			for cid in range(1,len(self.id2prototype)): #start form 1 to ignore unknown id 0
				cos = torch.matmul(self.id2prototype[cid],result_feature_map[bid].reshape(c,wh))
				topk_val, topk_inds = torch.topk(cos, topk, largest=True, dim=1)
				topk_val, topk_inds = torch.flatten(topk_val), torch.flatten(topk_inds)
				topk_val, sorted_inds = torch.sort(topk_val, descending=True)
				topk_inds = topk_inds[sorted_inds]
				_, unique_first_inds = np.unique(topk_inds.data.cpu().numpy(),return_index=True)
				topk_inds = topk_inds[unique_first_inds]
				topk_val = topk_val[unique_first_inds]
				
				dist_map = torch.zeros(wh).cuda()
				dist_map[topk_inds[topk_val>cos_th]] = topk_val[topk_val>cos_th]
	
				label_map[min_dist_map<dist_map] = cid
				min_dist_map[min_dist_map<dist_map] = dist_map[min_dist_map<dist_map]
	
			label_map = label_map.reshape(w,h).unsqueeze(0).unsqueeze(0).float()
			label_map = nn.functional.interpolate(label_map, size=(256,256)).long()
			label_map_list.append(label_map)
		label_map_list = torch.cat(label_map_list)
		
		if resize:
			images = self.face_pool(images)
		
		if return_labelmap and return_latents:
			return images, result_latent, label_map_list
		elif return_latents:
			return images, result_latent
		elif return_labelmap:
			return images, label_map
		else:
			return images
		
	def set_opts(self, opts):
		self.opts = opts

	def __load_latent_avg(self, ckpt, repeat=None):
		if 'latent_avg' in ckpt:
			self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
			if repeat is not None:
				self.latent_avg = self.latent_avg.repeat(repeat, 1)
		else:
			self.latent_avg = None
