#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   AAE.py    
@Desc    :   
@Project :   aae_hash
@Contact :   thefreer@outlook.com
@License :   (C)Copyright 2018-2019, TheFreer.NET
@WebSite :   www.thefreer.net
@Modify Time           @Author        @Version
------------           -------        --------
2019/06/22 21:13       the freer      1.0         
'''
import os
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch
import itertools
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.utils.data as Data
from torchvision import datasets

from utils import parse_img
from setting import OUTPUT_DIR, MODEL_DIR, TRAIN_DIR, CHECK_DIR

os.makedirs(OUTPUT_DIR + "dev/", exist_ok=True)
cuda = True if torch.cuda.is_available() else False
# 定义Pytorch张量
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

#再参数化
def reparameterization(mu, logvar, args):
	std = torch.exp(logvar / 2)
	sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), args.latent_dim))))
	z = sampled_z * std + mu
	return z

#编码器——生成器
class Encoder(nn.Module):
	def __init__(self, args):
		super(Encoder, self).__init__()
		
		self.args = args
		self.img_shape = (self.args.channels, self.args.img_size, self.args.img_size)
		self.model = nn.Sequential(
			nn.Linear(int(np.prod(self.img_shape)), 512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(512, 512),
			nn.BatchNorm1d(512),
			nn.LeakyReLU(0.2, inplace=True)
		)

		self.mu = nn.Linear(512, self.args.latent_dim)
		self.logvar = nn.Linear(512, self.args.latent_dim)

	def forward(self, img):
			img_flat = img.view(img.shape[0], -1)
			x = self.model(img_flat)
			mu = self.mu(x)
			logvar = self.logvar(x)
			z = reparameterization(mu, logvar, self.args)
			return z

#解码器
class Decoder(nn.Module):
	def __init__(self, args):
		super(Decoder, self).__init__()
		
		self.args = args
		self.img_shape = (self.args.channels, self.args.img_size, self.args.img_size)
		self.model = nn.Sequential(
			nn.Linear(self.args.latent_dim, 512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(512, 512),
			nn.BatchNorm1d(512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(512, int(np.prod(self.img_shape))),
			nn.Tanh()
		)

	def forward(self, z):
		img_flat = self.model(z)
		img = img_flat.view(img_flat.shape[0], *self.img_shape)
		return img

#判别器
class Discriminator(nn.Module):
	def __init__(self, args):
		super(Discriminator, self).__init__()
	
		self.args = args
		self.model = nn.Sequential(
			nn.Linear(self.args.latent_dim, 512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(512, 256),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(256, 1),
			nn.Sigmoid()
		)

	def forward(self, z):
		validity = self.model(z)
		return validity

def train(args):
	cuda = True if torch.cuda.is_available() else False
	# 定义Pytorch张量
	Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
	# 分别为自编码器和对抗网络设置损失函数
	adversarial_loss = torch.nn.BCELoss()
	pixelwise_loss = torch.nn.L1Loss()
	# 实例化
	encoder = Encoder(args)
	decoder = Decoder(args)
	discriminator = Discriminator(args)
	try:
		if args.retrain == 0:
			encoder.load_state_dict(torch.load(CHECK_DIR + "encoder_dict.pkl"))
			decoder.load_state_dict(torch.load(CHECK_DIR + "decoder_dict.pkl"))
			discriminator.load_state_dict(torch.load(CHECK_DIR + "discriminator_dict.pkl"))
			
			encoder.eval()
			decoder.eval()
			discriminator.eval()
	except:
		print("First Training!!!")
	
	if cuda:
		print("Cuda!!!")
		encoder.cuda()
		decoder.cuda()
		discriminator.cuda()
		adversarial_loss.cuda()
		pixelwise_loss.cuda()
	# 加载数据集
	train_data = datasets.ImageFolder(
		root=TRAIN_DIR,
		transform=transforms.ToTensor()
	)
	data_loader = Data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
	# Optimizers
	# 分别为编码器和对抗网络定义优化器
	optimizer_G = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()),
	                               lr=args.lr, betas=(args.b1, args.b2))
	optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr,
	                               betas=(args.b1, args.b2))
	
	# 每训练一次使用解码器生成图像结果
	def sample_image(n_row, batches_done):
		"""Saves a grid of generated digits"""
		# Sample noise
		# 随机产生z输入解码器
		z = Variable(Tensor(np.random.normal(0, 1, (n_row ** 2, args.latent_dim))))
		gen_imgs = decoder(z)
		save_image(gen_imgs.data, OUTPUT_DIR + 'dev/%d.png' % batches_done, nrow=n_row, normalize=True)
	
	# ----------
	#  Training
	# ----------
	# 进行训练
	for epoch in range(args.n_epochs):
		for i, (imgs, b_label) in enumerate(data_loader):
			# Adversarial ground truths
			# 定义real_deature和fake_feature
			valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
			fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
			# Configure input
			# real_imgs是读入的数据集中的真实图像
			real_imgs = Variable(imgs.type(Tensor))
			# -----------------
			#  Train Generator
			# -----------------
			# 训练生成器
			optimizer_G.zero_grad()
			encoded_imgs = encoder(real_imgs)
			decoded_imgs = decoder(encoded_imgs)
			# Loss measures generator's ability to fool the discriminator
			# 计算生成器的损失率
			g_loss = 0.001 * adversarial_loss(discriminator(encoded_imgs), valid) + \
			         0.999 * pixelwise_loss(decoded_imgs, real_imgs)
			# 损失率反向传播
			g_loss.backward()
			# 反向传播之后优化生成器
			optimizer_G.step()
			
			# ---------------------
			#  Train Discriminator
			# ---------------------
			# 训练判别器
			optimizer_D.zero_grad()
			# Sample noise as discriminator ground truth
			# z是以我们期望的分布生成的一维特征向量
			z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim))))
			# Measure discriminator's ability to classify real from generated samples
			# 分别计算真实损失率和虚假损失率
			real_loss = adversarial_loss(discriminator(z), valid)
			fake_loss = adversarial_loss(discriminator(encoded_imgs.detach()), fake)
			# 计算判别器损失率
			d_loss = 0.5 * (real_loss + fake_loss)
			# 反向传播
			d_loss.backward()
			# 优化判别器
			optimizer_D.step()
			print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (
			epoch, args.n_epochs, i, len(data_loader),
			d_loss.item(), g_loss.item()))
			batches_done = epoch * len(data_loader) + i
			# 保存结果图像以及模型
			if batches_done % args.sample_interval == 0:
				sample_image(n_row=10, batches_done=batches_done)
				if args.retrain == 0:
					# 只保存模型参数
					torch.save(encoder.state_dict(), CHECK_DIR+'encoder_dict.pkl')
					torch.save(decoder.state_dict(), CHECK_DIR+'decoder_dict.pkl')
					torch.save(discriminator.state_dict(), CHECK_DIR+'discriminator_dict.pkl')
					# 保存整个模型
					torch.save(encoder, CHECK_DIR+'encoder.pkl')
					torch.save(decoder, CHECK_DIR+'decoder.pkl')
					torch.save(discriminator, CHECK_DIR+'discriminator.pkl')
				else:
					# 只保存模型参数
					torch.save(encoder.state_dict(), MODEL_DIR + 'encoder_dict.pkl')
					torch.save(decoder.state_dict(), MODEL_DIR + 'decoder_dict.pkl')
					torch.save(discriminator.state_dict(), MODEL_DIR + 'discriminator_dict.pkl')
					# 保存整个模型
					torch.save(encoder, MODEL_DIR + 'encoder.pkl')
					torch.save(decoder, MODEL_DIR + 'decoder.pkl')
					torch.save(discriminator, MODEL_DIR + 'discriminator.pkl')


def encode_transform(ori_img, args):
	'''

	:param ori_img: ori_img 为 cv2 imread 对象
	:return: 10*10 PIL Image 对象
	'''
	
	cuda = True if torch.cuda.is_available() else False
	Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
	
	e_model = CHECK_DIR + "encoder_dict.pkl"
	encoder = Encoder(args)
	encoder.load_state_dict(torch.load(e_model))
	encoder.eval()
	if cuda:
		encoder.cuda()
	_trans = ori_img.ravel().reshape(1, 12288)
	_trans = torch.from_numpy(_trans)
	_trans = Variable(_trans.type(Tensor))
	_enc = encoder(_trans).reshape(10, 10)
	enc_img = parse_img(_enc.data)
	return enc_img