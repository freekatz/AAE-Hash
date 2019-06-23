#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   pipiline.py    
@Desc    :   
@Project :   aae_hash
@Contact :   thefreer@outlook.com
@License :   (C)Copyright 2018-2019, TheFreer.NET
@WebSite :   www.thefreer.net
@Modify Time           @Author        @Version
------------           -------        --------
2019/06/22 21:30       the freer      1.0         
'''
from gevent.pool import Pool
from glob import glob
import cv2 as cv
import os

from aae import encode_transform, train
from utils import img_migrate, img_reshape, img_cut
from hash import Hash

class Pipeline(object):
	def __init__(self, args):
		'''
		sum of step is: 4:0-4
		:param n_step:
		'''
		self.n_step = args.n_step
		self.pool_size = args.pool_size
		self.hash = Hash(hash_size=args.hash_size)
		self.args = args
		self.schedule()
		
	def multi(self, F, iterable):
		pool = Pool(size=self.pool_size)
		res = pool.imap(F, iterable)
		return res
	
	def schedule(self):
		ori_p = self.args.ori_p
		enc_p = self.args.enc_p
		if os.path.isfile(enc_p) and self.n_step == 0:
			enc_img = self.step_preprocess(enc_p)
			enc_img = self.step_encode(enc_img)
			hash_str = self.step_hash(enc_img)
			# res_list = self.step_query(hash_str)
			# return res_list
			return hash_str
		else:
			if self.n_step >= 1:
				try:
					if self.args.ori_p != "":
						self.step_preprocess_n(ori_p)
				except:
					raise print("请输入正确的原始图像路径！！！")
				if self.args.train == 1:
					self.step_train()
			if self.n_step >= 2:
				imgs = self.multi(cv.imread, [self.args.enc_p + m for m in os.listdir(self.args.enc_p)])
				enc_imgs = [i for i in self.step_encode_n(imgs)]
				# print(enc_imgs)
				# enc_imgs = []
				# for i in self.step_encode_n(imgs):
				# 	enc_imgs.append(i)
			else:
				return None
			
			if self.n_step >=3:
				hash_strs = [i for i in self.step_hash_n(enc_imgs)]
				# hash_strs = []
				# for i in self.step_hash_n(enc_imgs):
				# 	hash_strs.append(i)
			else:
				return enc_imgs
			if self.n_step >= 4:
				self.step_index(hash_strs)
			else:
				return hash_strs
	
	def step_preprocess(self, ori_p):
		img = img_cut(ori_p)
		img = cv.resize(img, (self.args.img_size, self.args.img_size))
		return img
	
	def step_preprocess_n(self, ori_p):
		img_migrate(ori_p=ori_p)
		img_reshape(self.args)
		
	def step_train(self):
		train(self.args)
		
	def step_encode(self, img):
		encode = encode_transform(img, self.args)
		return encode
	
	def step_encode_n(self, imgs):
		return self.multi(self.step_encode, imgs)
	
	def step_hash(self, img):
		h = self.hash.ahash(img)
		return h
	
	def step_hash_n(self, imgs):
		return self.multi(self.step_hash, imgs)
	
	def step_index(self, hash_strs):
		self.hash.index()
		
	def step_query(self, hash_str):
		self.hash.query()
	


if __name__ == '__main__':
	pool = Pool(size=100)
	imgs = glob("./data/imgs/*.jpg")