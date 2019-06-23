#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Hash.py    
@Desc    :   
@Project :   aae_hash
@Contact :   thefreer@outlook.com
@License :   (C)Copyright 2018-2019, TheFreer.NET
@WebSite :   www.thefreer.net
@Modify Time           @Author        @Version
------------           -------        --------
2019/06/22 21:15       the freer      1.0         
'''
import imagehash

class Hash(object):
	def __init__(self, hash_size=20):
		self.hash_size=hash_size
	
	def ahash(self, img):
		return imagehash.average_hash(img, hash_size=self.hash_size)
	
	def phash(self, img):
		return imagehash.phash(img, hash_size=self.hash_size)
	
	def phash_s(self, img):
		return imagehash.phash_simple(img, hash_size=self.hash_size)
	
	def dhash(self, img):
		return imagehash.dhash(img, hash_size=self.hash_size)
	
	def dhash_v(self, img):
		return imagehash.dhash_vertical(img, hash_size=self.hash_size)
	
	def whash(self, img):
		return imagehash.whash(img, hash_size=self.hash_size)
	
	def compare(self, h1, h2):
		'''
		计算两个 hash 的相似度
		:param h1: hash 字符串
		:param h2: hash 字符串
		:return:
		'''
		hash_1 = imagehash.hex_to_hash(h1)
		hash_2 = imagehash.hex_to_hash(h2)
		a = 1 - (hash_1 - hash_2) / len(hash_1.hash) ** 2
		return a
	
	def index(self):
		'''
		批量为库中图像设置 Hash 索引
		数据库操作
		:return:
		'''
		pass
	
	def query(self):
		'''
		查询单一图像在库中相似度前 n 位的图像
		相似度计算使用：self.compare(h1, h2)
		:return:
		'''
		pass
