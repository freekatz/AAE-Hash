#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py    
@Desc    :   
@Project :   aae_hash
@Contact :   thefreer@outlook.com
@License :   (C)Copyright 2018-2019, TheFreer.NET
@WebSite :   www.thefreer.net
@Modify Time           @Author        @Version
------------           -------        --------
2019/06/22 22:24       the freer      1.0         
'''
from torchvision.utils import make_grid
from PIL import Image
import torch
import shutil
import re
import cv2 as cv
import numpy as np
import os

from setting import INPUT_DIR, TRAIN_DIR

# 图像形态学的卷积核，可自定义或者使用api
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (25, 25))
dCount = 6  # 膨胀次数
eCount = 1  # 腐蚀次数


# 图像预处理需要的一些方法，函数中的control为使用函数里的那种方法
class imgTools():
	def __init__(self):
		pass
	
	# 图像灰度化
	def imgGray(self, img):
		self.grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
		return self.grayImg
	
	# 图像变换
	def imgConver(self, grayImg):
		fi = grayImg / 255
		gamma = 2
		self.newImg = np.power(fi, gamma)
		self.newImg = np.uint8(np.clip((1.5 * grayImg + 15), 0, 255))
		return self.newImg
	
	# 图像增强
	def imgEnhance(self, grayImg):
		clahe = cv.createCLAHE(clipLimit=5.0, tileGridSize=(2, 2))
		
		self.enhanceImg = clahe.apply(grayImg)
		return self.enhanceImg
	
	# 图像滤波
	def imgBlur(self, img):
		self.blurImg = cv.GaussianBlur(img, (11, 11), 0)
		return self.blurImg
	
	# 图像二值化
	def imgBinary(self, grayImg):
		(_, self.thresh) = cv.threshold(grayImg, 0, 255, cv.THRESH_OTSU)
		return self.thresh
	
	# 图像腐蚀膨胀
	def imgMorp(self, control, thresh):
		# 开运算
		# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
		self.closed = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
		if control == 'open':
			self.closed = cv.dilate(self.closed, None, iterations=dCount)  # dCount膨胀次数
			self.closed = cv.erode(self.closed, None, iterations=eCount)  # eCount腐蚀次数
			return self.closed
		# 闭运算
		elif control == 'close':
			self.closed = cv.erode(self.closed, None, iterations=eCount)  # 腐蚀
			self.closed = cv.dilate(self.closed, None, iterations=dCount)  # 膨胀
			return self.closed
		# 其他形态学操作
		else:
			pass


imgTools = imgTools()


# 图像分割算法需要改进
def img_cut(path):
	# print(path)
	img = cv.imread(path)
	print(img.shape)
	grayImg = imgTools.imgGray(img)
	newImg = imgTools.imgConver(grayImg)
	blurImg = imgTools.imgBlur(newImg)
	enhanceImg = imgTools.imgEnhance(blurImg)
	canny = cv.Canny(enhanceImg, 50, 50 * 3, apertureSize=3)
	thresh = imgTools.imgBinary(canny)
	closed = imgTools.imgMorp('open', thresh)
	cnts, _ = cv.findContours(
		closed.copy(),
		cv.RETR_EXTERNAL,
		cv.CHAIN_APPROX_SIMPLE
	)
	c = sorted(cnts, key=cv.contourArea, reverse=True)[0]
	rect = cv.minAreaRect(c)
	box = np.int0(cv.boxPoints(rect))
	Xs = [i[0] for i in box]
	Ys = [i[1] for i in box]
	x1 = min(Xs)
	x2 = max(Xs)
	y1 = min(Ys)
	y2 = max(Ys)
	if x1 > 0 and y1 > 0:
		pass
	else:
		x1 = 0
		y1 = 0
	hight = y2 - y1
	width = x2 - x1
	cutImg = img[y1:y1 + hight, x1:x1 + width]
	cv.imwrite('./temp.jpg', cutImg)
	cv.imshow('cut', cutImg)
	cv.waitKey()
	cv.destroyAllWindows()
	return cutImg


def img_reshape(args):
	os.makedirs(TRAIN_DIR + "imgs/", exist_ok=True)
	img_list = os.listdir(INPUT_DIR)
	count = 0
	for im in img_list:
		img_path = INPUT_DIR + im
		try:
			img = img_cut(img_path)
			img = cv.resize(img, (args.img_size, args.img_size))
			cv.imwrite(TRAIN_DIR + "imgs/" + im, img)
			count += 1
		except:
			print(INPUT_DIR + im)

def parse_img(tensor, nrow=8, padding=2,
	               normalize=False, range=None, scale_each=False, pad_value=0):
	grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
		                 normalize=normalize, range=range, scale_each=scale_each)
	# Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
	ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
	im = Image.fromarray(ndarr)
	return im

def img_migrate(ori_p, img_list=[]):
	os.makedirs(INPUT_DIR, exist_ok=True)
	dir_list = os.listdir(ori_p)
	if len(dir_list) > 0:
		for name in dir_list:
			path = ori_p + name
			if os.path.isdir(path):
				path+="/"
				print("dir: ", path)
				img_list = img_migrate(path, img_list)
			else:
				name = name.lower()
				t = re.split(r"\.", name)
				img = ""
				for i in range(len(t)-1):
					img+=re.sub("\W", "", t[i])
				img = img + "." + t[-1]
				print("img: ", img)
				img_list.append(img)
				shutil.copyfile(path, INPUT_DIR+img)
				# os.remove(path)
	return img_list

if __name__ == '__main__':
	print("不支持直接运行！！！")