#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py    
@Desc    :   
@Project :   aae_hash
@Contact :   thefreer@outlook.com
@License :   (C)Copyright 2018-2019, TheFreer.NET
@WebSite :   www.thefreer.net
@Modify Time           @Author        @Version
------------           -------        --------
2019/06/23 1:02       the freer      1.0         
'''
import argparse
import os

from pipiline import Pipeline
from setting import TRAIN_DIR

#参数
parser = argparse.ArgumentParser()
## input data: file/dir/""
parser.add_argument('--ori_p', type=str, default="", help='dir or file or None origin path')
parser.add_argument('--enc_p', type=str, default=TRAIN_DIR+"imgs/", help='dir or file encode path')
## pipeline control params bool
parser.add_argument('--train', type=int, default=0, help='train with exist models and models will be saved in checkpoints')
parser.add_argument('--retrain', type=int, default=0, help='train ignore exist models and models will be saved in models')
## pipeline control params int
parser.add_argument('--n_step', type=int, default=0, help='step of pipeline of exec')
parser.add_argument('--hash_size', type=int, default=20, help='size of the hash')
parser.add_argument('--pool_size', type=int, default=30, help='size of the gevent Pool')
## deep network params
parser.add_argument('--n_epochs', type=int, default=2500, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=512, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=28, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent code')
parser.add_argument('--img_size', type=int, default=64, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=400, help='interval between image sampling')

args = parser.parse_args()

args.enc_p = "E:\\GitHub\\thefreer98\\aae_hash\\aae_hash\\tests\\enc"
args.n_step = 3
if args.ori_p != "":
	if args.ori_p[-1]!="/" or args.ori_p[-1]!="\\":
		args.ori_p += "\\"
if os.path.isdir(args.enc_p):
	if args.enc_p[-1]!="/" or args.enc_p[-1]!="\\":
		args.enc_p += "\\"

pipeline = Pipeline(args)
try:
	res = pipeline.schedule()
except:
	raise print("请仔细检查参数，可参考说明文档！！！")

i = 0
try:
	for r in res:
		print(i, ":\t", r)
		i+=1
except:
	print(res)