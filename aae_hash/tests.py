#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   tests.py    
@Desc    :   
@Project :   aae_hash
@Contact :   thefreer@outlook.com
@License :   (C)Copyright 2018-2019, TheFreer.NET
@WebSite :   www.thefreer.net
@Modify Time           @Author        @Version
------------           -------        --------
2019/06/22 23:29       the freer      1.0         
'''
import cv2 as cv

from setting import TEST_DIR
from aae import encode_transform, train
from hash import Hash
from main import args

img_1 = cv.imread(TEST_DIR+"1.jpg")
img_2 = cv.imread(TEST_DIR+"2.jpg")
enc_1 = encode_transform(img_1, args)
enc_2 = encode_transform(img_2, args)
my_hash = Hash()
ah_1 = my_hash.ahash(enc_1)
ah_2 = my_hash.ahash(enc_2)
print("1 and 2:\t\n", ah_1, "\n", ah_2)
a1 = my_hash.compare(ah_1, ah_2)
print("1 and 2 similarity:\t", a1)

img_3 = cv.imread(TEST_DIR+"3.jpg")
img_4 = cv.imread(TEST_DIR+"4.jpg")
enc_3 = encode_transform(img_3, args)
enc_4 = encode_transform(img_4, args)
my_hash = Hash()
ah_3 = my_hash.ahash(enc_3)
ah_4 = my_hash.ahash(enc_4)
print("3 and 4:\t\n", ah_3, "\n", ah_4)
a2 = my_hash.compare(ah_3, ah_4)
print("3 and 4 similarity:\t", a2)
