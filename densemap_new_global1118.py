# -*- coding:utf-8 -*-
from PIL import Image
import numpy as np
import random
from random import choice
import h5py
import scipy.io as io
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy
import json
from matplotlib import cm as CM
# import torch

#part of the code is borrowed from https://github.com/leeyeehoo/CSRNet-pytorch
#this function is borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
def gaussian_filter_density(gt):
    print (gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density
    non_zero_gt=np.nonzero(gt)
    # pts = np.array(zip(non_zero_gt[1], non_zero_gt[0]))
    pts = np.array(np.stack([non_zero_gt[1],non_zero_gt[0]],axis=1) )
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    print ('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print ('done.')
    return density

root = './ShanghaiTech_Crowd_Counting_Dataset'
# part_A_train_gt = os.path.join(root,'part_A_final/train_data_newmap','ground_truth')
# part_A_train_gt = os.path.join(root,'part_A_final/test_data_15','ground_truth')
part_A_train_image= os.path.join(root,'part_A_final/train_data','images')
# part_A_train_newgt = os.path.join(root,'part_A_final/train_data_newmap','ground_truth_new_1')
# part_A_train_mat = os.path.join(root,'part_A_final/train_data_newmap','ground_truth_old')
path=part_A_train_image
img_paths = []
for img_path in glob.glob(os.path.join(path, '*.jpg')):
    img_paths.append(img_path)

# gt_paths=[
#     '/media/adminn/文档/ShanghaiTech_Crowd_Counting_Dataset/part_A_final/test_data_X/ground_truth/IMG_9.h5',
#     '/media/adminn/文档/ShanghaiTech_Crowd_Counting_Dataset/part_A_final/test_data_X/ground_truth/IMG_44.h5',
#     '/media/adminn/文档/ShanghaiTech_Crowd_Counting_Dataset/part_A_final/test_data_X/ground_truth/IMG_105.h5',
#     '/media/adminn/文档/ShanghaiTech_Crowd_Counting_Dataset/part_A_final/test_data_X/ground_truth/IMG_115.h5',
#     '/media/adminn/文档/ShanghaiTech_Crowd_Counting_Dataset/part_A_final/test_data_X/ground_truth/IMG_17.h5',
#     '/media/adminn/文档/ShanghaiTech_Crowd_Counting_Dataset/part_A_final/test_data_X/ground_truth/IMG_110.h5'
# ]
img_paths=['D:/counting_data/ShanghaiTech_Crowd_Counting_Dataset/part_A_final/train_data']
for img_path in img_paths:
    mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_'))
    img= plt.imread(img_path)
    init_map = np.zeros((img.shape[0],img.shape[1]))
    gt = mat["image_info"][0,0][0,0][0]
    for i in range(0,len(gt)):
        if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
            init_map[int(gt[i][1]),int(gt[i][0])]=1
    init_map = gaussian_filter(init_map,25)
    # h5_file_path=gt_path.replace('test_data_15', 'test_data_newmap1').replace('ground_truth', 'ground_truth')
    # h5_file_path = gt_path.replace('test_data_15', 'test_data_newmap1').replace('ground_truth', 'ground_truth_test')
    # gt_path=img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
    # h5_file_path = gt_path.replace('train_data', 'train_data_newmap1').replace('ground_truth', 'ground_truth')
    h5_file_path=img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
    if os.path.exists(h5_file_path):
        continue
    print(h5_file_path)
    gt_origin = init_map
    gt_oringin_c=gt_origin.copy()               #densemap produced by knn
    y_ems=np.sum(gt_oringin_c,axis=1)           #sum every row count of the densemap
    y_ems_i= y_ems / gt_origin.shape[1]         #computer the means of each row
    y_ems_mask_1=[]
    #enforce the dense
    for  i_colomns in range(gt_origin.shape[0]):    #for each row ,get the mask of the value above the means of the row
        y_ems_mask_i = gt_oringin_c[i_colomns] >(y_ems_i[i_colomns]*1)
        y_ems_mask_1.append(y_ems_mask_i)

    y_ems_mask_1=np.array([y_ems_mask_1]).squeeze(0)   #change the mask into array
    y_ems_count = np.sum(y_ems_mask_1, axis=1)    #count the value number above the means   for each row
    y_ems_1=gt_oringin_c*y_ems_mask_1            #get the value above the means
    y_ems_dense = np.sum(y_ems_1, axis=1)         #sum the value above the means  for each row
    y_ems_dense[y_ems_dense==0]=0.01              #if the value is abnormal ,set a special value
    y_ems_count[y_ems_count==0]=1                 #if the count value is abnormal ,set a small value
    distance = y_ems_dense / y_ems_count          #
    k = np.zeros((gt_oringin_c.shape[0], gt_oringin_c.shape[1]))


    for i in range(0, len(gt)):
        temp_density=np.zeros((gt_oringin_c.shape[0], gt_oringin_c.shape[1]))
        gt_x=int(gt[i][1])
        gt_y=int(gt[i][0])
        if  gt_x< gt_oringin_c.shape[0] and gt_y< gt_oringin_c.shape[1]:
            x1=8000
            x2=100
            y1=20
            y2=2.5
            a = (y2-y1)/(x2-x1)
            b = y1-a*x1
            x_ = (1 / distance[gt_x])
            theta = b + a * x_
            temp_density=gaussian_filter(temp_density,theta)
            k=k+temp_density

    with h5py.File(h5_file_path,'w') as hf:
        hf['density'] = k










