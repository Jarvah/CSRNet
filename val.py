#!/usr/bin/env python
# coding: utf-8

# In[42]:

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter 
import scipy
import json
import torchvision.transforms.functional as F
from matplotlib import cm as CM
from image import *
from model import CSRNet
import torch
from tqdm import tqdm
from matplotlib import cm as c
import glob
#get_ipython().magic(u'matplotlib inline')
import csv
# In[10]:


from torchvision import datasets, transforms
transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])


# In[3]:


root = '/home/waiyang/crowd_counting/Dataset/ShanghaiTech'

# In[4]:


#now generate the ShanghaiA's ground truth
part_A_train = os.path.join(root,'part_A/train_data','images')
part_A_test = os.path.join(root,'part_A/test_data','images')
part_B_train = os.path.join(root,'part_B/train_data','images')
part_B_test = os.path.join(root,'part_B/test_data','images')
path_sets = [part_B_test]
# In[5]:


img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)


# In[6]:


model = CSRNet()


# In[7]:


model = model.cuda()


# In[38]:


checkpoint = torch.load('download_model_best.pth.tar')


# In[39]:


model.load_state_dict(checkpoint['state_dict'])


# In[45]:


#mae = 0
#for i in tqdm(range(len(img_paths))):
    #img = 255.0 * F.to_tensor(Image.open(img_paths[i]).convert('RGB'))

    #img[0,:,:]=img[0,:,:]-92.8207477031
    #img[1,:,:]=img[1,:,:]-95.2757037428
    #img[2,:,:]=img[2,:,:]-104.877445883
    #img = img.cuda()
#    img = transform(Image.open(img_paths[i]).convert('RGB')).cuda()
#    gt_file = h5py.File(img_paths[i].replace('.jpg','.h5').replace('images','ground-truth'),'r')
#    groundtruth = np.asarray(gt_file['density'])
#    output = model(img.unsqueeze(0))
#    mae += abs(output.detach().cpu().sum().numpy()-np.sum(groundtruth))
#    print (i,mae)
#print (mae/len(img_paths))




#prediction


img = transform(Image.open('/home/waiyang/crowd_counting/Dataset/ShanghaiTech/part_A/test_data/images/IMG_86.jpg').convert('RGB')).cuda()

output = model(img.unsqueeze(0))
print("Predicted Count : ",int(output.detach().cpu().sum().numpy()))
#temp = np.asarray(output.detach().cpu().reshape(output.detach().cpu().shape[2],output.detach().cpu().shape[3]))
#plt.imshow(temp,cmap = c.jet)

#plt.show()

temp = h5py.File('/home/waiyang/crowd_counting/Dataset/ShanghaiTech/part_A/test_data/ground-truth/IMG_86.h5', 'r')
temp_1 = np.asarray(temp['density'])
plt.imshow(temp_1,cmap = c.jet)
print("Original Count : ",int(np.sum(temp_1)) + 1)
#plt.show()
#print("Original Image")
#plt.imshow(plt.imread('/home/waiyang/crowd_counting/Dataset/ShanghaiTech/part_B/test_data/images/IMG_20.jpg'))
#plt.show()

exit()
#test on target dataset
test_img_IMM='/home/waiyang/crowd_counting/Dataset/test_image_20190527/IMM'
test_img_jCube='/home/waiyang/crowd_counting/Dataset/test_image_20190527/jCube'
test_img_WestGate='/home/waiyang/crowd_counting/Dataset/test_image_20190527/WestGate'
test_img=test_img_WestGate

folders=glob.glob(os.path.join(test_img,'*'))
test_img_paths = []
i=0
for folder in folders:
    for path in glob.glob(folder+'/*.jpg'):
        test_img_paths.append(path)

print("predict target test images")
csvData=[[]]
for i in tqdm(range(len(test_img_paths))):
    img = transform(Image.open(test_img_paths[i]).convert(
        'RGB')).cuda()
    output = model(img.unsqueeze(0))
    image_no=os.path.basename(test_img_paths[i])
    predict=int(output.detach().cpu().sum().numpy())
    print("image no :", image_no)
    print("Predicted Count : ", predict)
    csvData.append((image_no,predict))

with open('WestGate_counting_CSRNet_partB.csv','w') as csvFile:
    writer=csv.writer(csvFile)
    writer.writerows(csvData)
csvFile.close()