import numpy as np
np.set_printoptions(threshold=np.inf)
import  tensorflow as tf
import  os
import yaml
import json
import skimage
from PIL import Image
# import numpy
# import tensorflow as tf
# a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
# b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
# c = tf.matmul(a, b)
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# print(sess.run(c))


#
# with open('/home/mt/桌面/Labelme/170908_081815404_Camera_5_json/info.yaml') as f:
#     temp = yaml.load(f.read())
#     labels = temp['label_names']
#     del labels[0]
# print(labels)



# a='info.yaml'
# filestr = a.split(".")[1]
# print(filestr)


# class_ids = np.array([class_names.index(s) for s in labels_form])

# with open('/home/mt/桌面/Labelme/170908_062145788_Camera_5.json') as f:
#     temp = yaml.load(f.read())
#     labels = temp['label_names']
#     del labels[0]
# print(labels)

# def from_json_get_labels_form(image_id):
#
#     info = image_info[image_id]
#     with open(info['json_path']) as f:
#         data = json.load(f)
#     data_info=data['objects']
#     count_list=[]
#     labels_form=[]
#     for i in range(len(data_info)):
#         count_list.append(data_info[i]['label'])
#     for i in count_list:
#         if i==17:
#             labels_form .append('sky')
#         elif i==33 or i==161:
#             labels_form .append(('car'))
#         elif i==34 or i==162:
#             labels_form .append('motorbicycle')
#         elif i == 35 or i==163:
#             labels_form.append(('bicycle'))
#         elif i == 38:
#             labels_form.append('truck')
#         elif i == 40:
#             labels_form.append(('tricycle'))
#         elif i == 36 or i==164:
#             labels_form.append('person')
#         elif i==37 or i==165:
#             labels_form .append(('rider'))
#         elif i==38 or i==166:
#             labels_form .append('truck')
#         elif i==39 or i==167:
#             labels_form .append(('bus'))
#         elif i==40 or i==168:
#             labels_form .append('tricycle')
#         elif i==49:
#             labels_form .append(('road'))
#         elif i==50:
#             labels_form .append('siderwalk')
#         elif i==65:
#             labels_form .append(('traffic_cone'))
#         elif i==66:
#             labels_form .append('road_pile')
#         elif i==67:
#             labels_form .append(('fence'))
#         elif i==81:
#             labels_form .append('traffic_light')
#         elif i==82:
#             labels_form .append(('pole'))
#         elif i==83:
#             labels_form .append('traffic_sign')
#         elif i==84:
#             labels_form .append(('wall'))
#         elif i==85:
#             labels_form .append('dustbin')
#         elif i == 86:
#             labels_form.append(('billboard'))
#         elif i == 97:
#             labels_form.append('building')
#         elif i==113:
#             labels_form.append('vegatation')
#     return labels_form


# dataset_root_path="/media/mt/Seagate Backup Plus Drive/road01_ins/train_data/"
# img_floder = dataset_root_path + "pic"
# mask_floder = dataset_root_path + "cv2_mask"
# dir=os.listdir(mask_floder)
# for i in dir:
#     if 'ins' in i:
#         print(i)

# a='asddasdas'
# if 'v' in a or 'b' in a:
#     print('true')
# else:
#     print('Fla')
# if 'b' or 'c' in a:
#     print()



# dataset_root_path="/media/mt/Seagate Backup Plus Drive/road01_ins/train_data/"
# img_floder = dataset_root_path + "pic"
# mask_floder = dataset_root_path + "cv2_mask"
# #yaml_floder = dataset_root_path
# imglist = os.listdir(img_floder)
# print(imglist)

#
# a=np.zeros([10,10,4],dtype=np.uint8)
# b=np.logical_not(a[:, :, -1]).astype(np.uint8)
# print(a[1]*b)
#
# skimage.draw.rectangle



# dataset_root_path="/media/mt/Seagate Backup Plus Drive/road01_ins/train_data/"
# img_floder = dataset_root_path + "pic"
# mask_floder = dataset_root_path + "cv2_mask"
# #yaml_floder = dataset_root_path
# imglist = os.listdir(img_floder)
# count = len(imglist)
#
#
# def get_obj_index( image):
#     n = np.max(image)
#     return n

# def draw_mask(self, num_obj, mask, image):
#     info = self.image_info[image_id]
#     for index in range(num_obj):
#         for i in range(info['width']):
#             for j in range(info['height']):
#                 at_pixel = image.getpixel((i, j))
#                 if at_pixel == index + 1:
#                     mask[j, i, index] = 1
#     return mask
#
#
# def load_mask(self, image_id):
#     """Generate instance masks for shapes of the given image ID.
#     """
#     global iter_num
#     print("image_id", image_id)
#     info = self.image_info[image_id]
#     count = 1  # number of object
#     img = Image.open(info['mask_path'])
#     num_obj = self.get_obj_index(img)
#     mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
#     mask = self.draw_mask(num_obj, mask, img, image_id)
#     occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
#     for i in range(count - 2, -1, -1):
#         mask[:, :, i] = mask[:, :, i] * occlusion
#
#         occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
#     labels = []
#     labels = self.from_yaml_get_class(image_id)
#     labels_form = []
#     for i in range(len(labels)):
#         if labels[i].find("tongue") != -1:
#             # print "box"
#             labels_form.append("tongue")
#     class_ids = np.array([self.class_names.index(s) for s in labels_form])


# with open('/home/mt/桌面/Labelme/test/170908_073030202_Camera_5.json') as f:
#     data = json.load(f)
# data_info = data['objects']
# count_list = []
# labels_form = []
# for i in range(len(data_info)):
#     count_list.append(data_info[i]['label'])
# for i in count_list:
#     if i == 17:
#         labels_form.append('sky')
#     elif i == 33 or i == 161:
#         labels_form.append(('car'))
#     elif i == 34 or i == 162:
#         labels_form.append('motorbicycle')
#     elif i == 35 or i == 163:
#         labels_form.append(('bicycle'))
#     elif i == 38:
#         labels_form.append('truck')
#     elif i == 40:
#         labels_form.append(('tricycle'))
#     elif i == 36 or i == 164:
#         labels_form.append('person')
#     elif i == 37 or i == 165:
#         labels_form.append(('rider'))
#     elif i == 38 or i == 166:
#         labels_form.append('truck')
#     elif i == 39 or i == 167:
#         labels_form.append(('bus'))
#     elif i == 40 or i == 168:
#         labels_form.append('tricycle')
#     elif i == 49:
#         labels_form.append(('road'))
#     elif i == 50:
#         labels_form.append('siderwalk')
#     elif i == 65:
#         labels_form.append(('traffic_cone'))
#     elif i == 66:
#         labels_form.append('road_pile')
#     elif i == 67:
#         labels_form.append(('fence'))
#     elif i == 81:
#         labels_form.append('traffic_light')
#     elif i == 82:
#         labels_form.append(('pole'))
#     elif i == 83:
#         labels_form.append('traffic_sign')
#     elif i == 84:
#         labels_form.append(('wall'))
#     elif i == 85:
#         labels_form.append('dustbin')
#     elif i == 86:
#         labels_form.append(('billboard'))
#     elif i == 97:
#         labels_form.append('building')
#     elif i == 113:
#         labels_form.append('vegatation')
# print(labels_form)


# img = Image.open('/home/mt/桌面/Labelme/170908_081815404_Camera_5_json/label.png')
# # num_obj = get_obj_index(img)
# # print(num_obj)
# a=np.array(img)
# print(np.where(a==np.max(a)))
# print(np.where(a == np.max(a, axis=0)))
# print(np.max(a))
# print(a)


# for i in range(3,-1,-1):
#     print(i)

# a=['a','b','c','d']
# class_ids = np.array([self.class_names.index(s) for s in labels_form])

#
# class_ids = np.arange(5)
# print(class_ids)

# class_info=[]
# class_info.append([{"source": 'shape','class_id':1,"name":'car'},{"source": 'shape','class_id':2,"name":'person'},{"source": 'shape','class_id':3,"name":'car'}])
# class_info=class_info[0]
# print(class_info)
#
#
# def clean_name(name):
#     """Returns a shorter version of object names for cleaner display."""
#     return ",".join(name.split(",")[:1])
# a=[clean_name(c["name"]) for c in class_info]
# print(a)
# class_names = [c["name"] for c in class_info]
# print(class_names)
# class_ids = np.array([class_names.index(s) for s in class_names])
#
# print(class_ids)

# img = Image.open('/home/mt/桌面/Labelme/170908_081815404_Camera_5_json/img.png')


# img = Image.open('/home/mt/桌面/Labelme/170908_081815404_Camera_5_json/t')



# def draw_mask1(num_obj, mask, image):
#
#     # print("info[width]----->",info['width'],"-info[height]--->",info['height'])
#     for index in range(num_obj):
#         for i in range(24):
#             for j in range(24):
#                 # print("image_id-->",image_id,"-i--->",i,"-j--->",j)
#                 # print("info[width]----->",info['width'],"-info[height]--->",info['height'])
#                 at_pixel = image.getpixel((i, j))
#                 if at_pixel == index + 1:
#                     mask[j, i, index] = 1
#     return mask
#
#
#
# def draw_mask(num_obj, mask, image):
#
#     # print("info[width]----->",info['width'],"-info[height]--->",info['height'])
#     for index in range(num_obj):
#         for i in range(24):
#             for j in range(24):
#                 # print("image_id-->",image_id,"-i--->",i,"-j--->",j)
#                 # print("info[width]----->",info['width'],"-info[height]--->",info['height'])
#                 at_pixel = image.getpixel((i, j))
#                 if at_pixel == index + 1:
#                     if at_pixel==255:
#                         mask[j, i, index] = 0
#                     else:
#                         mask[j, i, index] = 1
#     return mask
#
# img = Image.open('/home/mt/桌面/Labelme/170908_081815404_Camera_5_json/test3_json/label.png')
# mask =np.zeros([24, 24, 3], dtype=np.uint8)
# print(mask)
# print('====================================')
# mask1=draw_mask1(3,mask,img)
# print(mask1)
# print('==========================================================')
# mask = draw_mask(3, mask, img)
# print(mask)
# print((mask==mask1).all())
#
# occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
#
#
# for i in range(3 - 2, -1, -1):
#     mask[:, :, i] = mask[:, :, i] * occlusion
#
#     print('===============================================================================')
#     occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
# print(len(mask))



# a = [1,4,3,3,4,2,3,4,5,6,1]
# b=[]
# for i in a:
#     if i not in b:
#         b.append(i)
# print(b)

#
# def count_words(s,n):
#     """Return the n most frequently occuring words in s."""
#     # TODO: Count the number of occurences of each word in s
#     s=list(s.split(' ')[:])
#     class_type=set(s)
#     b=[]
#     a=[]
#     for i in class_type:
#         class_name=i
#         class_num=s.count(i)
#         b.append((class_name,class_num))
#     b=sorted(b,key=lambda a:a[1],reverse=True)
#
#     if b[0][1] == b[1][1]:
#         a= sorted(b, reverse=True)
#     else:
#         for i in range(len(b)):
#             if i+1==len(b):
#                 break
#             elif b[i][1]>b[i+1][1]:
#                 a.append(b[i])
#                 b.pop(0)
#             else:
#                 b=sorted(b,reverse=True)
#                 a.append(b[i])
#     top_n=a[:n]
#     return top_n
#
#
# def test_run():
#     """Test count_words() with some inputs."""
#     print(count_words("cat bat mat cat bat cat", 3))
#     print(count_words("betty bought a bit of butter but the butter was bitter", 3))
#
#
# if __name__ == '__main__':
#     test_run()

#
# a=[('butter', 2), ('a', 1), ('betty', 3)]
# print(sorted(a,key= lambda b:b[1],reverse=True))
#
# dataset_root_path="/media/mt/Seagate Backup Plus Drive/road01_ins/train_data/"
# img_floder = dataset_root_path + "pic"
# mask_floder = dataset_root_path + "cv2_mask"
# #yaml_floder = dataset_root_path
# imglist = os.listdir(img_floder)
# count = len(imglist)
# json_path = mask_floder + "/" + filestr + ".json"

# imglist=os.listdir("/home/mt/桌面/未命名文件夹/")
# for i in imglist:
#     with open("/home/mt/桌面/未命名文件夹/"+i) as f:
#         data = json.load(f)
#     data_info = data['objects']
#     count_list = []
#     labels_form = []
#     for i in range(len(data_info)):
#         count_list.append(data_info[i]['label'])
#     for i in count_list:
#         if i == 17:
#             labels_form.append('sky')
#         elif i == 33 or i == 161:
#             labels_form.append(('car'))
#         elif i == 34 or i == 162:
#             labels_form.append('motorbicycle')
#         elif i == 35 or i == 163:
#             labels_form.append(('bicycle'))
#         elif i == 38:
#             labels_form.append('truck')
#         elif i == 40:
#             labels_form.append(('tricycle'))
#         elif i == 36 or i == 164:
#             labels_form.append('person')
#         elif i == 37 or i == 165:
#             labels_form.append(('rider'))
#         elif i == 38 or i == 166:
#             labels_form.append('truck')
#         elif i == 39 or i == 167:
#             labels_form.append(('bus'))
#         elif i == 40 or i == 168:
#             labels_form.append('tricycle')
#         elif i == 49:
#             labels_form.append(('road'))
#         elif i == 50:
#             labels_form.append('siderwalk')
#         elif i == 65:
#             labels_form.append(('traffic_cone'))
#         elif i == 66:
#             labels_form.append('road_pile')
#         elif i == 67:
#             labels_form.append(('fence'))
#         elif i == 81:
#             labels_form.append('traffic_light')
#         elif i == 82:
#             labels_form.append(('pole'))
#         elif i == 83:
#             labels_form.append('traffic_sign')
#         elif i == 84:
#             labels_form.append(('wall'))
#         elif i == 85:
#             labels_form.append('dustbin')
#         elif i == 86:
#             labels_form.append(('billboard'))
#         elif i == 97:
#             labels_form.append('building')
#         elif i == 113:
#             labels_form.append('vegatation')
#     labels_form_re = []
#     for i in labels_form:
#         if i not in labels_form_re:
#             labels_form_re.append(i)
#     num_obj = len(labels_form_re)
#     print(labels_form_re,num_obj)


# mask = np.zeros([10, 4, 3],dtype=np.uint8)
# print(mask)

# import os
# import sys
# import random
# import math
# import re
# import time
# import numpy as np
# import cv2
# import matplotlib
# import matplotlib.pyplot as plt
# import json
# from mrcnn.config import Config
# from mrcnn import utils
# import mrcnn.model as modellib
# from mrcnn import visualize
# from mrcnn.model import log
#
#
# import yaml
# from PIL import Image
#
# # Root directory of the project
# ROOT_DIR = os.path.abspath("../../")
#
# # Import Mask RCNN
# sys.path.append(ROOT_DIR)
#
# # Directory to save logs and trained model
# MODEL_DIR = os.path.join(ROOT_DIR, "logs")
#
# iter_num=0
#
# # Local path to trained weights file
# COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# # Download COCO trained weights from Releases if needed
# if not os.path.exists(COCO_MODEL_PATH):
#     utils.download_trained_weights(COCO_MODEL_PATH)
#
#
# class ShapesConfig(Config):
#     """Configuration for training on the toy shapes dataset.
#     Derives from the base Config class and overrides values specific
#     to the toy shapes dataset.
#     """
#     # Give the configuration a recognizable name
#     NAME = "shapes"
#
#     # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
#     # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
#     GPU_COUNT = 1
#     IMAGES_PER_GPU = 1
#
#     # Number of classes (including background)
#     NUM_CLASSES = 1 + 22  # background + 3 shapes
#
#     # Use small images for faster training. Set the limits of the small side
#     # the large side, and that determines the image shape.
#     IMAGE_MIN_DIM =800
#     IMAGE_MAX_DIM =1600
#
#     # Use smaller anchors because our image and objects are small
#     RPN_ANCHOR_SCALES = (4 * 6, 8 * 6, 16 * 6,32 * 6, 64 * 6)  # anchor side in pixels
#
#     # Reduce training ROIs per image because the images are small and have
#     # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
#     TRAIN_ROIS_PER_IMAGE = 32
#
#     # Use a small epoch since the data is simple
#     STEPS_PER_EPOCH =30
#
#     # use small validation steps since the epoch is small
#     VALIDATION_STEPS = 5
#
#
# config = ShapesConfig()
# config.display()
#
# class DrugDataset(utils.Dataset):
#
#     # 解析labelme中得到的yaml文件，从而得到mask每一层对应的实例标签
#     def from_json_get_labels_form(self,image_id):
#         info = self.image_info[image_id]
#         with open(info['json_path']) as f:
#             data = json.load(f)
#         data_info = data['objects']
#         count_list = []
#         labels_form = []
#         for i in range(len(data_info)):
#             count_list.append(data_info[i]['label'])
#         for i in count_list:
#             if i == 17:
#                 labels_form.append('sky')
#             elif i == 33 or i == 161:
#                 labels_form.append(('car'))
#             elif i == 34 or i == 162:
#                 labels_form.append('motorbicycle')
#             elif i == 35 or i == 163:
#                 labels_form.append(('bicycle'))
#             elif i == 36 or i == 164:
#                 labels_form.append('person')
#             elif i == 37 or i == 165:
#                 labels_form.append(('rider'))
#             elif i == 38 or i == 166:
#                 labels_form.append('truck')
#             elif i == 39 or i == 167:
#                 labels_form.append(('bus'))
#             elif i == 40 or i == 168:
#                 labels_form.append('tricycle')
#             elif i == 49:
#                 labels_form.append(('road'))
#             elif i == 50:
#                 labels_form.append('siderwalk')
#             elif i == 65:
#                 labels_form.append(('traffic_cone'))
#             elif i == 66:
#                 labels_form.append('road_pile')
#             elif i == 67:
#                 labels_form.append(('fence'))
#             elif i == 81:
#                 labels_form.append('traffic_light')
#             elif i == 82:
#                 labels_form.append(('pole'))
#             elif i == 83:
#                 labels_form.append('traffic_sign')
#             elif i == 84:
#                 labels_form.append(('wall'))
#             elif i == 85:
#                 labels_form.append('dustbin')
#             elif i == 86:
#                 labels_form.append(('billboard'))
#             elif i == 97:
#                 labels_form.append('building')
#             elif i == 113:
#                 labels_form.append('vegatation')
#         labels_form_re=[]
#         for i in labels_form:
#             if i not in labels_form_re:
#                 labels_form_re.append(i)
#         num_obj=len(labels_form_re)
#         return labels_form_re,num_obj
#     # 重新写draw_mask
#     def draw_mask(self, num_obj, mask, image,image_id):
#         #print("draw_mask-->",image_id)
#         #print("self.image_info",self.image_info)
#         info = self.image_info[image_id]
#         #print("info-->",info)
#         #print("info[width]----->",info['width'],"-info[height]--->",info['height'])
#         for index in range(num_obj):
#             for i in range(info['width']):
#                 for j in range(info['height']):
#                     #print("image_id-->",image_id,"-i--->",i,"-j--->",j)
#                     #print("info[width]----->",info['width'],"-info[height]--->",info['height'])
#                     at_pixel = image.getpixel((i, j))
#                     if at_pixel==255:
#                         mask[j, i, index] = 0
#                     else:
#                         mask[j, i, index] = 1
#         return mask
#
#     # 重新写load_shapes，里面包含自己的自己的类别
#     # 并在self.image_info信息中添加了path、mask_path 、yaml_path
#     # yaml_pathdataset_root_path = "/tongue_dateset/"
#     # img_floder = dataset_root_path + "rgb"
#     # mask_floder = dataset_root_path + "mask"
#     # dataset_root_path = "/tongue_dateset/"
#     def load_shapes(self, count, width,height,img_floder, mask_floder, imglist, dataset_root_path):
#         """Generate the requested number of synthetic images.
#         count: number of images to generate.
#         height, width: the size of the generated images.
#         """
#         # Add classes
#         self.add_class("shapes", 1, "sky")
#         self.add_class("shapes", 2, "car")
#         self.add_class("shapes", 3, "motorbicycle")
#         self.add_class("shapes", 4, "bicycle")
#         self.add_class("shapes", 5, "person")
#         self.add_class("shapes", 6, "rider")
#         self.add_class("shapes", 7, "truck")
#         self.add_class("shapes", 8, "bus")
#         self.add_class("shapes", 9, "tricycle")
#         self.add_class("shapes", 10, "road")
#         self.add_class("shapes", 11, "siderwalk")
#         self.add_class("shapes", 12, "traffic_cone")
#         self.add_class("shapes", 13, "road_pile")
#         self.add_class("shapes", 14, "fence")
#         self.add_class("shapes", 15, "traffic_light")
#         self.add_class("shapes", 16, "pole")
#         self.add_class("shapes", 17, "traffic_sign")
#         self.add_class("shapes", 18, "wall")
#         self.add_class("shapes", 19, "dustbin")
#         self.add_class("shapes", 20, "billboard")
#         self.add_class("shapes", 21, "building")
#         self.add_class("shapes", 22, "vegatation")
#
#         for i in range(count):
#             # 获取图片宽和高
#
#             filestr = imglist[i].split(".")[0]
#             mask_path = mask_floder + "/" + filestr + ".png"
#             json_path = mask_floder + "/" + filestr + ".json"
#
#             # cv_img = cv2.imread(img_floder +"/"+ filestr + ".jpg")
#
#             self.add_image("shapes", image_id=i, path=img_floder + "/" + imglist[i],
#                             width=width, height=height, mask_path=mask_path,json_path=json_path)
#             #
#             # self.add_image("shapes", image_id=i, path=img_floder + "/" + imglist[i],
#             #                width=width, height=height, mask_path=mask_path,json_path=json_path)
#
#     # 重写load_mask
#     def load_mask(self, image_id):
#         """Generate instance masks for shapes of the given image ID.
#         """
#         global iter_num
#         info = self.image_info[image_id]
#         # count = 1  # number of object
#         img = Image.open(info['mask_path'])
#         labels_form_re,num_obj = self.from_json_get_labels_form(image_id)
#         mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
#         mask = self.draw_mask(num_obj, mask, img,image_id)
#         occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
#         for i in range(num_obj- 2, -1, -1):
#             mask[:, :, i] = mask[:, :, i] * occlusion
#             occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
#         class_ids = np.array([self.class_names.index(s) for s in labels_form_re])
#         return mask, class_ids.astype(np.int32)
#
# def get_ax(rows=1, cols=1, size=8):
#     """Return a Matplotlib Axes array to be used in
#     all visualizations in the notebook. Provide a
#     central point to control graph sizes.
#
#     Change the default size attribute to control the size
#     of rendered images
#     """
#     _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
#     return ax
#
# #基础设置
# dataset_root_path="/media/mt/Seagate Backup Plus Drive/road01_ins/train_data/"
# img_floder = dataset_root_path + "pic"
# mask_floder = dataset_root_path + "cv2_mask"
# #yaml_floder = dataset_root_path
# imglist = os.listdir(img_floder)
# count = len(imglist)
from PIL import Image

# with open("/home/mt/桌面/未命名文件夹/170908_073030202_Camera_6.json") as f:
#     data = json.load(f)
# data=data["objects"]
# label_list=[]
# polygons=[]
# for i in range(len(data)):
#     label_list.append(data[i]['label'])
#     polygons.append(data[i]['polygons'][0])
# print(label_list)
# print(polygons)
# img =Image.open("/home/mt/桌面/Mask_RCNN-master/train_data/cv2_mask/170908_073030202_Camera_5.png")
# at_pixel = img.getpixel((1234, 1234))
# print(at_pixel)
#
# b=[]
# a=np.zeros((5,4,3))
# for i in range(3):
#     b.append(a[:,:,i])
# print(a)
# print(np.shape(a))
# print("=============================")
# print(b)
# print(np.shape(b))


# polygons_list=[]
# for i in range(len(data)):
#     label_list.append(data[i]['label'])
#     polygons.append(data[i]['polygons'][0])
# for i in range(len(label_list)):
#     if label_list[i]==33 or label_list[i]==161:
#         labels_form.appe# img =Image.open("/media/mt/Seagate Backup Plus Drive/road01_ins/train_data/cv2_mask/170908_073030202_Camera_5.png")
# #
# # mask=np.zeros((24,24), dtype=np.uint8)
# # l=[]
# # l_list=[]
# # for i in range(24):
# #     for j in range(24):
# #         at_pixel = img.getpixel((i,j))
# #         mask[j,i]=at_pixel
# # print(mask)
# # for i in range(len(mask)):
# #     l.extend(mask[i])
# # for i in l:
# #     if i not in l_list:
# #         l_list.append(i)
# # print(l_list)
# # print(np.shape(l))
# # print(np.min(l))
# # print(np.max(l))
#
#
# # with open("/home/mt/桌面/未命名文件夹/170908_073030202_Camera_6.json") as f:
# #     data = json.load(f)
# # data=data["objects"]
# # label_list=[]
# # labels_form=[]
# # polygons=[]nd('car')
#         polygons_list.append(polygons[i])
#     elif label_list[i] == 34 or label_list[i] == 162:
#         labels_form.append('motorbicycle')
#         polygons_list.append(polygons[i])
#     elif label_list[i]== 35 or label_list[i] == 163:
#         labels_form.append('bicycle')
#         polygons_list.append(polygons[i])
#     elif label_list[i] == 36 or label_list[i] == 164:
#         labels_form.append('person')
#         polygons_list.append(polygons[i])
#     elif label_list[i] == 37 or label_list[i] == 165:
#         labels_form.append('rider')
#         polygons_list.append(polygons[i])
#     elif label_list[i] == 38 or label_list[i] == 166:
#         labels_form.append('truck')
#         polygons_list.append(polygons[i])
#     elif label_list[i] == 39 or label_list[i] == 167:
#         labels_form.append('bus')
#         polygons_list.append(polygons[i])
#     elif label_list[i] == 40 or label_list[i] == 168:
#         labels_form.append('tricycle')
#         polygons_list.append(polygons[i])
# num_obj=len(labels_form)
# print(labels_form)
# print(polygons_list[1])
#
# for index in range(num_obj):
#     for i,p in enumerate(polygons_list[index]):
#         rr,cc=p[0],p[1]
#         print(rr,cc)
#     print("=================================")

# a=np.zeros([4,5,3])
# a[3,4,2]=1
# print(a)

# img =Image.open("/home/mt/桌面/Mask_RCNN-master/train_data/cv2_mask/170908_073030202_Camera_5.png")
# at_pixel = img.getpixel((1234, 1234))
# print(at_pixel)
#
# b=[]
# a=np.zeros((5,4,3))
# for i in range(3):
#     b.append(a[:,:,i])
# print(a)
# print(np.shape(a))
# print("=============================")
# print(b)
# print(np.shape(b))

#
# img =Image.open("/media/mt/Seagate Backup Plus Drive/road01_ins/Label/Record085/Camera 5/170908_081815992_Camera_5_instanceIds.png")
# count=[]
# car_list=[]
# for i in range(3384):
#     for j in range(2710):
#         count.append(img.getpixel((i, j)))
# for i in count:
#     if i not in car_list:
#         car_list.append(i)
# print(car_list)
# with open("/media/mt/Seagate Backup Plus Drive/road01_ins/Label/Record085/Camera 5/170908_081815992_Camera_5.json") as f:
#     data = json.load(f)
# data=data["objects"]
# label_list=[]
# for i in range(len(data)):
#     label_list.append(data[i]['label'])
# print(label_list)
#


# polygons_list=[]
# for i in range(len(data)):
#     label_list.append(data[i]['label'])
#     polygons.append(data[i]['polygons'][0])
# for i in range(len(label_list)):
#     if label_list[i]==33 or label_list[i]==161:
#         labels_form.appe# img =Image.open("/media/mt/Seagate Backup Plus Drive/road01_ins/train_data/cv2_mask/170908_073030202_Camera_5.png")
# #
# # mask=np.zeros((24,24), dtype=np.uint8)
# # l=[]
# # l_list=[]
# # for i in range(24):
# #     for j in range(24):
# #         at_pixel = img.getpixel((i,j))
# #         mask[j,i]=at_pixel
# # print(mask)
# # for i in range(len(mask)):
# #     l.extend(mask[i])
# # for i in l:
# #     if i not in l_list:
# #         l_list.append(i)
# # print(l_list)
# # print(np.shape(l))
# # print(np.min(l))
# # print(np.max(l))
# b=[]
# a=[255, 33015, 33016, 33019, 33014, 33012, 33013, 33009, 33010, 33000, 33011, 39000, 33001, 33002, 33017, 33003, 33004, 33008, 33005, 65535, 33007, 33006, 36002, 33018, 36000, 36001]
# for i in a:
#     if 33000<=i<=33999:
#         b.append(i)
# num_obj=len(b)
# print(np.ones((10)))

with open("/media/mt/Seagate Backup Plus Drive/road01_ins/Label/Record016/Camera 5/170908_062145788_Camera_5.json") as f:
    data = json.load(f)
data = data["objects"]
label_list = []
labels_form = []
# polygons = []
# polygons_list = []
for i in range(len(data)):
    label_list.append(data[i]['label'])
for i in range(len(label_list)):
    if label_list[i] == 33:
        labels_form.append(33000 + i)
num_obj = len(labels_form)
print(labels_form)
print(num_obj)