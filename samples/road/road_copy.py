import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import json
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


import yaml
from PIL import Image

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

iter_num=0

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 22  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_RESIZE_MODE = 'square'
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1600

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


config = ShapesConfig()
config.display()

class DrugDataset(utils.Dataset):
    # 得到该图中有多少个实例（物体）
    def get_obj_index(self, image):
        n = np.max(image)
        return n

    # 解析labelme中得到的yaml文件，从而得到mask每一层对应的实例标签
    def from_json_get_labels_form(self,image_id):
        info = self.image_info[image_id]
        with open(info['json_path']) as f:
            data = json.load(f)
        data_info = data['objects']
        count_list = []
        labels_form = []
        for i in range(len(data_info)):
            count_list.append(data_info[i]['label'])
        for i in count_list:
            if i == 17:
                labels_form.append('sky')
            elif i == 33 or i == 161:
                labels_form.append(('car'))
            elif i == 34 or i == 162:
                labels_form.append('motorbicycle')
            elif i == 35 or i == 163:
                labels_form.append(('bicycle'))
            elif i == 38:
                labels_form.append('truck')
            elif i == 40:
                labels_form.append(('tricycle'))
            elif i == 36 or i == 164:
                labels_form.append('person')
            elif i == 37 or i == 165:
                labels_form.append(('rider'))
            elif i == 38 or i == 166:
                labels_form.append('truck')
            elif i == 39 or i == 167:
                labels_form.append(('bus'))
            elif i == 40 or i == 168:
                labels_form.append('tricycle')
            elif i == 49:
                labels_form.append(('road'))
            elif i == 50:
                labels_form.append('siderwalk')
            elif i == 65:
                labels_form.append(('traffic_cone'))
            elif i == 66:
                labels_form.append('road_pile')
            elif i == 67:
                labels_form.append(('fence'))
            elif i == 81:
                labels_form.append('traffic_light')
            elif i == 82:
                labels_form.append(('pole'))
            elif i == 83:
                labels_form.append('traffic_sign')
            elif i == 84:
                labels_form.append(('wall'))
            elif i == 85:
                labels_form.append('dustbin')
            elif i == 86:
                labels_form.append(('billboard'))
            elif i == 97:
                labels_form.append('building')
            elif i == 113:
                labels_form.append('vegatation')
        return labels_form
    # 重新写draw_mask
    def draw_mask(self, num_obj, mask, image,image_id):
        #print("draw_mask-->",image_id)
        #print("self.image_info",self.image_info)
        info = self.image_info[image_id]
        #print("info-->",info)
        #print("info[width]----->",info['width'],"-info[height]--->",info['height'])
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    #print("image_id-->",image_id,"-i--->",i,"-j--->",j)
                    #print("info[width]----->",info['width'],"-info[height]--->",info['height'])
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        return mask

    # 重新写load_shapes，里面包含自己的自己的类别
    # 并在self.image_info信息中添加了path、mask_path 、yaml_path
    # yaml_pathdataset_root_path = "/tongue_dateset/"
    # img_floder = dataset_root_path + "rgb"
    # mask_floder = dataset_root_path + "mask"
    # dataset_root_path = "/tongue_dateset/"
    def load_shapes(self, count, img_floder, mask_floder, imglist, dataset_root_path):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("shapes", 1, "sky")
        self.add_class("shapes", 2, "car")
        self.add_class("shapes", 3, "motorbicycle")
        self.add_class("shapes", 4, "bicycle")
        self.add_class("shapes", 5, "person")
        self.add_class("shapes", 6, "rider")
        self.add_class("shapes", 7, "truck")
        self.add_class("shapes", 8, "bus")
        self.add_class("shapes", 9, "tricycle")
        self.add_class("shapes", 10, "road")
        self.add_class("shapes", 11, "siderwalk")
        self.add_class("shapes", 12, "traffic_cone")
        self.add_class("shapes", 13, "road_pile")
        self.add_class("shapes", 14, "fence")
        self.add_class("shapes", 15, "traffic_light")
        self.add_class("shapes", 16, "pole")
        self.add_class("shapes", 17, "traffic_sign")
        self.add_class("shapes", 18, "wall")
        self.add_class("shapes", 19, "dustbin")
        self.add_class("shapes", 20, "billboard")
        self.add_class("shapes", 21, "building")
        self.add_class("shapes", 22, "vegatation")

        for i in range(count):
            # 获取图片宽和高

            filestr = imglist[i].split(".")[0]
            mask_path = mask_floder + "/" + filestr + ".png"
            json_path = mask_floder + "/" + filestr + ".json"
            cv_img = cv2.imread(img_floder +"/"+ filestr + ".jpg")

            self.add_image("shapes", image_id=i, path=img_floder + "/" + imglist[i],
                            width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path,json_path=json_path)
            #
            # self.add_image("shapes", image_id=i, path=img_floder + "/" + imglist[i],
            #                width=width, height=height, mask_path=mask_path,json_path=json_path)

    # 重写load_mask
    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        global iter_num
        info = self.image_info[image_id]
        count = 1  # number of object
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img,image_id)
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        labels_form =self.from_json_get_labels_form(image_id)
        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

#基础设置
dataset_root_path="/media/mt/Seagate Backup Plus Drive/road01_ins/train_data/"
img_floder = dataset_root_path + "pic"
mask_floder = dataset_root_path + "cv2_mask"
#yaml_floder = dataset_root_path
imglist = os.listdir(img_floder)
count = len(imglist)

#train与val数据集准备
dataset_train = DrugDataset()
dataset_train.load_shapes(count,img_floder, mask_floder, imglist,dataset_root_path)
dataset_train.prepare()

#print("dataset_train-->",dataset_train._image_ids)

dataset_val = DrugDataset()
dataset_val.load_shapes(7,img_floder, mask_floder, imglist,dataset_root_path)
dataset_val.prepare()

#print("dataset_val-->",dataset_val._image_ids)

# Load and display random samples
#image_ids = np.random.choice(dataset_train.image_ids, 4)
#for image_id in image_ids:
#    image = dataset_train.load_image(image_id)
#    mask, class_ids = dataset_train.load_mask(image_id)
#    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=10,
            layers='heads')



# Fine tune all layers
# Passing layers="all" trains all layers. You can also
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=30,
            layers="all")

class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# Test on a random image
image_id = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_val, inference_config,
                           image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                            dataset_train.class_names, figsize=(8, 8))

results = model.detect([original_image], verbose=1)

r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                            dataset_val.class_names, r['scores'], ax=get_ax())