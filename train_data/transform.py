import os
dataset_root_path="/home/mt/桌面/Mask_RCNN-master/train_data"
img_floder = dataset_root_path + "/pic"
mask_floder = dataset_root_path + "/cv2_mask"
img_list=os.listdir(img_floder)
from PIL import Image
count=0
# for i in range(len(img_list)):
#     filestr = img_list[i].split(".")[0]
#     mask_path = mask_floder + "/" + filestr + "_instanceIds.png"
#     img_path=img_floder+"/"+filestr+".jpg"
#
#     img=Image.open(img_path)
#     resize_img=img.resize((1024,1024))
#     resize_img.save(dataset_root_path+"/re_pic/"+filestr+".jpg")
#     count+=1
#     print(count)


for i in range(len(img_list)):
    filestr = img_list[i].split(".")[0]
    mask_path = mask_floder + "/" + filestr + "_instanceIds.png"
    img_path=img_floder+"/"+filestr+".jpg"

    img=Image.open(mask_path)
    resize_img=img.resize((1024,1024))
    resize_img.save(dataset_root_path+"/re_cv2_mask/"+filestr+"_instanceIds.png")
    count+=1
    print(count)