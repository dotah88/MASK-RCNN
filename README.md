# Mask-rcnn  
Mask rcnn为实例检测分割网络,此处将其应用与百度Apollo无人驾驶数据集,数据集为车辆每10米拍摄一次的不同路段图像数据,分为左右摄像头,标签为像素级标注.  

1.数据处理    
训练数据和标签样例存于train_data文件夹,因为原始数据标签和实际网络输入需要标签标注方式不同,所以需要对其进行重新标注,并且网络为实例级分割所以需要对  
每个实例生成一个Mask,所以最终得到的图像矩阵为图像中实例个数个Mask,每个Mask的行和列为当前图像的长宽像素标记值,依照这样的方式对原始图像进行处理生成
新的Mask矩阵.  

2.网络参数调整  
处理完输入数据需要对相应的网络参数和代码进行调整,将类别设置为自己数据集的类别并相应的调整网络参数,此处参数较多,主要注意的是RPN_ANHCHO_SCALES参数应设置  
为32的整数倍, 因为mask rcnn用FPN网络提取的多个金子塔形状的特征图,每一层增幅为2的整数倍;IMAGES_PER_GPU不能太大,两张1024*1024需要12G显存空间以及IMAGE_RESIZE_MODE设置图片调整模式'square'(正方形)或者'crop'随机挑选(只用与训练),调整好相应参数.  
3.网络训练  
使用在coco数据集预训练好的网络参数进行训练,因为采用restnet101网络的实例分割,所以实例较多的情况会占用大量显存空间,此处我把原图向resize成1024*1024每次训练一张 
几轮迭代后显存还是会溢出(Nvidia 1060 6G),下面为正常训练后的测试情况
![image](https://github.com/dotah88/Mask-rcnn/blob/master/image/index.png)
![image](https://github.com/dotah88/Mask-rcnn/blob/master/image/index1.png)
![image](https://github.com/dotah88/Mask-rcnn/blob/master/image/index2.png)
