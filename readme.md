## YOLOv5算法训练跌倒检测模型

#### 一、详细训练步骤

##### 1、yolov5模型下载并解压：

下载地址：https://github.com/ultralytics/yolov5

解压：<img src="D:\notebook\apply\img\9X[UO0EOVIV6CYLQMJGEN]0.png" style="zoom: 67%;" />

##### 2、数据准备

（1）在yolov5/data文件下创建如下文件目录

<img src="D:\notebook\apply\img\dirt.png" style="zoom:67%;" />

将所有的图片放到JPEGImages和images文件夹下，将所有voc格式的xml文件放入到Annotations文件夹下 。在根目录下创建make_txt.py文件，代码如下，运行代码后ImageSets中生成数据集分类txt文件。

```python
import os
import random
trainval_percent = 0.1
train_percent = 0.9
xmlfilepath = 'data/Annotations'
txtsavepath = 'data/ImageSets'
total_xml = os.listdir(xmlfilepath)
num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)
ftrainval = open('data/ImageSets/trainval.txt', 'w')
ftest = open('data/ImageSets/test.txt', 'w')
ftrain = open('data/ImageSets/train.txt', 'w')
fval = open('data/ImageSets/val.txt', 'w')
for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftest.write(name)
        else:
            fval.write(name)
    else:
        ftrain.write(name)
ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
```

![](D:\notebook\apply\img\set.png)

(2)根目录下继续创建 voc_label.py文件，代码如下：需要注意的是，sets中改为本次sets的名字（make_txt生成的） classes修改为你需要检测的类别，在本案例中，我们只需要检测stand和fall两种类别。

```python
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
sets = ['train', 'test', 'val']
classes = ['stand', 'fall']
def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)
def convert_annotation(image_id):
    in_file = open('data/Annotations/%s.xml' % (image_id))
    out_file = open('data/labels/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
wd = getcwd()
print(wd)
for image_set in sets:
    if not os.path.exists('data/labels/'):
        os.makedirs('data/labels/')
    image_ids = open('data/ImageSets/%s.txt' % (image_set)).read().strip().split()
    list_file = open('data/%s.txt' % (image_set), 'w')
    for image_id in image_ids:
        list_file.write('data/images/%s.jpg\n' % (image_id))
        convert_annotation(image_id)
    list_file.close()
```

运行以上代码后，可以发现生成了voc格式的标签文件labels（显示数据集的具体标注数据），并且在data文件下出现了train、val、test的txt文件，保存了图片的路径。（带有图片的路径）

<img src="D:\notebook\apply\img\data-label.png" style="zoom:67%;" />

<img src="D:\notebook\apply\img\train-txt.png" style="zoom:67%;" />

<img src="D:\notebook\apply\img\label.png" style="zoom:67%;" />



##### 3、修改配置文件

（1）修改coco.yaml文件

这里的yaml和以往的cfg文件是差不多的，但需要配置一份属于自己数据集的yaml文件。复制data目录下的coco.yaml，我这里命名为My_train.yaml 主要修改三个地方：

<img src="D:\notebook\apply\img\my_train.png" style="zoom:67%;" />

a. 修改train,val,test的路径为自己刚刚生成的路径

b. nc 里的数字代表数据集的类别，这里有stand和fall类，所以修改为2

c. names 里为自己数据集标注的类名称，这里是"stand"和”fall"

(2)修改model.yaml文件

models下有四个模型，smlx需要训练的时间依次增加，按照需求选择一个文件进行修改即可。这里修改了yolov5m.yaml，只需要将nc的类别修改为自己需要的即可。

<img src="D:\notebook\apply\img\model.png" style="zoom:67%;" />

##### 4、修改train.py并开始训练

weights，yaml，data按照自己文件的路径修改， epochs迭代次数自己决定，这里仅用9次进行测训练。batch-size过高可能会影响电脑运行速度，根据自己电脑硬件条件决定增加还是减少修改完成，运行开始训练。

<img src="D:\notebook\apply\img\train.png" style="zoom:67%;" />

##### 6、训练过程

（1）网络结构

<img src="D:\notebook\apply\img\network.png" alt="network" style="zoom: 50%;" />

（2）训练epoch

<img src="D:\notebook\apply\img\epoch.png" style="zoom:67%;" />

(3)训练结束

<img src="D:\notebook\apply\img\result.png" alt="result" style="zoom:67%;" />

#### 二、训练结果

##### 1、相关训练文件

<img src="D:\notebook\apply\img\result_img.png" style="zoom:67%;" />

##### 2、模型文件

<img src="D:\notebook\apply\img\result_model.png" alt="result_model" style="zoom:67%;" />

#### 三、遇到的问题

##### 1、内存爆炸

主要的原因在于batch-size的大小问题，将batch-size设置小一点就能将问题解决。

##### 2、指定路径不存在相应文件

主要的原因在于没有将训练图片放入data文件下的images文件夹。训练时读取的图片文件夹不是自己创建的JPEGImages文件夹。