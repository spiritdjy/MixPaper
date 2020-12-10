# CV - 目标检测

<!-- TOC -->

- [CV - 目标检测](#cv---目标检测)
  - [[YOLOv3算法]](#yolov3算法)

<!-- /TOC -->

## [YOLOv3算法]
![](https://pic3.zhimg.com/80/v2-f870fcb490036841017903a9a3f6030a_1440w.jpg)
- 在训练过程中对于每幅输入图像，YOLOv3会预测三个不同大小的3D tensor，对应着三个不同的scale。设计这三个scale的目的就是为了能够检测出不同大小的物体。在这里我们以13x13的tensor为例做一个简单讲解。对于这个scale，原始输入图像会被分成分割成13x13的grid cell，每个grid cell对应着3D tensor中的1x1x255这样一个长条形voxel。255这个数字来源于(3x(4+1+80))，其中的数字代表bounding box的坐标，物体识别度（objectness score），以及相对应的每个class的confidence
- 如果训练集中某一个ground truth对应的bounding box中心恰好落在了输入图像的某一个grid cell中（如图中的红色grid cell），那么这个grid cell就负责预测此物体的bounding box，于是这个grid cell所对应的objectness score就被赋予1，其余的grid cell则为0。此外，每个grid cell还被赋予3个不同大小的prior box。在学习过程中，这个grid cell会逐渐学会如何选择哪个大小的prior box，以及对这个prior box进行微调（即offset/coordinate）。但是grid cell是如何知道该选取哪个prior box呢？在这里作者定义了一个规则，即只选取与ground truth bounding box的IOU重合度最高的哪个prior box
  - 三个预设的不同大小的prior box，但是这三个大小是怎么计算得来的呢？作者首先在训练前，提前将COCO数据集中的所有bbox使用K-means clustering分成9个类别，每3个类别对应一个scale，这样总共3个scale。这种关于box大小的先验信息极大地帮助网络准确的预测每个box的offset/coordinate，因为从直观上，大小合适的box将会使网络更快速精准地学习

- YOLOv3的网络模型结构图，此结构主要由75个卷基层构成，卷基层对于分析物体特征最为有效。由于没有使用全连接层，该网络可以对应任意大小的输入图像。此外，池化层也没有出现在YOLOv3当中，取而代之的是将卷基层的stride设为2来达到下采样的效果，同时将尺度不变特征传送到下一层。除此之外，YOLOv3中还使用了类似ResNet和FPN网络的结构，这两个结构对于提高检测精度也是大有裨益
![](https://pic4.zhimg.com/80/v2-fb8b964727ccfea93345ba1361c4c8a3_1440w.jpg)

- Scales：更好地对应不同大小的目标物体
通常一幅图像包含各种不同的物体，并且有大有小。比较理想的是一次就可以将所有大小的物体同时检测出来。因此，网络必须具备能够“看到”不同大小的物体的能力。并且网络越深，特征图就会越小，所以越往后小的物体也就越难检测出来。SSD中的做法是，在不同深度的feature map获得后，直接进行目标检测，这样小的物体会在相对较大的feature map中被检测出来，而大的物体会在相对较小的feature map被检测出来，从而达到对应不同scale的物体的目的。
然而在实际的feature map中，深度不同所对应的feature map包含的信息就不是绝对相同的。举例说明，随着网络深度的加深，浅层的feature map中主要包含低级的信息（物体边缘，颜色，初级位置信息等），深层的feature map中包含高等信息（例如物体的语义信息：狗，猫，汽车等等）。因此在不同级别的feature map中进行检测，听起来好像可以对应不同的scale，但是实际上精度并没有期待的那么高。
在YOLOv3中，这一点是通过采用FPN结构来提高对应多重scale的精度的
![](https://pic3.zhimg.com/80/v2-2794a0cd1c59e7c4e9293ee757d91872_1440w.jpg)

- 替换softmax层：对应多重label分类
Softmax层被替换为一个1x1的卷积层+logistic激活函数的结构。使用softmax层的时候其实已经假设每个输出仅对应某一个单个的class，但是在某些class存在重叠情况（例如woman和person）的数据集中，使用softmax就不能使网络对数据进行很好的拟合。



