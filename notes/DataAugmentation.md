# Data Augmentation
[toc]

## NLP
#### [EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification ](../resources/notes/d0001/DataAugNLP_201901_EDA__Easy_Data_Augmentation_Techniques_for_Boosting_Performance_onText_Classification_Tasks.md)
- https://arxiv.org/abs/1901.11196
    - 随机删除一个词
    - 随机选择一个词，用它的同义词替换
    - 随机选择两个词，然后交换它们的位置
    - 随机选择一个词，然后随机选择一个它的近义词，然后随机插入句子的任意位置

![](../resources/images/d0001/00201250123204322501.png))

理解：
- 通过四种操作改变以前的向量表示，引入了噪声，避免了过拟合
- 通过引入同义词，引入了新词汇，一定程度上解决了验证时模型碰到训练时所没有碰到的词语的情况
- 操作并没有改变语义，但是改变了分类前的句子表示，因此在一定程度上扩充了数据

##### []
- IBM 提出基于语言模型的数据增强新方法 https://blog.csdn.net/jdbc/article/details/103105633