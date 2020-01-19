# Hierarchically-Refined Label Attention Network for Sequence Labeling
[toc]

论文链接: https://arxiv.org/abs/1908.08676
源码链接: https://github.com/Nealcly/BiLSTM-LAN

https://zhuanlan.zhihu.com/p/91031332

## Abstract
- 在深度学习时代，很多情况下 BiLSTM-CRF 并没有比不对输出序列进行建模的 BiLSTM-softmax 取得更好的效果。一个可能的原因是神经网络编码器已经有很强的序列信息编码能力，在此基础上基于马尔科夫假设的 CRF 并没有引入更多的有效信息

## 1 Introduction
- 在一些研究中显示，BiLSTM-CRF的效果并不比BILSTM-Softmax好，可能的原因是神经网络的表征能力已经足够强，能够捕捉到远程的标签依赖关系(CRF的功能)，这样本地局部预测(local prediction)的效果也足够好（理解：多层与循环，导致其信息自动学习到部分信息）
- CRF被其马尔科夫假设所限制，同样由于维特比算法，其计算消耗较大

## 2 Related Work


## 3 Baseline

### 3.1 Word Representation Layer

### 3.2 Sequence Representation Layer

### 3.3 Inference Layer

## 4 Label Attention Network

### 4.1 Label Representation

### 4.2 BiLSTM-LAN Layer

### 4.3 Training

### 4.4 Complexity

### 4.5 BiLSTM-LAN and BiLSTM-softmax

## 5 Experiments


## 6 Discussion
