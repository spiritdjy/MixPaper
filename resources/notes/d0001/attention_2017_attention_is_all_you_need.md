# Attention Is All You Need
[toc]

PAPER: https://arxiv.org/pdf/1706.03762.pdf
CODE: [TENSOR2TENSOR](https://github.com/tensorflow/tensor2tensor)
## Abstract
- 主流序列转导模型基于复杂的循环神经网络或卷积神经网络，这些神经网络包含一个编码器和一个解码器
- 我们提出一种新的简单的网络架构Transformer，仅基于attention机制并完全避免循环和卷积

## 1 Introduction
- 循环模型通常是对输入和输出序列的符号位置进行因子计算。 通过在计算期间将位置与步骤对齐，它们根据前一步的隐藏状态ht-1和输入产生位置t的隐藏状态序列ht。这种固有的顺序特性阻碍样本训练的并行化，这在更长的序列长度上变得至关重要，因为有限的内存限制样本的批次大小
- 在各种任务中，attention机制已经成为序列建模和转导模型不可或缺的一部分，它可以建模依赖关系而不考虑其在输入或输出序列中的距离
## 2 Background
- 减少顺序计算的目标也构成扩展的神经网络GPU、ByteNet和ConvS2S的基础，它们都使用卷积神经网络作为基本构建模块、并行计算所有输入和输出位置的隐藏表示。 在这些模型中，关联任意两个输入和输出位置的信号所需的操作次数会随着位置之间的距离而增加，ConvS2S是线性增加，而ByteNet是对数增加。 这使得学习远程位置之间的依赖性变得更加困难。 在Transformer中，这种操作减少到固定的次数，尽管由于对用attention权重化的位置取平均降低了效果，但是我使用Multi-Head Attention进行抵消

## 3 Model Architecture


### 3.1 Encoder and Decoder Stacks

### 3.2 Attention

#### 3.2.1 Scaled Dot-Product Attention

#### 3.2.2 Multi-Head Attention

#### 3.2.3 Applications of Attention in our Model

### 3.3 Position-wise Feed-Forward Networks

### 3.4 Embeddings and Softmax

### 3.5 Positional Encoding

## 4 Why Self-Attention

## 5 Training

### 5.1 Training Data and Batching
### 5.2 Hardware and Schedule
### 5.3 Optimizer
### 5.4 Regularization
## 6 Results
### 6.1 Machine Translation
### 6.2 Model Variations
### 6.3 English Constituency Parsing

## 7 Conclusion