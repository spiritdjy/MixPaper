# Large-Scale Hierarchical Text Classification with Recursively Regularized Deep Graph-CNN
[toc]

## ABSTRACT
- 将文本转换为词语的图，然后使用基于图的卷积处理，这样可以捕捉非直觉的、长距离的语义
- 正则化标签之间的依赖关系结构

## 1 INTRODUCTION
- RNN对短句以及词语级语法语义更有效，对于长文档使用HAN类似结构，层级RNN假设文档和句子有一个天然的边界
- CNN近似于n-grams
- 对文档级主题，其序列信息可能不及其在语言模型、情感分析中重要，对于主题分类来说，关键词、短语以及他们的组合是非常重要的，比起连续的信息，非连续短语以及长距离词语依赖对计算语义更加重要，如餐厅、菜单、三明治等几个词可以出现不在一个窗口内，但是其决定了主题是食物
- 本文提出 Hierarchically Regularized Deep Graph-CNN (HR-DGCNN)
  - 输入：将文档转为图
    - 可以基于窗口共现构造图，每个节点使用词向量进行表示
  - 

## 2 RELATED WORK
### 2.1 Traditional Text Classification

### 2.2 Deep Learning for Text Classification

## 3 DOCUMENTS AS GRAPHS

### 3.1 Word Co-occurrence Graph

### 3.2 Sub-graph of Words

### 3.3 Graphs of Embeddings

## 4 HIERARCHICALLY REGULARIZED DEEP GRAPH-CNN
### 4.1 Convolutional Layers

### 4.2 Fully Connected and Output Layers

### 4.3 Recursive Regularization

### 4.4 Recursive Hierarchical Segmentation

## 5 EXPERIMENTS
### 5.1 Datasets and Evaluation Metrics

### 5.2 Methods for Comparison

### 5.3 Experimental Settings

### 5.4 Performance on RCV1

### 5.5 Performance on NYTimes

### 5.6 Time Consumption

## 6 CONCLUSIONS
