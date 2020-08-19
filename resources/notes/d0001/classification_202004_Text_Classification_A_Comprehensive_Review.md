# Deep Learning Based Text Classification: A Comprehensive Review

[toc]

- https://arxiv.org/abs/2004.03705

## ABSTRACT

## 1.INTRODUCTION
- 目的是为文本单元（例如句子，查询，段落和文档）分配标签（labels）或标签（tags）
- 题回答，垃圾邮件检测，情感分析，新闻分类，用户意图分类，内容审核
- 基于规则的方法
  - 对领域有深入的了解，并且系统难以维护
- 基于机器学习（数据驱动）的方法
  - 具有更高的可伸缩性，并且可以应用于各种任务
- 混合方法
- 经典的基于机器学习的模型
  - 第一步是从文档（或任何其他文本单元）中提取一些手工制作的特征
  - 第二步将这些特征输入到分类器中以进行分类。做个预测
  - 单词袋（BoW）及其扩展名
  - 分类算法的流行选择包括朴素贝叶斯，支持向量机（SVM），隐马尔可夫模型（HMM），梯度增强树和随机森林
  - 局限性
    - 依赖于手工制作的特征需要繁琐的特征工程和分析才能获得良好的性能
    - 设计功能对领域知识的高度依赖使得该方法难以轻松地推广到新任务
    - 由于特征（或特征模板）是预定义的，因此这些模型无法充分利用大量训练数据。


### 1.1.TEXT CLASSIFICATION TASKS
- 情感分析
  - 目的是分析人们在文本数据（例如产品评论，电影评论和推文）中的观点，并提取其极性和观点。情感分类可以是二元问题，也可以是多分类问题。二进制情感分析是将文本分为阳性和阴性两类，而多类情感分析则侧重于将数据分为细粒度标签或多级强度
- 新闻分类
  - 新闻分类系统可以帮助用户实时获取感兴趣的信息。识别新兴新闻主题并根据用户兴趣推荐相关新闻是新闻分类的两个主要应用
- 主题分类
  - 主题分析试图通过识别文本主题来自动从文本中获取含义。主题分类是主题分析最重要的组成技术之一。主题分类的目的是为每个文档分配一个或多个主题，以使其更易于分析
- 问题解答（QA）
  - 分为两种：提取式和生成式
  - 可以视为文本分类的一种特殊情况。给定一个问题和一组候选答案，我们需要将每个候选答案分类为正确与否
- 自然语言推断（NLI）
  - 识别文本蕴含（RTE），可预测是否可以从另一文本推断出文本的含义。特别是，系统需要为每对文本单元分配一个标签，例如包含，矛盾和中性
  - 释义是NLI的一种广义形式，也称为文本对比较。任务是测量一个句子对的语义相似性，以确定一个句子是否是另一个句子的释义
### 1.2.PAPER STRUCTURE

## 2.DEEP LEARNING MODELS FOR TEXT CLASSIFICATION

### 2.1.FEED-FORWARD NEURAL NETWORKS
前馈神经网络
- 将文本视为一袋单词
- word2vec或Glove嵌入模型学习向量表示，将嵌入的向量和或平均值作为文本的表示，通过多层感知器（MLP），然后使用分类器（例如逻辑回归，朴素贝叶斯或SVM）对最终层的表示形式进行分类
- 深度平均网络（DAN）
- fastText
- doc2vec，它使用一种无​​监督算法来学习可变长度文本（例如句子，段落和文档）的定长特征表示
  - 附加的段落标记通过矩阵映射到段落向量d。在doc2vec中，此向量与三个单词的上下文的串联或平均值用于预测第四个单词。段落向量表示当前上下文中缺少的信息，可以用作该段落主题的记忆。在训练之后，将段落矢量用作该段落的特征（例如，代替BoW或除BoW之外），并馈入分类器进行预测


### 2.2.RNN-BASED MODELS


### 2.3.CNN-BASED MODELS


### 2.4.CAPSULE NEURAL NETWORKS


### 2.5.MODELS WITH ATTENTION MECHANISM


### 2.6.MEMORY-AUGMENTED NETWORKS


### 2.7.TRANSFORMERS


### 2.8.GRAPH NEURAL NETWORKS


### 2.9.SIAMESE NEURAL NETWORKS


### 2.10.HYBRID MODELS


### 2.11.BEYOND SUPERVISED LEARNING


## 3.TEXT CLASSIFICATION DATASETS

### 3.1.SENTIMENT ANALYSIS DATASETS


### 3.2.NEWS CLASSIFICATION DATASETS

### 3.3.TOPIC CLASSIFICATION DATASETS

### 3.4.QA DATASETS


### 3.5.NLI DATASETS


## 4.EXPERIMENTAL PERFORMANCE ANALYSIS

### 4.1.POPULAR METRICS FOR TEXT CLASSIFICATION


### 4.2.QUANTITATIVE RESULTS


## 5.CHALLENGES AND OPPORTUNITIES


## 6.CONCLUSION

