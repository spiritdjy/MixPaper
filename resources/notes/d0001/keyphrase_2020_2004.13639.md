<!-- TOC -->

- [Joint Keyphrase Chunking and Salience Ranking with BERT](#joint-keyphrase-chunking-and-salience-ranking-with-bert)
  - [ABSTRACT](#abstract)
  - [INTRODUCTION](#introduction)
  - [METHODOLOGY](#methodology)

<!-- /TOC -->
# Joint Keyphrase Chunking and Salience Ranking with BERT
- https://arxiv.org/abs/2004.13639
- https://github.com/thunlp/bert-kpe


## ABSTRACT
- JointKPE 使用组块网络来识别高质量的短语，并使用排名网络来了解它们在文档中的显著性

## INTRODUCTION
- KPE 最近采用的预先训练的语言模型，例如 ELMo Peters 等人(2018年) ，主要利用语境化嵌入到大块高质量短语; 显著性排名更依赖于频率信号 Xiong 等人(2019年)。这有利于在文件中频繁出现的带有头脑的短语，并可能隐含地偏向于较短的短语，而在许多情况下，长尾的短语和较长的短语也传达了文件的代表性信息

- 多任务情境下联合学习组块自包含短语并估计它们的显著性的方法。从 BERT 表示出发，JointKPE 使用 CNN 组合 n-gram embedding，一个用于识别高质量短语的组块网络，以及一个用于选择文档中最突出短语的排名网络。在学习过程中，组块网络使用短语水平损失来表示 n-grams 的意义，排名网络使用学习排序损失来估计突显度，并将两个任务结合起来平衡短语质量和突显度

- JointKPE 依靠提取长的和非实体的关键短语而蓬勃发展，这对以前的 KPE 技术是一个挑战

- 消融研究证实了 JointKPE 的有效性，主要归因于多任务学习中关键词质量和显著性的联合评估


## METHODOLOGY
- 将文档 d = { w 1，... ，w i，... ，w n }作为输入，并学习从文档中自包含的显著单元 n-gram 中提取键短语 p
- 通过两个主要组成部分实现的: 一个组块网络，识别有意义的 n-grams，和一个排名网络，分配突出分数的短语

