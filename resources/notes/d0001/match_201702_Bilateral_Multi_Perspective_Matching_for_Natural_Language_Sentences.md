# Bilateral Multi-Perspective Matching for Natural Language Sentences
[toc]

https://arxiv.org/pdf/1702.03814.pdf

## 1 Introduction
- Siamese：基本思想为输入的两个橘子用同一个处理网络处理到同一个embedding space，再做判断。
- matching-aggregation/compare-aggregate：为了解决siamese里没有引入两个句子交叉信息的问题，对两个句子先做各种形式的信息交叉，再通过对齐模型，将融合信息用于后续处理

主要关注点
- 大多之前的matching-aggregation框架都是问题对齐答案或者答案对齐问题，即单向的，本文做了双向（Bilateral）的对齐
- 对齐模型的修改，作者在如何对齐两个句子向量上做了多种尝试，提出Multi-Perspective
## 

