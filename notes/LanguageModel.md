# 语言模型
[toc]

### [201911 Generalization through Memorization: Nearest Neighbor Language Models](../resources/notes/d0001/lm_201911_knn_lm.md)
- https://arxiv.org/abs/1911.00172
![](../resources/images/d0001/132011421113503.png)
在TranformLM的基础上通过叠加KNN相似度进行下一个词语的预测，提高了性能
原理：在某些train出现较少，test出现与train上文非常相似的情况下改进性能
同时文章比较了LM+全记忆LM, LM+n-grams均没有knn-LM的效果好