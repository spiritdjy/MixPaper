# 文本匹配
[toc]

##### [201908 ACL RE2: Simple and Effective Text Matching with Richer Alignment Features](resources/notes/d0001/match201908_ACL_RE2__Simple_and_Effective_Text_Matching_with_Richer_Alignment_Features.md)
- https://arxiv.org/abs/1908.00300
![](resources/images/d0001/512006091108602.png)

##### [201804 SAN - Stochastic Answer Networks for Natural Language Inference](resources/notes/d0001/match_201804_Stochastic_Answer_Networks_for_Natural_Language_Inference.md)
- https://arxiv.org/pdf/1804.07888.pdf
![](resources/images/d0001/482006351408602.png)

##### [201609 ESIM Enhanced LSTM for Natural Language Inference](resources/notes/d0001/match_2016_Enhanced_LSTM_for_Natural_Language_Inference.md)
- https://arxiv.org/abs/1609.06038
![](resources/images/d0001/01201300217207503002.png)
- input encoding:
  - 使用BiLSTM对输入序列（前提和假设）进行编码，也可以使用Tree LSTM
- local inference modeling
  - 使用注意力机制来软对齐前提与假设之间的词语
  - 基于相似性矩阵计算注意力并基于注意力计算每个词语的软对齐词语
  - 进一步的增强，计算差值以及点积
- inference composition
  - 使用BiLSTM层来对上层的输入分别进行信息的组合
  - 为了控制复杂性，可以使用一个单层RELU的前馈神经网络对输入进行降维
  - 使用平均与最大pooling将变长的序列输入变成固定长度向
  - 分类器进行分类

##### [201606 A Decomposable Attention Model for Natural Language Inference](resources/notes/d0001/match_201606_A_Decomposable_Attention_Model_for_Natural_Language_Inference.md)
- https://arxiv.org/abs/1606.01933v2

##### [2013 DSSM Learning Deep Structured Semantic Models for Web Search using Clickthrough Data](resources/notes/d0001/match_2013_Learning_Deep_Structured_Semantic_Models_for_Web_Search_using_Clickthrough_Data.md)

##### [201503 Convolutional Neural Network Architectures for Matching Natural Language Sentences]()
- https://arxiv.org/abs/1503.03244



##### [201702 Bilateral Multi-Perspective Matching for Natural Language Sentences](resources/notes/d0001/match_201702_Bilateral_Multi_Perspective_Matching_for_Natural_Language_Sentences.md)
- https://arxiv.org/abs/1702.03814


