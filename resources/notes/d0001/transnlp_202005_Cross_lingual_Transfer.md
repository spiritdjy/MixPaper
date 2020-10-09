# Cross-lingual Transfer of Twitter Sentiment Models Using a Common Vector Space

[toc]

- https://arxiv.org/pdf/2005.07456.pdf
### 简单笔记
- 两种方法进行嵌入空间映射
  - 从一个语言的词向量映射到另外一个语言的词向量
  - 将两个语言映射到公共的词向量空间，词语有对其的时候
  - 通过对其语言的词向量空间，可以进行机器学习模型的迁移
- 本文主要是采用推特的13种语言情感分析语料集进行试验，跨语言的情感分析效果
  - 两种方法
    - 模型迁移
    - 在训练时扩充目标语言实例
  - 相似语言之间的模型转移是明智的，而数据集扩展并没有提高预测性能

- 原理
  - 现代单词嵌入空间在各种语言中也表现出相似的结构
  - 意味着可以将由单语种文本资源独立产生的嵌入进行对齐，从而产生称为跨语言嵌入的通用跨语言表示形式，从而可以快速有效地集成不同语言的信息。

- 三种对其方法
  - 使用单语嵌入和双语词典的可选帮助来对齐嵌入对
    - 第一种方法将代表一种语言的单词的向量映射到另一种语言的向量空间中（反之亦然）。第二种方法将两种语言的嵌入映射到公共向量空间
    - 目标是相同的：具有相同含义的单词的嵌入必须在最终向量空间中尽可能接近
    - 监督的方法要求使用双语字典，该字典用于匹配等效单词的嵌入。嵌入使用Moore-Penrose伪逆对齐，从而使平方的欧几里德距离的总和最小
      - Mikel Artetxe, Gorka Labaka, and Eneko Agirre. 2018b. A robust self-learning method for fully unsupervised cross-lingual mappings of word embeddings. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 789–798.
    - 半监督方法使用一个小的初始种子字典
      - 使用两个嵌入的相似性矩阵来构建初始字典。此初始词典通常质量较差，但足以用于后续处理
      - 建立初始字典（通过给字典添加种子或使用相似性矩阵）后，将应用迭代算法。该算法首先使用伪逆方法为给定的初始字典计算最佳映射。然后，计算给定嵌入的最佳字典，并使用新字典重复该过程
    - 无监督方法则在没有任何双语信息
      - 对抗方法 Alexis Conneau, Guillaume Lample, Marc’Aurelio Ranzato, Ludovic Denoyer, and Hervé Jégou. 2018. Word translation without parallel data. In Proceedings of International Conference on Learning Representation ICLR 2018.
      - 使用单词的频率 Mikel Artetxe, Gorka Labaka, and Eneko Agirre. 2018b. A robust self-learning method for fully unsupervised cross-lingual mappings of word embeddings. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 789–798.

  - 使用双语对齐（可比甚至平行）的语料库来以所有涉及的语言
    - 构造的嵌入必须在公共向量空间中尽可能以不同的语言映射相同的单词
    - LASER（与语言无关的语句表示）是一个Facebook研究项目，致力于多种语言的联合语句表示
    - 使用编码器-解码器架构。编码器在大型并行语料库上进行训练，将任何语言或脚本的句子翻译成英语或西班牙语（以并行语料库中存在的形式为准）的并行句子，从而以共享的形式形成多种语言的整个句子的联合表示

  - 基于大型预训练的多语言掩蔽语言模型
    - 多语言BERT
  
  - 本文使用 第二种方法的 LASER库中实现并适用于93种语言的多种语言的联合语句表示

- 两种情感分析方法
  - 模型在源语言上进行训练，并用于目标语言中的分类-这种模型转移是可能的，因为所有相关语言的文本都嵌入到语言中
  - 使用来自其他语言的实例扩展训练集，然后在神经网络训练过程中将所有实例映射到公共向量空间

- 数据集
  - Twitter情感数据集，删除重复的推文，Web链接和主题标签清除了上述数​​据集，删除了阿尔巴尼亚语和西班牙语数据集

- 试验结果
  - 测试相同语言族的相似语言之间的预测模型转移。迁移学习模型与使用目标语言进行模型训练之间的差距为4％至20％。对于没有额外目标数据的直接模型传输，这些结果令人鼓舞
  - 在不同语言族的语言上重复对相同语言族的语言所做的实验。在这种情况下，转移不太可能成功，我们预计在这些不利条件下性能会降低
  - 使用扩展数据集的学习模型与本机模型之间的差距。这些结果表明，测试的数据集扩展不成功，即，目标语言提供的实例数量已经足以成功学习。来自其他语言的其他实例的质量可能会比本地实例低，因此会降低性能。
    - 使用其他语言的数据集来进行扩充，其他语言使用同样的向量空间，因此原则上类似增加数据集
    - 70%作为训练集，30%作为测试集
    - 对于保加利亚语和塞尔维亚语，使用多种语言和显着扩大数据集是成功的，许多语言的训练都有效
  - 比较嵌入
    - 将LASER以及mbert对单语料分别进行训练以及测试（不考虑迁移学习），使用SVM进行分类，可以得到mbert的结果更好，其词嵌入表示质量更好，但是微调和执行需要更多的计算时间

- 参考
  - 使用单语嵌入和双语词典的可选帮助来对齐嵌入对  Mikel Artetxe, Gorka Labaka, and Eneko Agirre. 2018a. Generalizing and improving bilingual word embedding mappings with a multi-step framework of linear transformations. In Thirty-Second AAAI Conference on Artificial Intelligence.
  - 使用双语对齐（可比甚至平行）的语料库来以所有涉及的语言 Mikel Artetxe and Holger Schwenk. 2019. Massively multilingual sentence embeddings for zero-shot cross-lingual transfer and beyond. Transactions of the Association for Computational Linguistics, 7:597–610.
  - 基于大型预训练的多语言掩蔽语言模型 Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186.
  - LASER库中实现并适用于93种语言的多种语言的联合语句表示 Mikel Artetxe and Holger Schwenk. 2019. Massively multilingual sentence embeddings for zero-shot cross-lingual transfer and beyond. Transactions of the Association for Computational Linguistics, 7:597–610.
  - Sogaard2019提供了跨语言方法的详细概述和分类
  - 现有单语嵌入方法的全面摘要 Mikel Artetxe, Gorka Labaka, and Eneko Agirre. 2018a. Generalizing and improving bilingual word embedding mappings with a multi-step framework of linear transformations. In Thirty-Second AAAI Conference on Artificial Intelligence.
    - https://github.com/artetxem/vecmap
