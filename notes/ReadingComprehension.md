# 阅读理解
[toc]

## 综述
#### [201812 Neural Reading Comprehension and Beyond]()

#### [201907 Neural Machine Reading Comprehension: Methods and Trends](http://arxiv.org/abs/1907.01118)
机器阅读理解（MRC），要求机器根据给定的背景回答问题，在过去几年中，深度学习的出现引起了越来越广泛的关注。 尽管基于深度学习的MRC研究正在蓬勃发展，但缺乏一篇全面的调查文章来总结所提出的方法和最近的趋势。 因此，我们对这一充满希望的领域最新的研究工作进行了全面的概述。 具体而言，我们比较不同维度的MRC任务并介绍一般架构。 我们进一步提供了流行模型中使用的最先进方法的分类。 最后，我们讨论了一些新的趋势，并通过描述该领域的一些开放性问题得出结论。

https://www.optbbs.com/thread-4667058-1-1.html

## 开放领域
#### [201911 Knowledge Guided Text Retrieval and Reading for Open Domain Question Answering]()
 - https://arxiv.org/pdf/1911.03868.pdf


#### [201611 BiDAF Bi-Directional Attention Flow For Machine Comprehension]

#### Reading Wikipedia to Answer Open-Domain Questions
 “基于维基百科的开放性QA系统”阅读笔记    https://zhuanlan.zhihu.com/p/92361851


#### SQUAD的rnet复现踩坑记 https://www.cnblogs.com/rocketfan/p/9103878.html



https://github.com/MurtyShikhar/Question-Answering
https://github.com/robinjia/adversarial-squad
https://github.com/hitvoice/DrQA
https://github.com/aswalin/SQuAD
https://github.com/YerevaNN/R-NET-in-Keras
https://github.com/kamalkraj/BERT-SQuAD


---
 ACL 2019
#### [Open-Domain Why-Question Answering with Adversarial Learning to Encode Answer Texts]
https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/P19-1414
这篇工作蛮自然的把GAN用在的QA问题上，作者选定的是why相关的问题，我想可能是因为why这部分的回答更需要多文章的理解吧，一些简单的问题可能可以直接通过词语的类型就能判断了。直接上图了，先验部分，已知question和passage，讲道理是能固定答案相关的语义的；那另一方面后验概率已知answer和question，也是能固定一个表示。先验作为fake，后验作为real，一个天然的GAN的问题形成了。可能这方面的文章关注的不够，个人觉着还是挺有意思，至少非常自然

#### [Natural Questions: a Benchmark for Question Answering Research]
https://link.zhihu.com/?target=https%3A//storage.googleapis.com/pub-tools-public-publication-data/pdf/1f7b46b5378d757553d3e92ead36bda2e4254244.pdf
Google的工作，文章提出了一个新的QA数据，SQuAD之后这两年已经不缺少各式各样的花式QA数据了，不过作者的出发点是源自于现在数据标注存在的一个很有意思的问题。首先一个问题的回答是不好严格界定的，多一些内容少一些内容都可以作为一个合适的回答，但是既然作为ground truth是要严格保证最优是最理想的，但是作者评估了标注结果，从表格上来看，有一半的标注是不严格的，甚至16%的短answer是错的。然后文章引入了史无前例的25-way cross labeling，从结果上来看QA问题在较少标注的情况下准确率是非常堪忧的，另外图3作者也做贝叶斯优化的角度分析了为什么直观上大家都认同的，一个人标注是一个不好的Upper Bound，同理如果多个人效果也不理想依然不能很好的指导模型朝正确的方法发展。所以除了一个更优质（更多人参与标注）的数据集，这些发现的本身也值得思考

#### Compositional Questions Do Not Necessitate Multi-hop Reasoning
https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/P19-1416

这又是一篇引发思考的工作，今年蛮多觉着有意思的都是此类工作，大家都关心的、直觉上多少认可的、缺乏实证的实验工作。文章把full-wiki上做的QA认为是必须要multi-hop的终极例子，然后现有的single-hop的任务是有点的，那multi-hop QA的需求从单文档到全文档就是一个递进的过程。这些背景不太关键，主要的是作者认为很多multi-hop的任务都是不需要multi-hop的技术来做的，为什么呢，作者分析了四种情况：a) 需要multi-hop；b) 干扰项少；c) 提供的很多信息是冗余的；d) Non-compositional 1-hop。可以看到虽然非multi-hop的问题不多，但是真正必须要使用推理才能回答的问题并不多，这一点非常符合现在大家对于doc QA任务上的理解，很多时候不需要multi-hop因为问的内容文章里只有一个（例如，when/who/which government）。作者取了部分数据分析，只有27%真正需要multi-hop。也许这才是值得思考的，如何用multi-hop解决好multi-hop该解决的问题也许才是让模型更好的学习的通道，完全的一视同仁的end2end也许很难说模型学到的是些什么内容

#### Improving Question Answering over Incomplete KBs with Knowledge-Aware Reader
https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/P19-1417
使用结构化的文本和非结构化的文本（抽取结构化文本时的context）来提高QA任务的效果。文章把KB和text放到一起，text使用KB reformulated的query来建模，这样其实会更关注KB里面的内容，理解是提供一些KB之外的context信息（如果两个分支有区分的话）。从试验中w/o没有直接使用KB的结果看也是这样



























