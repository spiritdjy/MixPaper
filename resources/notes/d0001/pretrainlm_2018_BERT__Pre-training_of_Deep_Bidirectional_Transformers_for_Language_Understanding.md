# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
[toc]

- https://arxiv.org/pdf/1810.04805.pdf
- https://zhuanlan.zhihu.com/p/54853340

## Abstract
- Bidirectional Encoder Representations from Transformers
- BERT，它代表来自transformer的双向编码器表示
- BERT旨在通过在所有层中联合调节左右上下文来预先训练深度双向表示
- 预训练好的BERT表示可以通过一个额外的输出层进行微调，从而为各种任务(如问答和语言推理)创建最新的模型，而无需对特定于任务的架构进行实质性修改

## 1 Introduction
- 语言模型预训练已经证明对改善许多自然语言处理任务是有效的
- 有两种将预训练的语言表示应用于下游任务的现有策略:基于特征和微调。
    - 基于特征的方法，如ELMo使用特定于任务的体系结构，将预训练表示作为额外特征。
    - 微调方法，如生成式预训练transformer（openai gpt），引入了最小的特定于任务的参数，并通过简单地微调预先训练的参数来对下游任务进行训练。
    - 在以前的工作中，这两种方法在训练前都有相同的目标函数，它们使用单向语言模型来学习通用语言表达。
- 当前的技术严格限制了预训练表示的能力，特别是微调方法。主要的限制是标准语言模型是单向的，这限制了可以在预训练期间使用的架构的选择
    - 每个词只能关注transformer self-attention层中的先前词。这种限制对于句子级别的任务来说不是最佳的，当对SQuAD之类的词级别任务应用基于微调的方法时，这种限制可能会带来毁灭性的影响，其中从两个方向应用语言模型非常重要
- 通过transformer提出BERT :双向编码器表示来改进基于微调的方法
    - 新的预训练对象masked language model( MLM )，来解决上面提到的单向约束，灵感来自完形填空任务
    - 引入了一个“next sentence prediction”任务，共同预训练文本对表示
    - 预训练的表示消除了许多精心设计的任务特定体系结构的需求。BERT是第一个可以在大量句子级和词级任务上实现最先进的性能，优于许多具有任务特定体系结构的系统的基于微调的表示模型

## 2 Related Work
### 2.1 Feature-based Approaches
- 预训练词嵌入被认为现代nlp系统的重要部分，提供了显著的性能提升
- Elmo提出在语言模型中提取context-sensitive特征，得到语境相关的词嵌入

### 2.2 Fine-tuning Approaches
- 语言模型中迁移学习最近的趋势是去预训练一些模型架构，然后根据监督下游任务微调模型参数。这种方式的优势是要调整的参数很少
- OpenAI GPT

### 2.3 Transfer Learning from Supervised Data
- 无监督学习的优势是基本可以获取无限的数据，有监督的迁移学习任务也被证明很有效

## 3 BERT





















