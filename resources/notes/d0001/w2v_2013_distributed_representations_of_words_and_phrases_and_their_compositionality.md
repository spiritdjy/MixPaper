# Distributed Representations of Words and Phrases and their Compositionality
[toc]

- http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf

## Abstract
- 最新的skip-gram能学习到大量词语之间语法和语义之间的关系
- 对高频词语进行下采样，提高训练速度以及更多的常规单词表示regular word representation
- 采用负采样替代层次softmax: 计算成本太高
- 提出一种简单的方法来学习词组的含义

## 1 Introduction
- Skip-gram模型，这是一种从大量非结构化文本数据中学习高质量单词向量表示的有效方法
- Skip-gram模型的训练不涉及密集矩阵乘法，训练非常高效
- 生成的向量支持词类别任务

## 2 The Skip-gram Model
- 目标是使平均对数概率最大化
![](https://img-blog.csdnimg.cn/20190331152049464.png)
- Softmax: 
![](https://img-blog.csdnimg.cn/20190331152211363.png)

### 2.1 Hierarchical Softmax
- Hierarchical softmax使用输出层的二叉树表示，以W个单词作为叶节点，并且对于每个节点，显式地表示其子节点的相对概率。它们定义了一个随机walk，将概率分配给单词。
 ![](https://img-blog.csdnimg.cn/20190331152552236.png)

### 2.2 Negative Sampling
- 替代hierarchical softmax最大值的方法是噪声对比估计(NCE)
- 负抽样(NEG)目标
![](https://img-blog.csdnimg.cn/20190331153001669.png)
- 实验表明，在5~20范围内的k值对于小的训练数据集是有用的，而对于大的数据集，k值可以小到2.5
- NCE和NEG均将噪声分布Pn(w)作为一个自由参数。我们研究了Pn(w)的多种选择，发现单克分布U(w)提高到3/4次方(即。U(w)3/4/Z)在NCE和NEG的每个任务上(包括语言建模，这里没有报告)都明显优于unigram和均匀分布

### 2.3 Subsampling of Frequent Words
- 在非常大的语料库中，最频繁的单词很容易出现数亿次(例如In、the和a)。这些词通常提供的信息价值比罕见的词少
- 例如，虽然Skip-gram模型从观察法国和巴黎的共同出现中获益，但是从观察法国和the的频繁共同出现中获益要少得多，因为几乎每个单词都经常在一个句子中与the同时出现。这个想法也可以用在相反的方向;经过对数百万个例子的训练，频繁词的向量表示没有显著变化。
- 为了解决罕见词和频繁词之间的不平衡，一种简单的子抽样方法:将训练集中的每个单词wi丢弃，由如下公式计算概率
![](https://img-blog.csdnimg.cn/20190331153223298.png)
f(wi)是单词wi的频率，t通常是一个选择的阈值，通常在10 - 5之间。将频率比t大的单词进行子样本，同时保持频率的排序。当f(wi) > t时，该值为正，否则为负数，f(wi) 越大，该值越大

## 3 Empirical Results
![](https://img-blog.csdnimg.cn/20190331153241991.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzk5MzI1MQ==,size_16,color_FFFFFF,t_70)
- 任务有两大类:句法类比(如quick: fast:: slow: slow)和语义类比(如country to capital city relationship)
- 负采样在类比推理任务上优于Hierarchical softmax最大值，甚至比噪声对比估计性能稍好。对常用词进行子抽样后，训练速度提高了几倍，单词表示的准确性显著提高

## 4 Learning Phrases
![](https://img-blog.csdnimg.cn/20190331153806976.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzk5MzI1MQ==,size_16,color_FFFFFF,t_70)
- 许多短语的意思并不是由单个单词的意思简单组合而成的
- 要学习短语的向量表示，我们首先要找到经常出现在一起的单词，以及不经常出现在其他上下文中的单词。例如，“New York Times”和“Toronto Maple Leafs”在训练数据中被独特的令牌所替代
- 理论上，我们可以使用所有的n-gram来训练Skip-gram模型，但是那样会占用太多内存
- 这里使用简单的方法来获取短语
![](https://img-blog.csdnimg.cn/20190331153900989.png)
δ是作为discounting系数和防止太多的非常罕见的词组成的短语。得分高于所选阈值的bigram将用作短语
通常，运行2-4遍阈值递减的训练数据，允许形成由**多个单词**组成的更长的短语

### 4.1 Phrase Skip-Gram Results
- 向量维数300和上下文窗口大小5
- 结果表明，当k = 5时，负抽样也能达到令人满意的精度，而当k = 15时，则能获得更好的性能。令人惊讶的是，虽然我们发现Hierarchical softmax最大值在不进行子采样的情况下训练时性能较差，但当我们对频繁出现的单词进行降采样时，它成为了性能最好的方法。这表明，子抽样可以导致更快的训练，也可以提高准确性，至少在某些情况下
![](https://img-blog.csdnimg.cn/2019033115403371.png)
![](https://img-blog.csdnimg.cn/20190331154202130.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzk5MzI1MQ==,size_16,color_FFFFFF,t_70)
- 似乎短语的最佳表示是通过Hierarchical softmax最大值和子抽样模型学习的

## 5 Additive Compositionality
- 证明了由Skip-gram模型学习到的单词和短语表示具有线性结构，这使得使用简单的向量算术进行精确的类比推理成为可能
- 发现跳跃符号表示呈现出另一种线性结构，这种结构使得通过元素明智地添加单词的向量表示来有意义地组合单词成为可能
![](https://img-blog.csdnimg.cn/20190331154207709.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzk5MzI1MQ==,size_16,color_FFFFFF,t_70)
- 通过对训练目标的考察，可以解释向量的可加性。字向量与软极大非线性的输入呈线性关系。通过训练单词向量来预测句子中周围的单词，向量可以被看作是单词出现时上下文的分布。这些值与输出层计算的概率呈对数关系，因此两个词向量的和与两个上下文分布的乘积有关。乘积在这里作为AND函数:两个单词向量都赋予高概率的单词将具有高概率，而其他单词将具有低概率。因此，如果伏尔加河与俄语、River这两个词频繁出现在同一个句子中，那么这两个词向量的和就会得到一个与伏尔加河向量非常接近的特征向量

## 6 Comparison to Published Word Representations
![](https://img-blog.csdnimg.cn/20190331154706552.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzk5MzI1MQ==,size_16,color_FFFFFF,t_70)
- 在大型语料库上训练的Skip-gram模型在学习表示的质量上明显优于所有其他模型
- 虽然训练集要大得多，但是Skip-gram模型的训练时间只是以前模型体系结构所需时间复杂度的一小部分

## 7 Conclusion
- 使用Skip-gram模型训练单词和短语的分布式表示，并证明这些表示具有线性结构，使得精确的类比推理成为可能
- 成功地在比以前发布的模型多几个数量级的数据上训练了模型。这使得学习到的单词和短语表示的质量有了很大的提高，特别是对于罕见的实体。我们还发现，对经常出现的单词进行子抽样，不仅训练速度更快，而且对不常见单词的表示也明显更好
- 另一个贡献是负采样算法，它是一种非常简单的训练方法，可以学习准确的表示，特别是对于频繁出现的单词
- 影响性能的最关键的决策是模型体系结构的选择、向量的大小、子采样率和训练窗口的大小
- 向量这个词可以用简单的向量加法有意义地组合起来。本文提出的另一种学习短语表示的方法是用单个标记简单地表示短语。这两种方法的组合提供了一种强大而简单的方法来表示较长的文本，同时具有最小的计算复杂度
