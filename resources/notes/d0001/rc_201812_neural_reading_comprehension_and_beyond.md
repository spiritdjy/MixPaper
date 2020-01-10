# Neural Reading Comprehension and Beyond
[toc]

- https://www.cs.princeton.edu/~danqic/papers/thesis.pdf
- https://chendq-thesis-zh.readthedocs.io/en/latest/

## 摘要
- 如何构建计算机系统来阅读一篇文章并回答理解问题,是评估计算机系统理解人类语言能力的一项重要任务
- 如果我们能够构建高性能的阅读理解系统，它们将成为问答和对话系统等应用的关键技术
- 目标是涵盖神经阅读理解的本质，并介绍我们在构建有效的神经阅读压缩模型方面所做的努力，更重要的是了解神经阅读理解模型实际学习了什么，以及解决当前任务需要多大的语言理解深度

## 1 Introduction
### 1.1 **Motivation**
处理文本
- 词性标注（part-of-speech tagging）
- 命名实体识别（named entity recognition）
- 语法解析（syntactic parsing）
- 指代消解

<font color=red>阅读理解可能是评估语言理解最合适的任务（reading comprehension could be the most suitable task for evaluating language understanding）</font>

阅读理解的成功可以归因于两个原因：
- 1）以（文章、问题、答案）三元组的形式的大规模监督数据集创建；
- 2）神经阅读理解模型的建立。

这篇论文
: - 现代神经阅读理解的本质：问题的形成，系统的构建模块和关键成分，以及理解当前的神经阅读理解系统在哪些方面可以做得更好，哪些方面仍然落后
  - 构建出高性能的阅读理解系统，它们将成为问答和对话系统等应用的关键技术

两个以神经阅读理解为核心的研究方向
: - 开放领域的问答（Open-domain question answering）结合了来自信息检索和阅读理解的挑战，旨在回答来自Web或大型百科全书（如Wikipedia）的一般问题
  - 会话形式问答（Conversational question answering）结合了来自对话和阅读理解的挑战，解决了在一段文本中进行多轮问答的问题，比如用户如何与会话代理进行交互

### 1.2 Thesis Outline

## 2 An Overview of Reading Comprehension
### 2.1 History
#### 2.1.1 Early Systems
- 这一阶段开发的系统主要是基于规则的词包方法
- 例如DEEP READ 系统（Hirschman et al. 1999）中进行词干分析、语义类识别和代词解析等浅层语言处理，或者像是QUARC系统（Riloff and THElen，2000）中手动生成基于词汇和语义对应的规则或者是以上两个的组合体（Charnizak et al.， 2000）。这些系统在检索正确句子时达到了30%-40%的准确率

#### 2.1.2 Machine Learning Approaches
- 研究员以（文章，问题，回答）三元组的形式收集人类标注好的训练例子，希望我们可以训练统计模型来学习将一段话和问题形成的对映射到他们相对应的答案上面去：f（passage， question）–>answer
- 两个值得注意的数据集是MCTEST （Richardson et al.， 2013）和PROCESSBANK （Berant et al.， 2014）
- 一种是启发式滑动窗口方法，它测量问题、答案和滑动窗口中单词之间的加权单词重叠/距离信息；另一种方法是通过将每个问答对转换为一个语句来运行现成的文本蕴涵系统
- 这些模型大多建立在一个简单的max-margin学习框架之上，该框架具有丰富的手工设计的语言特性，包括句法依赖、语义框架、指代消解、篇章关系和单词嵌入。MC500的性能从63%略微提高到70%左右

缺点总结如下
: 1. 这些模型严重依赖于现有的语言工具，如依赖依存解析和语义角色标记（SRL）系统。然而，这些语言表示任务还远远没有解决，现成的工具通常是从单个领域（的文章）（例如，newswire文章）训练而来，在实际使用中存在泛化问题。因此，利用现有的语言注释作为特性有时会在这些基于特性的机器学习模型中增加噪音，而更高级别的注释（例如，篇章关系与词性标记），会让情况变得更糟糕。
  2. 模拟人类水平的理解是一个难以捉摸的挑战，而且总是很难从当前的语言表征中构建有效的特征。例如，对于图1.1中的第三个问题：How many friends does Alyssa have in this story？，当证据散布在整个文章中，基本不可能构建出一个有效特征的。
  3. 尽管我们可以从人类标记的阅读理解示例中训练模型，这确实激励人心，但这些数据集仍然太小，无法支持表达性统计模型。例如，用于训练依存解析器的English Penn Treebank数据集包含39，832个示例，而在MCTEST中，用于训练的示例仅为1480个——更不用说阅读理解了，作为一项综合性的语言理解任务，阅读理解更加复杂，并且需要不同的推理能力

#### 2.1.3 A Resurgence： The Deep Learning Era
- THE ATTENTIVE READER, 基于attention机制的LSTM模型，证明它在很大程度上优于符号NLP方法
- CNN数据集: CNN和《每日邮报》附有一些要点，总结了文章中所包含的信息。他们将一篇新闻文章作为passage，通过使用一个placeholder 来替换一个实体（entity）的方式将其中的一个要点转换为一个完形填空式的问题，而答案就是这个被替换的实体。为了确保这个系统需要真正的理解文章来完成这个任务，而不是使用世界知识（译者，知识库，即符号系统）或者语言模型来回答问题，他们运行了实体识别和指代消解系统，并且将所有在指代链中提到的每个实体替换为一个抽象的实体标记（例如：@entity6，可以在Table2.1（a）中看到例子）。最后，他们几乎没有任何成本地收集了近100万个数据示例

神经阅读理解模型有几个优点
: 1. 他们不依赖于任何下游的语言学特征（比如，依存分析或者指代消解），并且所有的特征是在一个统一的端到端的的框架中独立学习来的。这避免了语言学标注的噪音，并且也在可用的特征空间中提供了更好的了灵活性。
  2. 传统的符号NLP系统受困于一个严重的问题：特征通常非常稀疏，并且泛化性非常差。例如，为了回答一个问题：
 How many individual libraries make up the main school library？
而文章中的相关内容如下
“… Harvard Library， which is the world’s largest academic and private library system， comprsing 79 individual libraries with over 18 million volumes”
所以一个系统必须基于标记好的特征来学习comprising与make up的一致性，例如下面的特征：
pwi=comprising∧qwj=make∧qwj+1=up.
这里并没有足够的数据来给这些特征赋予正确的权重。这在所有的非神经NLP模型中是一个共有的问题。利用低维，稠密的词向量共享相似词语之间在统计上的强度，可以有效的缓解稀疏性。
  3. 这些模型从从构建大量手工特征的劳动中解脱出来。因此，神经模型在概念上更简单，（研究）重点可以转移到神经结构的设计（译者注：可以从构建手工特征中解放，转而研究神经网络结构）。由于现代深度学习框架如TENSORFLOW和PYTORCH的发展，已经取得了很大的进步，现在开发新的模型又快又容易。


















