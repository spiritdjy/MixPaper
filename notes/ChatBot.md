# ChatBot
[toc]

### Task-Oriented
#####  [202003 Recent Advances and Challenges in Task-oriented Dialog System](../resources/notes/d0001/chatbot_202003_Recent_Advances_and_Challenges_in_Task_oriented_Dialog_System.md)
- https://arxiv.org/pdf/2003.07490.pdf

### Dialogue
#####  [202001 Towards a Human-like Open-Domain Chatbot]()
- https://arxiv.org/abs/2001.09977

##### [201911 DialoGPT: Large-Scale Generative Pre-training for Conversational Response Generation](../resources/notes/d0001/chatbot_201911_DialoGPT__Large_Scale_Generative_Pre_training_for_Conversational_Response.md)
- https://arxiv.org/abs/1911.00536?context=cs.LG

##### [2019 ACLOne Time of Interaction May Not Be Enough: Go Deep with an  Interaction-over-Interaction Network for Response Selection in Dialogues]
- https://www.aclweb.org/anthology/P19-1001
- https://github.com/chongyangtao/IOI
Retrieval-based Chat主流的发展方向，进一步提高Context aware的answer ranking的效果。相比于之前一次性的交互，文章设计了一个interaction-block对每一个context中的句子都与response进行了多次交互（有点类似于MRC对于document和query进行反复重构的思路），并且在不同轮数、不同level的交互之上用RNN进行了轮数之间的抽象，更加强调对于context关系的（自动）推断

##### [Incremental Transformer with Deliberation Decoder for Document Grounded Conversations]
- https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/P19-1002
微信团队的工作，文章针对的是有document ref的对话系统，每轮有对应document（轮之间doc是否一样不做限制），从使用的数据上看，document提供的是wiki页面，提供了一些需要回答的内容信息。

提出的Deliberation Network在整个架构中贡献比较大，主要考虑是用context生成主句主干和方向（第一步），用document来修正knowledge（第二步）。

这也跟对话问题本身一对多的关系比较match，毕竟很多时候context并不一定能完全限定具体可以回答那个主题

##### [Improving Multi-turn Dialogue Modelling with Utterance ReWriter]
- https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/P19-1003
文章主要关注使用context和query提到的信息进行rewrite，是的，跟copynet/pointer-generator的思路比较接近。文章提出的模型中有一个lambda选择copy的时候更多关注context还是query（验证也是加了context好了一点）；把pointer和transformer一起使用（transformer又带来了比较多的提升）。

文章使用了线上的CPS作为测试（NRG上线的优势，text generation过于依赖人工评测的问题急需解决啊）；另外文章测试rewriter的时候，把这个过称加入了到意图识别中，把用户的话借鉴前面的词重写，改善意图识别的准确性，这个思路也蛮不错的。

不过现场也有同行问到没有回报英文上结果的问题，同事解释到英文上效果不如中文理想，在中文中pointer成功的可能性更高一些（英文中的说法变化更多一些）

##### [Do Neural Dialog Systems Use the Conversation History Effectively? An Empirical Study]
https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/P19-1004
一篇实验性的文章，还蛮有意思，尤其是结果对于相关工作者是用借鉴和思考的意义的，毕竟很多思考需要实验的支撑。文章实验了各种修改context（sentence level + word level, shuffle、reverse、drop、truncate）来对于不同生成式模型的影响，一些操作猛如虎的，从影响应该可以看一下context起到的作用，以及不同数据集合上的一些特点。今年有不少这样类似的工作，相比于一些工作还是来的有点意思

##### [Constructing Interpretive Spatio-Temporal Features for Multi-Turn Responses Selection]
https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/P19-1006

做多轮回复排序的工作，直接把answer selection作为了一个目标，从里面直接优化ground truth。虽然大家都在用P@k、MAP等指标评价回复排序问题，但是更多的都是在用Point-wise、Pair-wise再做优化，那引入List-wise是非常直接且肯定有效的想法。那之前大家为什么不这样做，一方面是list-wise的开销相对比较大，需要比之前更大的显存（现在大家应该都有了）；另一方面只要优化目标类似，模型上的提升已经能显现出来。

同时文章还提了4D的交互，其实有点类似文章1给context之间更多的交互和判断，来提高对于回复质量的排序。总体来说，文章进一步提升了回复排序的效果

##### [A Large-Scale Corpus for Conversation Disentanglement]
https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/P19-1374
这篇工作做的是一个数据集，但是有意思的是根据标注结果的反应的一些现象。文章标注了7w+的message（主要来自Ubuntu数据，加了一些同主题数据），主要是回复关系的标注（一个session下，某句话是回复的哪句话，因为通常一个长对话里往往隐藏了很多个小对话），还做了一些实验，验证了ubuntu的质量。

文章指出他们发现20%的对话是优质的，58%缺少了一些信息，3%包含了其他对话的部分，19%存在以上两个问题。“not be learning from accurate human dialogues.”。另外，94.9%的ubuntu对话在2分钟内，全部都在一小时以内。88.3%的对话不超过8句，99.4%的对话不超过100句。其实这里就与之前很多Ubuntu上的数据整理方式有一定的出入。

文章的数据还是不足够大，用来做test是很好的资源了（虽然几万做test也有点浪费，或者说哪一些做validation吧），但是作为训练是远远不够的。不过这里也引发了一些思考，怎么样整理质量足够高的context、选择合适长度的context用于NRG、识别哪些context是真正对当前对话有效的是不是比构建一个更复杂的、参数更多的模型来的更有意义也更有效果呢？

##### [Pretraining Methods for Dialog Context Representation Learning]
https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/P19-1373

虽然大家都在用context信息，但是基本是的思路都是各种交互后的自动选择和语义融合，作者为context的pretrain设定了一些目标：
1. context可以retrival出来ground-truth reply
2. context可以generate出来reply
3. mask一句context，可以retrival
4. mask一句可以识别哪一句被mask了（也就是h会更多包涵当前context信息？）

不同的目标对于不同的下游任务适用程度会有不同，但基本都有提升。尤其对于少量数据finetune/multidomain的transfer都有帮助。也是一篇有助于推动多轮对话的探索性工作

##### [Learning a Matching Model with Co-teaching for Multi-turn Response Selection in Retrieval-based Dialogue Systems]
https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/P19-1370
有点Teacher-Student的意思，但是是交替作为两个角色中的一个。主要思想是利用正负例之间的差，一个模型只需要比另一个模型的差大就好。purpose引用一下文章中的话：

Intuitively, one model may assign a small margin to a negative example if it identifies the example as a false negative. Then, its peer model will pay less attention to such an example in its optimization. This is how the two peer models help each other combat with noise under the strategy of teaching with dynamic margins.

一方面，数据中本身有些噪声，尤其是负例（一般是随机的，但是容易随机到一些合适的回复，当数据比较大的时候还是挺多的），而ML的训练目标一致是坚持一致的，pair-wise的话就是不管是不是noise都要正例得分优于负例一个阈值，而dynamic的这种方式就会降低这些case的要求（其实这些case本身就需要高一点的分数）；另一方面，随机的抽取有些是对于模型很容易区分的，例如可能随机出这样的负例“Q: 今天天气真好；A: 我超爱吃这家的Pizza。”，对于模型其实非常初期就学习到了，能区分开了，那teacher模型会对student给这个case以比较小的weight去学习，有点像focal loss，不知道会不会比focal loss效果更明显一些，毕竟加上本身动态的loss，不同阶段的要求会不同。Focal loss在ranker问题上感觉还是不是很稳定好用

##### [Improving Neural Conversational Models with Entropy-Based Data Filtering]
https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/P19-1567
文章提出了一种NRG数据的清洗方式，实际做NRG的任务中数据的清洗当属关键的步骤，作者针对这个问题做了一些常识，可能也是很多人都尝试过的，但是做的会更加成体系一些。这个文章也是蛮早就挂在了arxiv的。
作者提到：
one-to-many problem，dialogue要解决的问题；
many-to-one problem，dialogue想要避免的问题。
针对这个情况，作者给句子做了聚类，如果通过一个qa之间的关系来计算概率分布，从而计算entropy。也就是通过聚类找到类似的，general的句子。那么直接从数据中找多对一的句子，然后聚类找相似可能效果也接近（只是猜测）。因此这个工作也依赖聚类效果，例如文章中提报了在twitter更加发散的场景下效果就不如DailyDialog理想了

##### [Are Training Samples Correlated? Learning to Generate Dialogue Responses with Multiple References]
https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/P19-1372
NRG中一对多的关系是其不同于其他文本生成任务的最主要区别，这篇文章针对这个问题提出来一种新的建模流程。

作者认为虽然query-response是一对多的关系，但是语义空间上这些不同的回复应该有着一个语义中心，那么这个学习过程就可以变为第一步学习这个语义的中心点，第二步学习一个中心点到具体特定回复的偏移。

第一步即然要学习中心点，那么就需要一次性考虑一个query对应的所有response，作者提出了一种loss function和机制可以来优化这个问题。

第二步就比较直接，有了语义中心的表示c，融入到CVAE框架就可以做NRG了。这个思路个人感觉是比较make sense的，我们也在做类似的方向，但是感觉第二步还是有些奇怪，因为问题变成了given语义中心和query，加一个噪声，是怎么确定到一个specific response来提升效果的呢？当然效果说话的话，结果是更好了，不过解释性可能还有很多工作可以做，但是感觉方向是很好的，还有很多可以做，值得follow

##### Neural Response Generation with Meta-Words
https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/P19-1538
同样是NRG的工作，文章提出了一种整体化的memory框架，不管是句子长度、Action、Copy行为、多句话组成、Specificity都是memory的一部分，也就是题目中的Meta-Words。既然是memory就有读写操作（针对不同的已知类型有不同的操作，又有统一的信息存储方式），把memory结合state就可以引入到NRG model中了。把各种肯定会影响回复生成的信息一体化融入是有必要的，现在更多的工作都是针对某一个点去优化的。Btw，今天share的文章拍照技术都有待提高

##### Towards Empathetic Open-domain Conversation Models: a New Benchmark and Dataset
https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/P19-1534
有意思的task配合上质量有保证的新数据集都是值得关注的，这篇文章就属于这个类型。带有情感的对话一直是大家蛮关注的一个场景，文章对数据做了足够的分析和介绍，试验了检索式和生成式不同的结果


Constructing Interpretive STM for Responses Selection
Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels
http://link.zhihu.com/?target=https%3A//arxiv.org/abs/1804.06872


##### Attention-informed mixed-language training for zero-shot cross-lingual task-oriented dialogue systems

### TODO

- 基于深度学习的开放领域对话系统研究综述
  - http://cjc.ict.ac.cn/online/onlinepaper/42-7-1-201974192124.pdf

- 从文本中构建领域本体技术综述
  - http://cjc.ict.ac.cn/online/onlinepaper/rfl-2019319125830.pdf


