# 自然语言处理论文阅读笔记

[toc]

## Word Segment
- [Long Short-Term Memory Neural Networks for Chinese Word Segmentation](https://www.aclweb.org/anthology/D15-1141/)

## Dependency Parser
Dependency paths identify semantic
relations – e.g., for protein interaction
[Erkan et al. EMNLP 07, Fundel et al. 2007, etc.]

Danqi Chen, and Christopher D. Manning. "A Fast and Accurate
Dependency Parser using Neural Networks." EMNLP. 2014.
Kuebler, Sandra, Ryan McDonald, and Joakim Nivre. “Dependency parsing.” Synthesis Lectures on Human Language Technologies 1.1 (2009): 1-127.

A neural dependency parser
[Chen and Manning 2014]

Greedy choice of attachments guided by good machine learning classifiers
MaltParser (Nivre et al. 2008)

Leading to SyntaxNet and the Parsey McParseFace model
https://research.googleblog.com/2016/05/announcing-syntaxnet-worlds-most.html


[Universal Dependencies: http://universaldependencies.org/ ;
cf. Marcus et al. 1993, The Penn Treebank, Computational Linguistics]

A Neural graph-based dependency parser
[Dozat and Manning 2017; Dozat, Qi, and Manning 2017]


## Word Representation
### Word2Vector
- [Distributed Representations of Words and Phrases and their Compositionality]
- [https://papers.nips.cc/paper/7368-on-the-dimensionality-of-word-embedding.pdf]
- Improving Word Representations Via Global Context And Multiple Word Prototypes (Huang et al. 2012)
- [Mikolov et al., 2013] Mikolov, T., Chen, K., Corrado, G., and Dean, J. (2013). Efficient estimation of word representations in vector space. CoRR, abs/1301.3781.
- [Rong, 2014] Rong, X. (2014). word2vec parameter learning explained. CoRR, abs/1411.2738.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and
Kristina Toutanova. 2018. Bert: Pre-training of deep
bidirectional transformers for language understanding.

### GLOVe
- Jeffrey Pennington, Richard Socher,
and Christopher D. Manning. 2014.
GloVe: Global Vectors for Word Representation


## Sentence Representation


## Keyphrase Extraction
### [REVIEW 201905 A review of keyphrase extraction](resources/notes/d0001/keyphrase_2019_A_Review_of_Keyphrase_Extraction.md)

### [EmbedRank++ 2018 ACL Simple Unsupervised Keyphrase Extraction using Sentence Embeddings](resources/notes/d0001/keyphrase_2018_simple_unsupervisd_keyphrase_embedding.md)    
![](resources/images/d0001/071945141708511.png)
使用sent2vec将正文以及被选短语进行编码，然后采用MMR依次对被选短语进行筛选



## MT
Guillaume Klein, Yoon Kim, Yuntian Deng, Jean
Senellart, and Alexander M Rush. 2017. Opennmt:
Open-source toolkit for neural machine translation.
In arXiv preprint arXiv:1701.02810.





