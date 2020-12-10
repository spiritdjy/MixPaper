# 深度学习 - 正则化

<!-- TOC -->

- [深度学习 - 正则化](#深度学习---正则化)
  - [Normalization](#normalization)
    - [[1805.11604 How Does Batch Normalization Help Optimization?]](#180511604-how-does-batch-normalization-help-optimization)
  - [[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift ]](#batch-normalization-accelerating-deep-network-training-by-reducing-internal-covariate-shift-)

<!-- /TOC -->

## Normalization
### [1805.11604 How Does Batch Normalization Help Optimization?]
- https://arxiv.org/abs/1805.11604

- BN 是一个广泛应用的用于快速稳定地训练深度神经网络的技术，但是我们对其有效性的真正原因仍然所知甚少。
- 输入分布的稳定性和 BN 的成功之间关系很小，BN 对训练过程更根本的影响是：它让优化更加平滑。这种平滑让梯度更加可预测更加稳定，从而加速训练。

- BN 和 internal covariate shift
  - 在原始论文 Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift 中，作者认为 BN 减小了所谓的 internal covariate shift(ICS)，这也被认为是 BN 成功的根基
  - 在有和没有 BN 的情况下分别训练了一个 VGG 网络，然后观察某一层在训练过程中的分布情况。
  ![](https://pic4.zhimg.com/80/v2-fa7eae494882f63a1b75a904c281d6e3_1440w.jpg)
  加入 BN 后训练速度和准确度都有所提升，但是某一层的分布情况却相差甚微。基于此观察，作者提出了两个问题：
    - BN 的有效性是否真的和 ICS 有关？
    - BN 让输入分布稳定是否能有效减小 ICS？

- BN 的性能提升是否来源于控制 ICS
  - BN 的中心思想就是说控制输入分布的均值和方差可以提升训练性能，怎么才能证明吗？那我们就来看看在 BN 后面引入随机噪声会发生什么  ![](https://pic3.zhimg.com/80/v2-71801a6f4fcec5572a03f651ec4f540e_1440w.jpg)
  - 噪声在每一次都是不一样的，所以就会使激活值产生严重的偏移，实验结果如下  ![](https://pic4.zhimg.com/80/v2-0ddcba121212fbc5d342e3aa8dc5520b_1440w.jpg)  引入噪声后的 BN 相比标准网络分布更加不稳定，但是在训练过程中依然表现非常好，这显然与之前的说法冲突

- BN 是否能减小 ICS
  - 如果我们将 ICS 和输入分布的均值及方差稳定性联系在一起的话，显然 ICS 和训练性能没有直接联系。但是，我们可能会怀疑：是否有一个对 ICS 更广泛的定义使其与训练性能有直接联系呢？如果是的话，BN 确实能减小 ICS 吗?
  - 原论文中关于 ICS 的定义是：网络中参数的变化引起的输入分布的变化。为了衡量前面层参数的更新而导致后面参数需要调整的程度，我们来比较每一层在前面所有层参数更新前后的梯度差异。也就有了下面的定义： ![](https://pic4.zhimg.com/80/v2-c7b55d7fca2c5f88d4acc4b96c132457_1440w.jpg)  G 和 G' 的差异就反映了输入的变化引起的优化环境相对于权重的变化。针对上面的定义，作者比较了有无 BN 的网络在训练过程中上述差异的变化情况，如下所示 ![](https://pic4.zhimg.com/80/v2-7bc35c0eb0ce2f890eb14be69d43c4e3_1440w.jpg)  配备了 BN 后，上述差异变化得反而更加剧烈，尤其是相对于没有激活函数的 DLN 网络。也即是，从优化的角度来说， BN 甚至没有减小 ICS

- BN 为什么有效
  - 除了减小 ICS，BN 还有其他很多优点：防止梯度消失爆炸、对诸如学习率和参数初始化策略的不同设置比较鲁棒、防止落入激活函数的饱和区。这些特性明显都有利于训练，但是，它们都是 BN 简单的结果而不是真正底层的原因
  - BN 的平滑效果
    - 一个函数 f 是 L-Lipschitz 的，如果它满足： ![](https://pic1.zhimg.com/80/v2-283066f864e89fe78445814649f59ee8_1440w.jpg) 非负的实数 L 是斜率的最小上确界，称之为利普希茨常数 ![](https://pic2.zhimg.com/80/v2-7ac4a726b1a89d13bf75496b7f5a1951_1440w.jpg) 也就是说，利普希茨常数限制了函数可以改变得多快。换句话说，如果没有利普希茨常数，函数可以无限快地改变，也就是不连续了
    - 将上式展开可以得到： ![](https://pic3.zhimg.com/80/v2-1b3c095dd39ff32efef6cd2b0292d7d2_1440w.jpg) 也就是固定了 x1，x2 的函数值是一个线性函数，并且被 L 约束在了一个范围内  ![](https://pic3.zhimg.com/80/v2-0639aa2c21073f3f9997da2b9b0219ea_1440w.jpg)  一个连续可微的函数是贝塔平滑的如果它的梯度满足贝塔利普希茨，也就是： ![](https://pic3.zhimg.com/80/v2-628eda0c8f828b0326703ae5bb6948da_1440w.png) 同理我们可知，贝塔平滑是限制一个函数的一阶导数变化得多快的。
    - 作者发现，BN 的关键作用是让优化过程更加平滑，也就是，损失改变得更慢而且梯度的幅值也更小。更深层次的原因则是 BN 的重新参数化让损失函数更加利普希茨更加贝塔平滑。 ![](https://pic4.zhimg.com/80/v2-c00b13b256fa43210df430915edb29e3_1440w.jpg) 引入 BN 后的损失函数就像上面的右图那样更加平滑，梯度更加可靠更加可预测，即使我们走了一大步，梯度的方向仍然是真实梯度的准确估计
  - 优化过程的探索
    - 为了展示 BN 对优化过程的稳定，作者进行了下面的实验。在训练过程中的每一步，通过移动不同的步长，我们来观察损失函数、梯度的变化情况 ![](https://pic1.zhimg.com/80/v2-0d0e54c5571e89c85cde6eda628b4854_1440w.jpg)  可以发现，引入 BN 后损失函数、梯度以及“有效”的贝塔平滑都更加稳定。![](https://pic1.zhimg.com/80/v2-63fa143157e04908e5ccb3ada0857e34_1440w.jpg)
  - BN 是最好的或者唯一的平滑优化过程的方法吗？
    - ![](https://pic4.zhimg.com/80/v2-13a4448dde5462426f18fb3cc998416b_1440w.jpg) 作者发现，用范数进行归一化也可以取得和 BN 类似的效果，BN 的有效可能只是一个偶然的发现
  - 理论分析
    - ![](https://pic1.zhimg.com/80/v2-1574572686da20f8206676a5fa478508_1440w.jpg)针对未引入 BN 的普通网络以及引入 BN 后的网络，作者推导了它们梯度的相对关系 ![](https://pic4.zhimg.com/80/v2-cb0ead3342c651c51940fca6afc8caf3_1440w.png) 引入了 BN 后，损失函数相对于激活函数值的梯度幅值更小，也即损失函数更加利普希兹 ![](https://pic2.zhimg.com/80/v2-2815f91154e4a065401883dff14d59fd_1440w.jpg) 引入了 BN 后，损失函数相对于激活函数值的二阶项幅值更小，也即损失函数更加贝塔平滑。 ![](https://pic2.zhimg.com/80/v2-6cd4330cca23338320448f5d152973e1_1440w.png) 同理，损失函数相对于权重的梯度幅值也更小 ![](https://pic3.zhimg.com/v2-54658df70b8072139c03c4b035eb81d6_r.jpg) 引入 BN 后，权重的最优解与初始解的距离也更小，也即神经网络更快就可以训练到最佳表现

## [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift ]