## Gym_waf_pytorch
本系统是[gym_waf]()的pytorch版本，分别使用原始pytorch和tianshou框架两种方式对模型进行了训练和测试。


#### RL模型抽象  
- action: 混淆操作，能够对xss进行混淆的操作，这里包括了四种简单的混淆操作
- state: 这里使用xss攻击字符串中的各个字符对应的ascii个数来与xss攻击字符长度来进行刻画，257维向量  
- reward: 成功绕过检测模型则认为模型绕过成功了  
- 待绕过系统：简单的规则系统


​    

#### 分析：

- 这里的state表示其实存在比较大的问题，xss攻击字符串中的个字符对应的ascii与xss攻击字符串长度虽然能够是直接收到action影响，并且也是直接影响是否能够xss攻击成功，但是这种刻画方式一方面过于简单了，并不能对各个字符在攻击payload中的位置与是否组成了单词进行刻画，而且这种采用xss payload自身相关的payload，并没有考虑到攻击系统的环境问题，因此在不同的防御环境中并不会指定专业的策略来选择action
- 模型的绕过能力很大程度上受限于提供的action，即可供选择的混淆操作
- 模型的作用是能够再更少的尝试次数下生成对抗样本，主要作用是减少了尝试次数，并不是做了一些传统方法做不了的工作，只是减少尝试次数而已