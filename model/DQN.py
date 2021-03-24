import torch                                    
import torch.nn as nn                           
import torch.nn.functional as F                 
import numpy as np                           

# 定义Net类 (定义网络)
class Net(nn.Module):
    def __init__(self,states_dim,n_actions):                                                         # 定义Net的一系列属性
        super(Net, self).__init__()                                             

        self.fc1 = nn.Linear(states_dim, 50)                                    # 权重初始化 (均值为0，方差为0.1的正态分布)
        self.fc1.weight.data.normal_(0, 0.1)                                    
        self.out = nn.Linear(50, n_actions)                                     
        self.out.weight.data.normal_(0, 0.1)                                    # 权重初始化 (均值为0，方差为0.1的正态分布)

    def forward(self, x):                                                       
        x = F.relu(self.fc1(x))                                                
        actions_value = self.out(x)                                             
        return actions_value                                                  


# 定义DQN类 (定义两个网络)
class DQN(object):
    def __init__(self,state_dim,n_actions,memory_size,epsilon,gamma,target_replace_iter,batch_size,lr):
        """
            state_dim: 状态的向量的维度
            n_actions: 可选行为的个数
            batch_size:每次从记忆库中要抽取的进行q现实与q估计的差值，并进行eval_net更新的批处理大小
            gamma: 未来收益折线比率
            epsilon: 贪心算法系数，有epsilon的概率选择最优的action，有1-epsilon的概率随机进行选择
            memory_size: 记忆库容量的大小，每次目标网路的学习是在记忆库中随机进行抽取样本进行学习的
            target_replace_iter: 更新频率，经过多少个step后，使用eval_net的参数替换target网络的参数
            lr: eval_net的学习速率
        """
        self.state_dim = state_dim 
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.gamma = gamma
        self.memory_size = memory_size
        self.target_replace_iter = target_replace_iter
        self.batch_size = batch_size
        
        
        self.eval_net, self.target_net = Net(state_dim,n_actions), Net(state_dim,n_actions) # 利用Net创建两个神经网络: 评估网络和目标网络
        self.learn_step_counter = 0                                              # 记录当前step数，以此来判断什么时候更新目标网络
        self.memory_counter = 0                                                  # 存储的记忆数量，以此来判断什么时候进行记忆替换
        self.memory = np.zeros((memory_size, state_dim * 2 + 2))                 # 初始化记忆库，一行代表一个transition
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)    
        self.loss_func = nn.MSELoss()                                           

    def choose_action(self, obs):
        """
            行为选择函数，根据当前obsversation进行action选择
        """
        obs = torch.FloatTensor(obs)                                    
        if np.random.uniform() < self.epsilon:                                  # 生成一个在[0, 1)内的随机数，如果小于EPSILON，选择最优动作，否则随机选择
            actions_value = self.eval_net.forward(obs)                            
            action = torch.max(actions_value, 1)[1].data.numpy()                # 找出行为q值最大的索引，并转化为numpy数组
            action = action[0]                                                  
        else:                                                                   
            action = np.random.randint(0, self.n_actions)                           
        return action                                                           

    def store_transition(self, s, a, r, s_):                                  
        """
            记忆存储函数
        """
        s,s_ = np.squeeze(s),np.squeeze(s_)
        # 在水平方向上拼接数组
        transition = np.hstack((s, [a, r], s_))                                
        # 如果记忆库满了，便覆盖旧的数据
        index = self.memory_counter % self.memory_size                         
        self.memory[index, :] = transition                                     
        self.memory_counter += 1                                                

    def learn(self): 
        """
            学习函数，记忆库存储满后开始进行学习
        """
        # 每隔target_replace_iter步后，target网络参数更新
        if self.learn_step_counter % self.target_replace_iter == 0:            
            self.target_net.load_state_dict(self.eval_net.state_dict())         
        self.learn_step_counter += 1                                            

        # 抽取记忆库中的batch_size个记忆数据
        sample_index = np.random.choice(self.memory_size, self.batch_size)         
        b_memory = self.memory[sample_index, :]                                 
        
        # 记忆数据中的state、action、reward、state_分开
        b_s = torch.FloatTensor(b_memory[:, :self.state_dim])
        b_a = torch.LongTensor(b_memory[:, self.state_dim:self.state_dim+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.state_dim+1:self.state_dim+2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.state_dim:])

        # 获取32个transition的评估值和目标值，并利用损失函数和优化器进行评估网络参数更新
        q_eval = self.eval_net(b_s).gather(1, b_a)                        # eval_net(b_s)通过评估网络输出32行每个b_s对应的一系列动作值，然后.gather(1, b_a)代表对每行对应索引b_a的Q值提取进行聚合
        q_next = self.target_net(b_s_).detach()                           # Note: q_next不进行反向传递误差，所以detach
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)            # q_next.max(1)[0]表示只返回每一行的最大值，不返回索引(长度为32的一维张量)；.view()表示把前面所得到的一维张量变成(BATCH_SIZE, 1)的形状；最终通过公式得到目标值
        loss = self.loss_func(q_eval, q_target)
        
        # 输入32个评估值和32个目标值，使用均方损失函数
        self.optimizer.zero_grad()                                      # 清空上一步的残余更新参数值
        loss.backward()                                                 # 误差反向传播, 计算参数更新值
        self.optimizer.step()                                           # 更新评估网络的所有参数


