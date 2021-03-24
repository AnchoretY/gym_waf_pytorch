import gym
import numpy as np

import torch                                    
import torch.nn as nn                           
import torch.nn.functional as F                 

import gym_waf.envs.wafEnv
from gym_waf.envs.wafEnv  import samples_test, samples_train
from gym_waf.envs.features import Features
from gym_waf.envs.waf import Waf_Check
from gym_waf.envs.xss_manipulator import Xss_Manipulator

from model.DQN import DQN


def train(dqn,episode,max_episode_steps,memory_capacity):
    """
    模型训练
        param dqn:要训练的DQN模型
        param episode: 训练样本抽样的进行强化学习的次数
        param max_episode_steps: 一个episode中最多能进行action次数
        param memory_capacity: 记忆库容量
    """
    
    for i in range(episode):                                                # 使用的训练样本数
        s = env.reset()                                                     # 重置环境
        episode_reward_sum = 0                                              # 初始化该循环对应的episode的总奖励
        step = 0

        done = False
        # 样本成功完成任务或到达可走最大step结束循环
        while not done or step<max_episode_steps:                                                
            step += 1
            a = dqn.choose_action(s)                                        # 输入该步对应的状态s，选择动作
            s_, r, done, info = env.step(a)                                 # 执行动作，获得反馈
            dqn.store_transition(s, a, r, s_)                               # 存储样本
            episode_reward_sum += r                                         # 逐步加上一个episode内每个step的reward

            s = s_                                                          # 更新状态

            if dqn.memory_counter > memory_capacity:                        
                dqn.learn()   
    return dqn

def test(dqn,test_data,max_episode_steps,action_lookup):
    """
    测试模型效果
    param dqn: 要进行测试的DQN模型
    param test_data: 测试数据集
    param max_episode_steps: 一个episode中最多能进行action的次数
    param action_lookup: 行为值到具体操作字符串的映射
        
    """

    features_extra = Features()     # 特征向量
    waf_checker = Waf_Check()       # waf检验免杀效果
    xss_manipulatorer = Xss_Manipulator()   # 根据动作修改当前样本，来达到免杀
    
    success = 0     # 免杀成功数
    sum = 0         # 总数目
    
    for sample in test_data:
        sum += 1

        for _ in range(max_episode_steps):
            if not waf_checker.check_xss(sample) :
                success += 1
                break
            f = features_extra.extract(sample)
            f = torch.FloatTensor(f)                                    # 将x转换成32-bit floating point形式，并在dim=0增加维数为1的维度
            actions_value = dqn.eval_net.forward(f)                            # 通过对评估网络输入状态x，前向传播获得动作值
            action = torch.max(actions_value, 1)[1].data.numpy()                # 输出每一行最大值的索引，并转化为numpy ndarray形式
            action = action[0]    
            sample = xss_manipulatorer.modify(sample,action_lookup[action])
    
    print("总数量：{} 成功：{}".format(sum,success))
    return sum,success




ENV_NAME = 'Waf-v0' # 之前注册的环境名，要按这样的形式命名，否则报错
env = gym.make(ENV_NAME)                        # 使用gym库中的环境：CartPole，且打开封装(若想了解该环境，请自行百度)



BATCH_SIZE = 32                                 # 样本数量
LR = 0.01                                       # 学习率
EPSILON = 0.9                                   # greedy policy
GAMMA = 0.9                                     # reward discount
TARGET_REPLACE_ITER = 100                       # 目标网络更新频率
MEMORY_CAPACITY = 2000                          # 记忆库容量
N_ACTIONS = env.action_space.n                  
STATES_DIM = env.observation_space.shape[1]     





if __name__ == '__main__':
    # 构造动作速查表
    ACTION_LOOKUP = {i:act for i,act in enumerate(Xss_Manipulator.ACTION_TABLE.keys())} # key为原动作字典的下标0123，value为原动作字典的key即免杀操作名


    max_episode_steps = 5

    epoch = 5
    episode = 10
    success_sum = 0


    for i in range(epoch):
        dqn = DQN(STATES_DIM,N_ACTIONS,MEMORY_CAPACITY,EPSILON,GAMMA,TARGET_REPLACE_ITER,BATCH_SIZE,LR)
        dqn = train(dqn,episode,max_episode_steps,MEMORY_CAPACITY)
        sum,success = test(dqn,samples_test,max_episode_steps,ACTION_LOOKUP)
        success_sum += success

    print("平均成功比例：{}/{}".format(success_sum/epoch,sum))

    





