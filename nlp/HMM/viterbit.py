# -*- coding: utf-8 -*-
# @Author: adewin
# @Date:   2017-07-26
###########################################################################################################
# Viterbi Algorithm for HMM
# dp, time complexity O(mn^2), m is the length of sequence of observation, n is the number of hidden states
#维特比算符，算出在已知观察序列以及隐马尔模型的情况下求解最可能的隐藏序列
##########################################################################################################


# five elements for HMM
#观测序列（observations）：实际观测到的现象序列
#隐含状态（states）：所有的可能的隐含状态
#初始概率（start_probability）：每个隐含状态的初始概率
#转移概率（transition_probability）：从一个隐含状态转移到另一个隐含状态的概率
#发射概率（emission_probability）：某种隐含状态产生某种观测现象的概率

states = ('Healthy', 'Fever')

observations = ('normal', 'cold', 'dizzy')

start_probability = {'Healthy': 0.6, 'Fever': 0.4}

transition_probability = {
    'Healthy': {'Healthy': 0.7, 'Fever': 0.3},
    'Fever': {'Healthy': 0.4, 'Fever': 0.6},
}

emission_probability = {
    'Healthy': {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
    'Fever': {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6},
}


def Viterbit(obs, states, s_pro, t_pro, e_pro):
    path = {s: [] for s in states}#path的初始值为{'Healthy': [], 'Fever': []}
    curr_pro = {}
    for s in states:
        curr_pro[s] = s_pro[s] * e_pro[s][obs[0]]#结果：{'Healthy': 0.3, 'Fever': 0.04000000000000001}
    for i in xrange(1, len(obs)):
        last_pro = curr_pro
        curr_pro = {}
        for curr_state in states:#遍历当前的每一个隐状态，计算每个隐状态的概率值
            max_pro, last_sta = max(
                ((last_pro[last_state] * t_pro[last_state][curr_state] * e_pro[curr_state][obs[i]], last_state)
                 for last_state in states))
            print last_pro
            curr_pro[curr_state] = max_pro
            path[curr_state].append(last_sta)#记录当前隐状态的上一个隐状态

    # find the final largest probability
    max_pro = -1
    max_path = None
    for s in states:
        path[s].append(s)#没有什么特别的作用，只是最后一个隐状态加到path当中，利于输出
        if curr_pro[s] > max_pro:
            max_path = path[s]
            max_pro = curr_pro[s]
            # print '%s: %s'%(curr_pro[s], path[s]) # different path and their probability
    return max_path


if __name__ == '__main__':
    obs = ['normal', 'cold', 'dizzy']
    print Viterbit(obs, states, start_probability, transition_probability, emission_probability)