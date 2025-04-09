import numpy as np
from class_chess import Status
# from read_data import get_belief

calculation_time = float(60)  # 计算时间
num_process = 8  # 多进程数目
alpha = 0.01  # 信念更新参数
# beliefs = get_belief('datas/Data_UCTwithDemo')  # 信念


def cal_belief(n):
    if n == 0:
        return 0.
    elif n > 0:
        return alpha * np.log(n + 1)
    else:
        return -alpha * np.log(-n + 1)


def get_belief(pre_path='datas/Data_UCTwithDemo'):  # static beliefs
    belief = {}
    paths = ['/', '_same/', '_same21HZ/']
    for path in paths:
        data_input = pre_path + path + 'input'
        data_record = pre_path + path + 'record'
        data_result = pre_path + path + 'result'
        for i in range(1, 47):
            # data_name = datas + str(i) + '.npy'
            inputs = np.load(data_input + str(i) + '.npy')
            records = np.load(data_record + str(i) + '.npy')
            results = np.load(data_result + str(i) + '.npy')
            # print(inputs.shape, records.shape, results.shape)
            # print(len(inputs), len(records), len(results))
            # print(records[-1])
            # print(results)
            for j in range(1, len(inputs)):  # 初始状态不记录
                state = inputs[j][0] + inputs[j][1]
                # 将棋盘与上一个玩家(红方0, 蓝方1)合并为一个元组
                s_tuple = tuple(state.flatten()) + (1 - j % 2,)
                res = 1 if results[records[j]] == 1 else -1
                belief[s_tuple] = belief.get(s_tuple, 0) + res
    return belief


def get_belief_state(S: Status) -> tuple:
    if S is None:
        return ()
    return tuple(np.array(S.map).flatten()) + (S.pPawn // 7,)
