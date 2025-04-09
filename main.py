import gc
import os
import time
import torch

from torch import multiprocessing as mp

from class_chess import *
from algorithms.red_neural_uct import run_simulation_network_t
from algorithms.red_bs_mcts import *

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
torch.backends.cudnn.benchmark = True

COUNT = 0  # 记录回合数
myTime = 0  # 己方所用时间
yourTime = 0  # 对方所用时间
RESULT = [0, 0]  # 记录比赛结果
pathNet = 'models/0830_1111_Alpha1.pt'
pathValueNet = 'models/0830_1111_Alpha1_value_little.pt'

gpus = [0, 1]
Dice = []  # 每回合色子
Games = 0  # 总模拟次数
Records = None  # 记录棋局
A = 0  # 信念控制系数
current_beliefs = {}  # 当前动态信念
global_beliefs = get_belief('datas/Data_UCTwithDemo')  # 全局静态信念


def init():
    global S
    S = Status()


def resetInfo():  # 重置比赛信息
    S.map = getcommap_rand(COUNT)
    S.pawn = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # 棋子初始化
    S.pro = [1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 /
             6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6]


def makeMove(p, PawnMoveTo):  # 移动棋子，更新地图信息，和棋子存活情况
    back_S = copy.deepcopy(S)
    row, col = getLocation(p, S.map)
    x = y = 0
    if PawnMoveTo == LEFT:
        y = -1
    elif PawnMoveTo == RIGHT:
        y = +1
    elif PawnMoveTo == UP:
        x = -1
    elif PawnMoveTo == DOWN:
        x = +1
    elif PawnMoveTo == LEFTUP:
        x = -1
        y = -1
    elif PawnMoveTo == RIGHTDOWN:
        x = +1
        y = +1
    else:
        return False
    # 移动无效
    if notInMap(row + x, col + y):
        return False
    S.map[row][col] = 0
    row = row + x
    col = col + y
    if S.map[row][col] != 0:
        i = S.pawn.index(S.map[row][col])
        S.pawn[i] = 0
    S.map[row][col] = p
    value = getLocValue(S)  # 获取所有棋子的位置价值
    S.pro = getPawnPro(S)  # 获取所有棋子被摇到的概率
    S.value = getPawnValue(value, S.pro)
    S.parent = back_S
    if back_S.parent is not None:
        S.parent_before = back_S.parent.map
    else:
        S.parent_before = None
    S.parent_3 = back_S.parent_before
    S.parent_4 = back_S.parent_3
    return True


def tryMakeMove(p, PawnMoveTo, S):  # 尝试移动，并且返回移动后的棋局地图与棋子存活情况
    newS = copy.deepcopy(S)
    row, col = getLocation(p, newS.map)
    x = y = 0
    if PawnMoveTo == LEFT:
        y = -1
    elif PawnMoveTo == RIGHT:
        y = +1
    elif PawnMoveTo == UP:
        x = -1
    elif PawnMoveTo == DOWN:
        x = +1
    elif PawnMoveTo == LEFTUP:
        x = -1
        y = -1
    elif PawnMoveTo == RIGHTDOWN:
        x = +1
        y = +1
    # 移动无效
    if notInMap(row + x, col + y):
        return False
    newS.map[row][col] = 0
    row = row + x
    col = col + y
    if newS.map[row][col] != 0:  # 检查移动的目标格子位是否有棋子，是的话被吃掉，赋值为0
        i = newS.pawn.index(newS.map[row][col])
        newS.pawn[i] = 0
    newS.map[row][col] = p
    value = getLocValue(newS)  # 获取所有棋子的位置价值
    newS.pro = getPawnPro(newS)  # 获取所有棋子被摇到的概率
    newS.value = getPawnValue(value, newS.pro)
    newS.parent = S
    if S.parent is not None:
        newS.parent_before = S.parent.map
    else:
        newS.parent_before = None
    newS.parent_3 = S.parent_before
    newS.parent_4 = S.parent_3
    newS.pPawn = p
    newS.pMove = PawnMoveTo
    if p < 7:
        newS.cPawn = [INFTY, INFTY, INFTY, INFTY, INFTY, INFTY]
        newS.cPawnSecond = [INFTY, INFTY, INFTY, INFTY, INFTY, INFTY]
    else:
        newS.cPawn = [-INFTY, -INFTY, -INFTY, -INFTY, -INFTY, -INFTY]
        newS.cPawnSecond = [-INFTY, -INFTY, -INFTY, -INFTY, -INFTY, -INFTY]
    return newS


def blue_rand(ans):  # 蓝方随机移动
    moveTo = ['leftup', 'up', 'left']
    SL = []  # 当前局面下合法的后续走法
    for p in ans:
        for m in moveTo:
            newStatus = tryMakeMove(p, m, S)
            if newStatus is not False:
                SL.append(newStatus)
                if isEnd(newStatus):
                    break
                del newStatus
    rand_i = np.random.randint(0, len(SL))
    bestp, bestm = SL[rand_i].pPawn, SL[rand_i].pMove
    print('移动棋子：', bestp, ' 移动方向：', bestm)
    return bestp, bestm


def blueByBraveOfMan(ans):
    moveTo = ['leftup', 'up', 'left']
    bestp = 0
    bestm = ''
    for p in ans:
        for m in moveTo:
            newS = tryMakeMove(p, m, S)
            if newS is not False:
                bestp = p
                bestm = m
                del newS
                break
    print('移动棋子：', bestp, ' 移动方向：', bestm)
    return bestp, bestm


def redByNeuralUCT(ans):
    global playsr
    global winsr
    global current_beliefs, Games
    # global Vsr
    # global recorder
    # global Vsr_all
    # print("red can move ", ans)
    # calculation_time = float(5)
    move = ['right', 'down', 'rightdown']
    Vsr = {}
    Vsr_all = {}
    recorder = {}
    Tempt = {}
    SL = []  # 当前局面下合法的后续走法

    datanew = get_alldata_5(S)

    for p in ans:
        for m in move:
            newStatus = tryMakeMove(p, m, S)
            if newStatus is not False:
                Vsr_all[newStatus] = 0.  # 改为 float
                SL.append(newStatus)
                if isEnd(newStatus):
                    return p, m
                del newStatus
    if len(SL) == 1:
        return SL[0].pPawn, SL[0].pMove  # 如只有一个合法后续，直接返回
    games = 0
    # playsr[S] = 0
    # num_p = min(num_process + COUNT // 2, 16)
    num_p = num_process  # 2024-04-27
    # print("进程数为：", num_p)
    cal_time = min(calculation_time + COUNT * 5, 160)
    # cal_time = calculation_time  # 2024-04-27
    # print("计算时间：", cal_time)
    pool = mp.Pool(processes=num_p)  # 进程池

    begin = time.time()
    # print("t=", end='')
    while time.time() - begin < cal_time:
        # start = time.time()
        # inputs_run_sim_net = []  # 多进程输入
        # for _ in range(num_p):
        #     inputs_run_sim_net.append([playsr, Vsr, SL, datanew, 30])
        inputs_run_sim_net = [[playsr, Vsr, SL, datanew, 30] for _ in range(num_p)]
        # print("Run time", time.time() - start)
        outputs_run_sim_net = pool.map(run_simulation_network_t, inputs_run_sim_net)  # 多进程输出
        # print("Run time", time.time() - start)
        # playsr[S] += num_p
        # games += num_p  # 2024-06-03
        # start = time.time()
        for output in outputs_run_sim_net:
            Qsr, The_total_choose, k, visited_states, visited_red, t = output
            # print(t, end=',')
            games += t  # 2024-06-03
            # if playsr.get(The_total_choose, -1) == -1:
            #     playsr[The_total_choose] = 0
            #     winsr[The_total_choose] = 0

            for move in visited_states:
                # Update beliefs
                belief_s = get_belief_state(move)
                if current_beliefs.get(belief_s, -1) == -1:
                    current_beliefs[belief_s] = 0
                # if move == The_total_choose:
                #     playsr[The_total_choose] += 1
                # else:
                if playsr.get(move, -1) == -1:  # 激活已选节点在playsr中，winsr中
                    playsr[move] = 0
                    winsr[move] = 0
                playsr[move] += 1
                if k == 1:
                    # if move == The_total_choose:
                    #     winsr[The_total_choose] += 1
                    # else:
                    winsr[move] += 1
                    current_beliefs[belief_s] += 1
                elif k == 2:
                    current_beliefs[belief_s] -= 1

                if move in visited_red:
                    if recorder.get(The_total_choose, 0) == 0:
                        recorder[The_total_choose] = 0
                    recorder[The_total_choose] += 1
                    if k == 1:
                        Qsr[move] = (Qsr.get(move, 0) + 1) / 2
                    elif k == 2:
                        Qsr[move] = (Qsr.get(move, 0) - 1) / 2
                    Vsr_all[The_total_choose] += Qsr.get(move, 0)
                Vsr[The_total_choose] = Vsr_all.get(The_total_choose, 0) / recorder.get(The_total_choose, 1)
        # print("Calculation time", time.time() - start)
        # if games % 40 == 0:
        #     for move in SL:
        #         Tempt[move] = winsr.get(move, 0) / playsr.get(move, 1)
        # file.write(str(Tempt[move]) + ' ')
        # print(Tempt)
        # file.write('\n')
    # print()
    print('We have searched for ', games)
    # Games += games
    # playsr[S] = 0
    pool.close()
    pool.join()
    # file.close()

    for move in SL:
        # 在此处添加 Belief 信息
        b_info = A * cal_belief(current_beliefs.get(get_belief_state(move), 0))
        Tempt[move] = winsr.get(move, 0) / playsr.get(move, 1) + b_info
        print(move.get_move(),
              "b=", round(b_info, 2),
              "v=", round(Tempt[move], 2))

    move_choose = max(Tempt, key=lambda x: Tempt[x])

    for move in SL:
        if move != move_choose and Tempt[move] == Tempt[move_choose] and move.pMove == 'rightdown':
            move_choose = move
    bestp = move_choose.pPawn
    bestm = move_choose.pMove

    # for move in SL:
    #     print(playsr.get(move, 0))
    # for move1 in SL:
    #     print(Tempt.get(move1, 0))
    # for move in SL:
    #     print("the valuenet total is", Vsr.get(move,0))

    # print('we have searched for ', games)
    # print('bestp=', bestp, ' bestm=', bestm)

    del Vsr
    del Vsr_all
    del recorder
    del SL
    gc.collect()
    return bestp, bestm


def playGame(Red, Blue, detail=False, now=0):  # 选择策略
    global COUNT, Records
    global recordplay
    global recordtime
    global tmpTime, myTime, yourTime
    global current_beliefs, Games
    movered_moveblue = {'right': 'left', 'down': 'up', 'rightdown': 'leftup'}
    coldict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
    S.indx = 0

    # COUNT = 0  # 回合数
    myTime = yourTime = 0  # 时间重置
    current_beliefs = global_beliefs.copy()  # 信念重置
    while True:
        moveTo = None  # 棋子移动方向
        # s = '己方'
        # if cnt % 2 == 0 and playhand == 'first' or cnt % 2 == 1 and playhand == 'second':
        #     s = '对方'
        # cnt += 1
        # number = input('请输入' + s + '的色子数: ')
        # number = int(number)
        number = roll_dice()  # 2023-11-14
        Dice.append(number)   # 2024-04-28
        # print('色子数为：', number)

        # n, ans = selectPawn(S)
        # ans: Represents a list of movable chess pieces
        n, ans = selectPawnnewone(S, COUNT, number)  # 2023-11-23
        # n, ans, COUNT = select_pawn_new_one_goahead(COUNT, S, n=number)  # 2023-11-23

        # mymap = np.array(S.map)
        # if playhand == 'second':
        #     mymap = rotate(mymap)
        # print(COUNT % 2, '行棋前，棋盘为：\n', mymap)
        p = 0  # 将要移动的棋子
        tmpTime = time.time()  # 计时开始
        if COUNT % 2 == 0:
            if Red == 'Nerual_Uct':
                p, moveTo = redByNeuralUCT(ans)
                # redByMinimax(ans)
                myTime += time.time() - tmpTime
                if playhand == 'first':
                    print('告诉对方：', p, moveTo)
                if playhand == 'second':
                    print('告诉对方：', p, movered_moveblue[moveTo])
        elif COUNT % 2 == 1:
            if Blue == 'GoAhead':
                p, moveTo = blueByBraveOfMan(ans)
            if Blue == 'Rand':
                p, moveTo = blue_rand(ans)
            # 计算对方用时
            yourTime += time.time() - tmpTime

        # 输出时间
        print('红方用时：', round(myTime, 2), 's')
        print('蓝方用时：', round(yourTime, 2), 's')

        if moveTo is not None:
            makeMove(p, moveTo)
            # BEGIN: Added on 2023-8-4
            recordtime = recordtime + 1
            if playhand == 'first':
                pass
            elif playhand == 'second':
                if COUNT % 2 == 1:
                    secondmap = reverse(np.array(S.map))
                    secondmap = secondmap.tolist()
                    row, col = getLocation(p, secondmap)
                    myrow = 5 - row
                    mycol = coldict[col]
                    recordplay.append(
                        str(recordtime) + ':' + str(number) + ';(' + 'R' + str(p - 6) + ',' + mycol + str(myrow) + ')')
                if COUNT % 2 == 0:
                    secondmap = reverse(np.array(S.map))
                    secondmap = secondmap.tolist()
                    row, col = getLocation(p, secondmap)
                    myrow = 5 - row
                    mycol = coldict[col]
                    recordplay.append(
                        str(recordtime) + ':' + str(number) + ';(' + 'B' + str(p) + ',' + mycol + str(myrow) + ')')
            # END: Added on 2023-8-4
            S.indx += 1
            # lastInfo = [n, p, moveTo]
            # if moved:
            #     mapNeedRedraw = True

        result = isEnd(S)  # 检查游戏是否结束，返回游戏结果
        cur_map = np.array(S.map)  # 当前棋局
        # 将一个(5,5)矩阵 b 追加到(n,5,5)矩阵 a 后面，得到一个(n+1,5,5)矩阵
        Records = np.concatenate([Records, cur_map[np.newaxis, :, :]],
                                 axis=0)
        print(player(COUNT % 2) + '行棋后，棋盘为：\n', cur_map)
        COUNT += 1
        if result:
            return result


def startGame(Red, Blue, n, detail):
    global COUNT, Records, Dice
    global S
    global playsr, playsr2
    global winsr, winsr2
    global playhand
    global remymap
    global recordplay
    global recordtime

    playhand = 'None'
    remymap = []
    # name_they = input('请输入对手的名字：')
    init()
    RESULT[0] = 0
    RESULT[1] = 0
    rateline = []

    cnt = n  # 比赛局数减少
    game_round = 1  # 比赛局数增加

    while cnt:
        COUNT = 0  # 比赛回合数
        recordplay = []
        recordtime = 0
        # c = input('请输入先行棋的一方，红1蓝2: ')
        # c = int(c)
        c = 1
        # COUNT = c
        if c == 1:
            playhand = 'first'
        else:
            playhand = 'second'

        global S
        S = Status()
        resetInfo()
        playsr, playsr2 = {}, {}
        winsr, winsr2 = {}, {}
        Records = np.array(S.map).reshape(1, 5, 5)
        print('------第', game_round, '局------')
        print('初始棋盘为：\n', np.array(S.map))
        Dice = []
        result = playGame(Red, Blue, detail)  # 游戏开始，返回比赛结果
        game_round += 1
        gc.collect()

        print("The result is", result - 1)
        del S
        del winsr, winsr2
        del playsr, playsr2
        # if detail:
        #     pass
        RESULT[result - 1] += 1  # 更新比分
        cnt -= 1
        rateline.append(float(RESULT[0]) / sum(RESULT))
        if game_round >= n:  # 显示一次胜负情况
            print(sum(RESULT), '\t', round(100 * RESULT[0] / sum(RESULT), 4))
    return round(100 * RESULT[0] / sum(RESULT), 4)


if __name__ == '__main__':
    '''
    可选测试对象
    Red  ：Nerual_Uct
    Blue ：GoAhead | Rand
    Human表示棋手为人类.
    '''
    mp.set_start_method('spawn')  # 多进程启动方法

    Red = 'Nerual_Uct'
    Blue = 'Rand'

    for i in range(1):
        A = 1 # 调节信念更新参数
        cnt = 1  # 比赛局数
        allcnt = cnt
        result = startGame(Red, Blue, cnt, False)
        print(result)

# REDWIN = 1  # 代表RED方赢
# BLUEWIN = 2  # 代表玩家BLUE方赢