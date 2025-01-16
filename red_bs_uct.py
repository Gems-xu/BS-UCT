import math
import time

from random import choice
from class_chess import *

alpha = 0.01  # 信念更新参数
calculation_time = float(10)


def cal_belief(n):
    if n == 0:
        return 0.
    elif n > 0:
        return alpha * np.log(n + 1)
    else:
        return -alpha * np.log(-n + 1)


def run_simulation2(SL, plays, wins, max_actions=1000):  # 红方
    global k
    expand = True
    availables = SL  # 合法后继
    visited_states = set()  # 以访问节点，判断是否拓展
    for _ in range(1, max_actions + 1):
        if all(plays.get(move) for move in availables):  # 如果都访问过
            a = 0
            for move in availables:
                a += plays.get(move)  # 总访问次数
            move = availables[0]
            for moved in availables:
                if ((wins.get(move) / (plays.get(move) + 1e-99)) + 0.85 * (
                        2 * math.log(a) / (plays.get(move) + 1e-99)) ** 0.5 + 1.15 * getScore(move) / (
                            plays.get(move))) < ((wins.get(moved) / (plays.get(moved) + 1e-99)) + 0.85 * (
                        2 * math.log(a) / (plays.get(moved) + 1e-99)) ** 0.5 + 1.15 * getScore(moved) / plays.get(
                    moved)):
                    move = moved

        else:  # 随机选一个拓展
            peripherals = []
            for move in availables:
                if not plays.get(move):
                    peripherals.append(move)
            move = choice(peripherals)
        NL = []
        NL.append(move)
        availables = getTheNextStepStatus(NL)  # 更新合法后继
        if expand and move not in plays:
            expand = False
            plays[move] = 0
            wins[move] = 0
        visited_states.add(move)
        k = 0
        if move is not False:
            k = isEnd(move)
            if k:
                break
    for move in visited_states:
        if move in plays:
            plays[move] += 1
            # all visited moves
            if k == 1:
                wins[move] += 1
    return wins, plays, availables


def select_one_move(wins, availables):
    move = availables[0]
    for moved in availables:
        if wins.get(move, 0) <= wins.get(moved, 0):
            move = moved
    return move.pPawn, move.pMove


def redByUct(S, ans):
    # 制造红蓝棋谱
    datanew = []
    NPmap = np.array(S.map)
    reddata = np.where(NPmap < 7, NPmap, 0)
    bluedata = np.where(NPmap >= 7, NPmap, 0)
    datanew.append(reddata)
    datanew.append(bluedata)

    # calculation_time = float(120)
    bestp = 0
    bestm = ''
    move = ['right', 'down', 'rightdown']
    playsr = {}
    winsr = {}
    SL = []  # 当前局面下合法的后续走法
    for p in ans:
        for m in move:
            newStatus = tryMakeMove0(p, m, S)
            if newStatus is not False:
                SL.append(newStatus)
                if isEnd(newStatus):
                    return p, m
            del newStatus
    if len(SL) == 1:
        return SL[0].pPawn, SL[0].pMove  # 如只有一个合法后续，直接返回
    games = 0
    begin = time.time()
    while time.time() - begin < calculation_time:
        winsr, playsr, availables = run_simulation2(SL, playsr, winsr)
        games += 1
    bestp, bestm = select_one_move(winsr, SL)
    for move1 in SL:
        print(winsr.get(move1, 0) / playsr.get(move1, 1))
    print(games)
    print(bestp, bestm)
    return bestp, bestm
