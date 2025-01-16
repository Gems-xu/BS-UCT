import copy
import random

import numpy as np
import pygame

# COUNT = 0
INFTY = 10000
LEFT = 'left'
RIGHT = 'right'
UP = 'up'
DOWN = 'down'
LEFTUP = 'leftup'
RIGHTDOWN = 'rightdown'
REDWIN = 1  # 代表RED方赢
BLUEWIN = 2  # 代表玩家BLUE方赢


class Status(object):
    def __init__(self):
        self.map = None  # 矩阵棋盘
        self.value = None  # 所有棋子的价值
        self.pawn = None  # 棋子列表，没有的标记为0
        self.pro = None  # 所有棋子被摇到的概率
        self.parent_before = None
        self.parent = None  # 此局面上一轮时状态
        self.parent_3 = None
        self.parent_4 = None

        self.pPawn = None  # 上一轮选择的棋子
        self.pMove = None  # 上一轮选择的移动方向
        self.pDice = None  # 父节点的骰子数
        self.cPawn = None
        self.cPawnSecond = None
        self.indx = None  # 记录局面所处的步数
        self.cPM = [[], [], [], [], [], []]
        self.cPMSecond = [[], [], [], [], [], []]
        self.children = []

        # self.player = 1  # 1: red, 0: blue

    def __eq__(self, other):
        return (self.map == other.map) and (self.pPawn == other.pPawn) and (self.pMove == other.pMove)

    def __hash__(self):
        return hash((str(self.map), self.pPawn, self.pMove))

    def __str__(self):
        # print(Status)
        s = [item for row in self.map for item in row]
        return '(%s, %s, %s)' % (self.pPawn, self.pMove, str(s))

    def print(self):
        print(self.cPM)

    def get_map(self):
        return self.map

    def get_move(self):
        return '(%s, %s)' % (self.pPawn, self.pMove)


def init0():
    global IMAGE, tip, screen, font, maplib, Lyr, Lyb, Lx, S, matchPro
    pygame.init()
    S = Status()

    # 布局库
    maplib = [[6, 2, 4, 1, 5, 3],
              [6, 5, 2, 1, 4, 3],
              [1, 5, 4, 6, 2, 3],
              [1, 6, 3, 5, 2, 4],
              [1, 6, 4, 3, 2, 5],
              [6, 1, 2, 5, 4, 3],
              [6, 1, 3, 5, 4, 2],
              [1, 6, 4, 2, 3, 5],
              [1, 5, 2, 6, 3, 4],
              [1, 6, 5, 2, 3, 4],
              [1, 2, 5, 6, 3, 4],
              [6, 2, 5, 1, 4, 3],
              [1, 6, 3, 2, 4, 5],
              [6, 2, 3, 1, 5, 4],
              [1, 6, 3, 4, 2, 5],
              [1, 5, 4, 6, 3, 2]]
    # resetInfo()  # 重置比赛信息  # Annotated in 2023-8-4
    Lyr = []
    Lyb = []
    Lx = []
    matchPro = 0.85


def cal_first_choice(probli, Vsr, nowS, c=9, value_balance=0.1, _playsr=None):
    if _playsr is None:
        _playsr = {}
    U = c * probli / (_playsr.get(nowS, 0) + 1)
    V = Vsr.get(nowS, 0) * value_balance
    C = U + V
    return C


def findNearby(n, now_pawn):  # 寻找可以移动的棋子，n是当前骰子数，返回所有符合条件棋子
    ans = []
    if now_pawn[n - 1] != 0:
        ans.append(n)
    elif n > 6:
        for i in range(n - 1, 6, -1):
            if i in now_pawn:
                ans.append(i)
                break
        for i in range(n + 1, 13):
            if i in now_pawn:
                ans.append(i)
                break
    elif n <= 6:
        for i in range(n - 1, 0, -1):
            if i in now_pawn:
                ans.append(i)
                break
        for i in range(n + 1, 7):
            if i in now_pawn:
                ans.append(i)
                break
    return ans


def get_alldata_5(myS: Status):
    datanew = []
    NPmap = np.array(myS.map)
    reddata = np.where(NPmap < 7, NPmap, 0)
    bluedata = np.where(NPmap >= 7, NPmap, 0)
    datanew.append(reddata)
    datanew.append(bluedata)
    if myS.parent is not None:
        NPmap = np.array(myS.parent.map)
        reddata = np.where(NPmap < 7, NPmap, 0)
        bluedata = np.where(NPmap >= 7, NPmap, 0)
        datanew.append(reddata)
        datanew.append(bluedata)
    else:
        reddata = np.zeros([5, 5], int)
        bluedata = np.zeros([5, 5], int)
        datanew.append(reddata)
        datanew.append(bluedata)

    if myS.parent_before is not None:
        NPmap = np.array(myS.parent_before)
        reddata = np.where(NPmap < 7, NPmap, 0)
        bluedata = np.where(NPmap >= 7, NPmap, 0)
        datanew.append(reddata)
        datanew.append(bluedata)
    else:
        reddata = np.zeros([5, 5], int)
        bluedata = np.zeros([5, 5], int)
        datanew.append(reddata)
        datanew.append(bluedata)

    if myS.parent_3 is not None:
        NPmap = np.array(myS.parent_3)
        reddata = np.where(NPmap < 7, NPmap, 0)
        bluedata = np.where(NPmap >= 7, NPmap, 0)
        datanew.append(reddata)
        datanew.append(bluedata)
    else:
        reddata = np.zeros([5, 5], int)
        bluedata = np.zeros([5, 5], int)
        datanew.append(reddata)
        datanew.append(bluedata)

    if myS.parent_4 is not None:
        NPmap = np.array(myS.parent_4)
        reddata = np.where(NPmap < 7, NPmap, 0)
        bluedata = np.where(NPmap >= 7, NPmap, 0)
        datanew.append(reddata)
        datanew.append(bluedata)
    else:
        reddata = np.zeros([5, 5], int)
        bluedata = np.zeros([5, 5], int)
        datanew.append(reddata)
        datanew.append(bluedata)
    return datanew


def getcommap0(COUNT):
    global remymap
    if COUNT % 2 == 1:
        a1 = input('输入对手棋1: ')
        a2 = input('输入对手棋2: ')
        a3 = input('输入对手棋3: ')
        a4 = input('输入对手棋4: ')
        a5 = input('输入对手棋5: ')
        a6 = input('输入对手棋6: ')

        a1 = int(a1)
        a2 = int(a2)
        a3 = int(a3)
        a4 = int(a4)
        a5 = int(a5)
        a6 = int(a6)
        newMap = [
            [6, 2, 4, 0, 0],
            [1, 5, 0, 0, 0],
            [3, 0, 0, 0, a1 + 6],
            [0, 0, 0, a2 + 6, a3 + 6],
            [0, 0, a4 + 6, a5 + 6, a6 + 6]
        ]
    else:
        a1 = input('输入对手棋1: ')
        a2 = input('输入对手棋2: ')
        a3 = input('输入对手棋3: ')
        a4 = input('输入对手棋4: ')
        a5 = input('输入对手棋5: ')
        a6 = input('输入对手棋6: ')

        a1 = int(a1)
        a2 = int(a2)
        a3 = int(a3)
        a4 = int(a4)
        a5 = int(a5)
        a6 = int(a6)

        newMap = [
            [6, 2, 4, 0, 0],
            [1, 5, 0, 0, 0],
            [3, 0, 0, 0, a6 + 6],
            [0, 0, 0, a5 + 6, a4 + 6],
            [0, 0, a3 + 6, a2 + 6, a1 + 6]
        ]
    remymap = [a1, a2, a3, a4, a5, a6]
    return newMap


def getcommap(count):  # 棋盘固定开局用于测试
    # global remymap
    if count % 2 == 1:
        a1, a2, a3, a4, a5, a6 = 3, 5, 1, 4, 2, 6
        newMap = [
            [6, 2, 4, 0, 0],
            [1, 5, 0, 0, 0],
            [3, 0, 0, 0, a1 + 6],
            [0, 0, 0, a2 + 6, a3 + 6],
            [0, 0, a4 + 6, a5 + 6, a6 + 6]
        ]
    else:
        a1, a2, a3, a4, a5, a6 = 6, 2, 4, 1, 5, 3
        newMap = [
            [6, 2, 4, 0, 0],
            [1, 5, 0, 0, 0],
            [3, 0, 0, 0, a6 + 6],
            [0, 0, 0, a5 + 6, a4 + 6],
            [0, 0, a3 + 6, a2 + 6, a1 + 6]
        ]
    # remymap = [a1, a2, a3, a4, a5, a6]
    return newMap


def getcommap_rand(count):  # 对方棋子随机开局
    # global remymap
    a = [1, 2, 3, 4, 5, 6]
    random.shuffle(a)
    a1, a2, a3, a4, a5, a6 = a
    if count % 2 == 0:
        newMap = [
            [6, 2, 4, 0, 0],
            [1, 5, 0, 0, 0],
            [3, 0, 0, 0, a1 + 6],
            [0, 0, 0, a2 + 6, a3 + 6],
            [0, 0, a4 + 6, a5 + 6, a6 + 6]
        ]
    else:
        # a1, a2, a3, a4, a5, a6 = 6, 2, 4, 1, 5, 3
        newMap = [
            [a1, a2, a3, 0, 0],
            [a4, a5, 0, 0, 0],
            [a6, 0, 0, 0, 9],
            [0, 0, 0, 11, 7],
            [0, 0, 10, 8, 12]
        ]
    # remymap = [a1, a2, a3, a4, a5, a6]
    return newMap


def getDemoValue(S, x=0, k=2.2, lam=5):  # 此时蓝方的局面估值
    redToBlueOfThread, blueToRedOfThread = getThread(S)
    expRed = expBlue = 0
    for i in range(0, 12):
        if i < 6:
            expRed += S.value[i]
        else:
            expBlue += S.value[i]
    theValue = lam * (k * expBlue - expRed) + blueToRedOfThread - redToBlueOfThread
    # if x == 1:
    #     print(expBlue, expRed, blueToRedOfThread, redToBlueOfThread)
    return theValue


def getDemoValueblue(S, k=2.2, lam=5):
    move = ['right', 'down', 'rightdown']
    exp = 0
    for p in range(1, 7):
        if p in S.pawn:
            # theValue = maxValue = -INFTY
            maxValue = -INFTY  # 2023-11-24
            for m in move:
                newStatus = tryMakeMove0(p, m, S)
                if newStatus is not False:
                    theValue = getScorered(newStatus)
                    if theValue > maxValue:
                        maxValue = theValue
            exp += S.pro[p - 1] * maxValue
    return exp


def getLocation(p, Map):  # 返回传入地图下，棋子p的坐标
    for i in range(5):
        for j in range(5):
            if Map[i][j] == p:
                return i, j


def getLocValue(S):  # 棋子所在位置的价值
    blueValue = [[99, 10, 6, 3, 1],
                 [10, 8, 4, 2, 1],
                 [6, 4, 4, 2, 1],
                 [3, 2, 2, 2, 1],
                 [1, 1, 1, 1, 1]]
    redValue = [[1, 1, 1, 1, 1],
                [1, 2, 2, 2, 3],
                [1, 2, 4, 4, 6],
                [1, 2, 4, 8, 10],
                [1, 3, 6, 10, 99]]
    V = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for p in range(1, 13):
        if S.pawn[p - 1] != 0:
            if getLocation(p, S.map) is None:
                print('ERROR! p=', p)
            row, col = getLocation(p, S.map)
            if p <= 6:
                V[p - 1] = redValue[row][col]
            else:
                V[p - 1] = blueValue[row][col]
    return V


def getPawnPro(S):  # 返回棋子被摇到的概率
    value = getLocValue(S)
    pro = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for p in range(1, 13):
        pro[p - 1] = 1.0 / 6
    for p in range(1, 13):
        if S.pawn[p - 1] == 0:
            ans = findNearby(p, S.pawn)
            if len(ans) > 1:
                pr = ans[0] - 1
                pl = ans[1] - 1
                if value[pr] > value[pl]:
                    pro[pr] += pro[p - 1]
                elif value[pr] == value[pl]:
                    pro[pr] += pro[p - 1] / 2
                    pro[pl] += pro[p - 1] / 2
                else:
                    pro[pl] += pro[p - 1]
            elif len(ans) == 1:
                pro[ans[0] - 1] += pro[p - 1]
            elif len(ans) == 0:
                pass
            pro[p - 1] = 0
    return pro


def getPawnValue(value, pro):  # 棋子价值
    V = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(0, 12):
        V[i] = pro[i] * value[i]
    return V


def getScorered(S, k=2.2, lam=5):  # 计算此时红方的局面估值
    redToBlueOfThread, blueToRedOfThread = getThread(S)
    expRed = expBlue = 0
    for i in range(0, 12):
        if i < 6:
            expRed += S.value[i]
        else:
            expBlue += S.value[i]

    theValue = lam * (k * expBlue - expRed) - redToBlueOfThread + blueToRedOfThread
    return theValue


def getSum(L):
    value = 0
    for i in L:
        if i != INFTY and i != -INFTY:
            value += i
    return (1 / 6) * value


def getTheNextStepStatus(SL):  # 根据现局面，获得所有合法的后续局面
    NL = []
    if SL[0].pPawn > 6:
        move = ['right', 'down', 'rightdown']
        o = 0
    else:
        move = ['left', 'up', 'leftup']
        o = 6
    for s in SL:
        for i in range(1, 7):
            n, ans = selectPawn(s, i + o)
            for p in ans:
                for m in move:
                    newStatus = tryMakeMove0(p, m, s)
                    if newStatus is not False:
                        newStatus.pDice = i
                        NL.append(newStatus)
                    del newStatus
    return NL


def getThread(S):  # 获得红方对蓝方的威胁值，蓝方对红方的威胁值
    redToBlueOfThread = 0
    blueToRedOfThread = 0
    for p in range(1, 13):
        if S.pawn[p - 1] != 0:
            if p <= 6:
                nearbyBlueMaxValue = searchNearbyBlueMaxValue(p, S)
                redToBlueOfThread += S.pro[p - 1] * nearbyBlueMaxValue
            else:
                nearbyRedMaxValue = searchNearbyRedMaxValue(p, S)
                blueToRedOfThread += S.pro[p - 1] * nearbyRedMaxValue
    return redToBlueOfThread, blueToRedOfThread


def isEnd(S):  # 检测比赛是否结束 1:红方赢 2:蓝方赢 False:未结束
    if S.map[0][0] > 6:
        return BLUEWIN
    elif 0 < S.map[4][4] <= 6:
        return REDWIN
    cnt = 0
    for i in range(0, 6):
        if S.pawn[i] == 0:
            cnt += 1
    if cnt == 6:
        return BLUEWIN
    cnt = 0
    for i in range(6, 12):
        if S.pawn[i] == 0:
            cnt += 1
    if cnt == 6:
        return REDWIN
    return False


def makeMove0(p, PawnMoveTo, S):  # 移动棋子，更新地图信息，和棋子存活情况
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


def notInMap(x, y):  # 检测棋子是否在棋盘内移动
    if x in range(0, 5) and y in range(0, 5):
        return False
    return True


def player(n: int) -> str:  # 获取当前下棋方
    if n % 2 == 0:
        return '红方'
    return '蓝方'


def resetInfo0():  # 重置比赛信息
    # global tr, tb
    # tr = tb = 0
    # S.map = getNewMap()
    S.map = getcommap()
    S.pawn = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # 棋子初始化
    S.pro = [1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 /
             6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6]
    # value = getLocValue(S)
    # S.value = getPawnValue(S.pro, value)


def reverse(matrix):
    b = np.reshape(matrix, -1)
    c = b[::-1]
    d = np.reshape(c, [5, 5])
    return d


def roll_dice() -> int:  # 随机摇色子
    return random.randint(1, 6)


def rotate(chess_map):  # 将棋盘旋转180度
    e_1 = np.identity(chess_map.shape[0], dtype=np.int8)[:, ::-1]
    return e_1.dot(chess_map).dot(e_1)


def searchNearbyBlueMaxValue(p, S):  # 搜索附近蓝方最有价值的棋子
    nearby = []
    row, col = getLocation(p, S.map)
    if row + 1 < 5:
        if S.map[row + 1][col] > 6:
            nearby.append(S.value[S.map[row + 1][col] - 1])
    if col + 1 < 5:
        if S.map[row][col + 1] > 6:
            nearby.append(S.value[S.map[row][col + 1] - 1])
    if row + 1 < 5 and col + 1 < 5:
        if S.map[row + 1][col + 1] > 6:
            nearby.append(S.value[S.map[row + 1][col + 1] - 1])
    if nearby == []:
        return 0

    expValue = 0
    for v in nearby:
        expValue += v / sum(nearby)
    # print("the expvalue is",expValue)
    return expValue


def searchNearbyRedMaxValue(p, S):  # 搜索附近红方最有价值的棋子
    nearby = []
    row, col = getLocation(p, S.map)
    if row - 1 >= 0:
        if S.map[row - 1][col] <= 6 and S.map[row - 1][col] > 0:
            nearby.append(S.value[S.map[row - 1][col] - 1])
    if col - 1 >= 0:
        if S.map[row][col - 1] <= 6 and S.map[row][col - 1] > 0:
            nearby.append(S.value[S.map[row][col - 1] - 1])
    if row - 1 >= 0 and col - 1 >= 0:
        if S.map[row - 1][col - 1] <= 6 and S.map[row - 1][col - 1] > 0:
            nearby.append(S.value[S.map[row - 1][col - 1] - 1])
    if nearby == []:
        return 0
    expValue = 0
    for v in nearby:
        expValue += v / sum(nearby)

    return expValue


def selectPawn(S, n=0):  # 掷骰子，挑选可以移动的棋子。n为骰子数，0表示为了模拟
    # global COUNT
    # if n == 0:  # 未传入n说明不是根据现局面与骰子模拟，在模拟后面棋局
    #     count += 1
    #     if count % 2 == 0:  # 偶数是红，奇数是蓝
    #         n = random.randint(1, 6)  # 红
    #     else:
    #         n = random.randint(7, 12)  # 蓝
    #     ans = findNearby(n, S.pawn)
    # else:  # 如果传入了n，说明是为了模拟真实棋局
    ans = findNearby(n, S.pawn)  # n is always > 0  # 2023-11-24
    return n, ans


def selectPawnnewone(S, cnt, n=0):  # 掷骰子，挑选可以移动的棋子。n为骰子数，0表示为了模拟
    # global COUNT
    # count += 1
    print('回合数为：', cnt + 1)
    if n > 0:
        print('骰子数为：', n)
    if n == 0:  # 未传入n说明不是根据现局面与骰子模拟，在模拟后面棋局
        if cnt % 2 == 0:  # 偶数是红，奇数是蓝
            n = random.randint(1, 6)  # 红
        else:
            n = random.randint(7, 12)  # 蓝
        ans = findNearby(n, S.pawn)
    else:  # 如果传入了n，说明是为了模拟真实棋局
        if cnt % 2 == 0:
            ans = findNearby(n, S.pawn)  # red player
        else:
            ans = findNearby(n + 6, S.pawn)  # blue player
    return n, ans


def SoftMax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x


def tryMakeMove0(p, PawnMoveTo, S):  # 尝试移动，并且返回移动后的棋局地图与棋子存活情况
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
