import numpy as np
from class_chess import isEnd, tryMakeMove


def blue_rand(ans, S):
    moveTo = ['leftup', 'up', 'left']
    SL = []  # 当前局面下合法的后续走法
    for p in ans:
        for m in moveTo:
            newStatus = tryMakeMove(p, m, S)
            if newStatus is not False:
                SL.append(newStatus)
                if isEnd(newStatus):
                    return p, m
                del newStatus

    rand_i = np.random.randint(0, len(SL) - 1)
    bestp, bestm = SL[rand_i].pawn, SL[rand_i].move
    print('移动棋子：', bestp, ' 移动方向：', bestm)
    return bestp, bestm
