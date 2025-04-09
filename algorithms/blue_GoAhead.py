from class_chess import findNearby, tryMakeMove


def select_pawn_new_one_goahead(COUNT, S, n=0):  # 掷骰子，挑选可以移动的棋子。n为骰子数
    ans = findNearby(n + 6, S.pawn)  # +6 表示蓝方棋子
    return n, ans, COUNT + 1


def blueByBraveOfMan(ans, S):
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
