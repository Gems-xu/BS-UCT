import torch
from torch.autograd import Variable

from class_chess import *
from models.try_resnet_0706 import BasicBlock
from models.try_resnet_0706 import ResNet
from models.try_resnet_0713_value import ResNet as ResNetValue

# calculation_time = float(80)  # 计算时间
# num_process = 8  # 多进程数目
max_actions = 1000  # 最大行动数
device = torch.device("cuda")
pathNet = 'models/0830_1111_Alpha1.pt'
pathValueNet = 'models/0830_1111_Alpha1_value_little.pt'


def selectPawn_run_sim_net(S, n=0, _count=0):  # 掷骰子，挑选可以移动的棋子。n为骰子数，0表示为了模拟
    if n == 0:  # 未传入n说明不是根据现局面与骰子模拟，在模拟后面棋局
        _count += 1
        if _count % 2 == 0:  # 偶数是红，奇数是蓝
            n = random.randint(1, 6)  # 红
        else:
            n = random.randint(7, 12)  # 蓝
        ans = findNearby(n, S.pawn)
    else:  # 如果传入了n，说明是为了模拟真实棋局
        ans = findNearby(n, S.pawn)
    return n, ans, _count


def getTheNextStepStatus_updata_run_sim_net(SL, count):  # 根据现局面，获得所有合法的后续局面
    NL = []
    if SL[0].pPawn > 6:
        move = ['right', 'down', 'rightdown']
        o = 0
    else:
        move = ['left', 'up', 'leftup']
        o = 6
    for s in SL:
        i = random.randint(1, 6)
        n, ans, count = selectPawn_run_sim_net(s, i + o, _count=count)
        for p in ans:
            for m in move:
                newStatus = tryMakeMove0(p, m, s)
                if newStatus is not False:
                    newStatus.pDice = i
                    NL.append(newStatus)
                del newStatus
    return NL, count


def run_simulation_network(inputs):  # 红方
    # global winsr
    # global playsr
    # global S
    # global Vsr
    # global Vsr_all
    # global recorder
    # playsr, Vsr, availables, datanew, games, c, max_actions = inputs
    playsr, Vsr, availables, datanew, max_actions = inputs
    move_dict = {'right': 0, 'down': 1, 'rightdown': 2}

    # 注意网络层数对应的定义
    # device = torch.device("cuda")
    net = (ResNet(BasicBlock, [1, 1, 1, 1])
           .cuda(device))
    net.load_state_dict(torch.load(pathNet))
    net.eval()

    valuenet = (ResNetValue(BasicBlock, [1, 1, 1, 1], 1)
                .cuda(device))
    valuenet.load_state_dict(torch.load(pathValueNet))
    valuenet.eval()

    temperature = 0.1
    temperature_demo = 2.0
    count = 0  # 同 COUNT

    Qsr = {}
    # availables = SL  # 合法后继
    visited_states = set()  # 已访问节点，判断是否拓展
    visited_red = set()
    # playsr[S] += 1

    datanew = torch.tensor(np.array(datanew)).cuda(device)
    datanew = Variable(torch.unsqueeze(datanew, dim=0).float(), requires_grad=False)
    outnew = net(datanew)

    prenew = outnew
    out = torch.zeros(25)
    prenew = prenew.squeeze()
    # out = self.trans(out)
    out = out.cuda(device)
    # print(out)
    for i in range(0, 6):
        oneStatus = prenew[i * 4:(i + 1) * 4]
        out[i * 4:(i + 1) * 4] = 3 * torch.softmax(oneStatus, 0)
        for index, value in enumerate(out[i * 4:(i + 1) * 4]):
            if index < 3:
                if value < 0.2:
                    out[i * 4 + index] = 0.2
    out = out.tolist()
    # probli = []
    consider = {}

    for move in availables:
        move_p = move.pPawn
        move_to = move.pMove
        problis = out[(move_p - 1) * 4:move_p * 4 - 1]  # move = ['right', 'down', 'rightdown']
        probli = problis[move_dict[move_to]]
        # if games == 0:
        #     print(move.pMove, probli)
        C = cal_first_choice(probli, Vsr, move, 9, _playsr=playsr)
        consider[move] = C
    The_total_choose = max(consider, key=lambda x: consider[x])  # 最大值对应的键
    del consider
    move_choose = copy.deepcopy(The_total_choose)

    # 激活已选节点在playsr中，winsr中
    # if playsr.get(The_total_choose, -1) == -1:
    #     playsr[The_total_choose] = 0
    #     winsr[The_total_choose] = 0
    visited_states.add(The_total_choose)
    k = 0  # 判断是否结束
    if move_choose is not False:
        k = isEnd(move_choose)
        if k == False:
            for t in range(1, max_actions):
                NL = []
                NL.append(move_choose)
                availables, count = getTheNextStepStatus_updata_run_sim_net(NL, count)
                del NL
                for move in availables:
                    if move is not False:
                        k = isEnd(move)
                        if k:
                            # move_otherside = move
                            break
                if k:
                    break

                max_problis = []
                consider = {}
                max_consider = []
                for move in availables:
                    theValue = getDemoValueblue(move)
                    consider[move] = theValue
                    max_problis.append(theValue)
                    max_consider.append(move)

                max_problis = np.array(max_problis)
                move_otherside = random.choices(max_consider, SoftMax(max_problis / temperature_demo), k=1)
                move_otherside = move_otherside[0]

                visited_states.add(move_otherside)
                # if playsr.get(move_otherside, -1) == -1:
                #     playsr[move_otherside] = 0
                #     winsr[move_otherside] = 0
                if move_otherside is not False:
                    k = isEnd(move_otherside)
                    if k:
                        break
                NL = []
                NL.append(move_otherside)
                availables, count = getTheNextStepStatus_updata_run_sim_net(NL, count)
                del NL

                for move in availables:
                    if move is not False:
                        k = isEnd(move)
                        if k:
                            # move_choose = move
                            break
                if k:
                    break

                datanew = get_alldata_5(move_otherside)
                datanew = torch.tensor(np.array(datanew)).cuda(device)
                datanew = Variable(torch.unsqueeze(datanew, dim=0).float(), requires_grad=False)
                outnew = net(datanew)

                prenew = outnew
                out = torch.zeros(25)
                prenew = prenew.squeeze()
                # out = self.trans(out)
                out = out.cuda(device)
                # print(out)
                for i in range(0, 6):
                    oneStatus = prenew[i * 4:(i + 1) * 4]
                    out[i * 4:(i + 1) * 4] = 3 * torch.softmax(oneStatus, 0)
                    for index, value in enumerate(out[i * 4:(i + 1) * 4]):
                        if index < 3:
                            if value < 0.2:
                                out[i * 4 + index] = 0.2
                out = out.tolist()
                # probli = []
                consider = {}
                max_consider = []
                max_problis = []

                for move in availables:
                    move_p = move.pPawn
                    move_to = move.pMove
                    problis = out[(move_p - 1) * 4:move_p * 4 - 1]  # move = ['right', 'down', 'rightdown']
                    probli = problis[move_dict[move_to]]
                    max_problis.append(probli)
                    consider[move] = probli
                list_consider_value = list(consider.values())
                list_consider_keys = list(consider.keys())
                for one_pro in max_problis:
                    position = list_consider_value.index(one_pro)
                    max_consider.append(list_consider_keys[position])
                max_problis = torch.tensor(max_problis)

                move_choose = random.choices(max_consider, torch.softmax(max_problis / temperature, 0), k=1)

                move_choose = move_choose[0]

                Qsr[move_choose] = valuenet(datanew).tolist()[0][0]
                # Qsr[move_choose] = 0
                visited_red.add(move_choose)

                # 激活已选节点在playsr中，winsr中
                # if playsr.get(move_choose, -1) == -1:
                #     playsr[move_choose] = 0
                #     winsr[move_choose] = 0
                visited_states.add(move_choose)
                k = 0
                if move_choose is not False:
                    k = isEnd(move_choose)
                    if k:
                        break

                """
                    此处是要更新新的局面数据，以传入网络进行估值
                """

                """
                    把移动过的点加入进访问元组中，方便日后查询
                """

    return Qsr, The_total_choose, k, visited_states, visited_red


def run_simulation_network_t(inputs):  # 红方
    # global winsr
    # global playsr
    # global S
    # global Vsr
    # global Vsr_all
    # global recorder
    # playsr, Vsr, availables, datanew, games, c, max_actions = inputs
    playsr, Vsr, availables, datanew, max_actions = inputs
    move_dict = {'right': 0, 'down': 1, 'rightdown': 2}

    # 注意网络层数对应的定义
    # device = torch.device("cuda")
    net = (ResNet(BasicBlock, [1, 1, 1, 1])
           .cuda(device))
    net.load_state_dict(torch.load(pathNet))
    net.eval()

    valuenet = (ResNetValue(BasicBlock, [1, 1, 1, 1], 1)
                .cuda(device))
    valuenet.load_state_dict(torch.load(pathValueNet))
    valuenet.eval()

    temperature = 0.1
    temperature_demo = 2.0
    count = 0  # 同 COUNT

    Qsr = {}
    # availables = SL  # 合法后继
    visited_states = set()  # 已访问节点，判断是否拓展
    visited_red = set()
    # playsr[S] += 1

    datanew = torch.tensor(np.array(datanew)).cuda(device)
    datanew = Variable(torch.unsqueeze(datanew, dim=0).float(), requires_grad=False)
    outnew = net(datanew)

    prenew = outnew
    out = torch.zeros(25)
    prenew = prenew.squeeze()
    # out = self.trans(out)
    out = out.cuda(device)
    # print(out)
    for i in range(0, 6):
        oneStatus = prenew[i * 4:(i + 1) * 4]
        out[i * 4:(i + 1) * 4] = 3 * torch.softmax(oneStatus, 0)
        for index, value in enumerate(out[i * 4:(i + 1) * 4]):
            if index < 3:
                if value < 0.2:
                    out[i * 4 + index] = 0.2
    out = out.tolist()
    # probli = []
    consider = {}

    for move in availables:
        move_p = move.pPawn
        move_to = move.pMove
        problis = out[(move_p - 1) * 4:move_p * 4 - 1]  # move = ['right', 'down', 'rightdown']
        probli = problis[move_dict[move_to]]
        # if games == 0:
        #     print(move.pMove, probli)
        C = cal_first_choice(probli, Vsr, move, 9, _playsr=playsr)
        consider[move] = C
    The_total_choose = max(consider, key=lambda x: consider[x])  # 最大值对应的键
    del consider
    move_choose = copy.deepcopy(The_total_choose)

    # 激活已选节点在playsr中，winsr中
    # if playsr.get(The_total_choose, -1) == -1:
    #     playsr[The_total_choose] = 0
    #     winsr[The_total_choose] = 0
    visited_states.add(The_total_choose)
    k = 0  # 判断是否结束
    times = 1  # 步长
    if move_choose is not False:
        k = isEnd(move_choose)
        if k == False:
            for t in range(1, max_actions):
                NL = []
                NL.append(move_choose)
                availables, count = getTheNextStepStatus_updata_run_sim_net(NL, count)
                del NL
                for move in availables:
                    if move is not False:
                        k = isEnd(move)
                        if k:
                            # move_otherside = move
                            times = t
                            break
                if k:
                    times = t
                    break

                max_problis = []
                consider = {}
                max_consider = []
                for move in availables:
                    theValue = getDemoValueblue(move)
                    consider[move] = theValue
                    max_problis.append(theValue)
                    max_consider.append(move)

                max_problis = np.array(max_problis)
                move_otherside = random.choices(max_consider, SoftMax(max_problis / temperature_demo), k=1)
                move_otherside = move_otherside[0]

                visited_states.add(move_otherside)
                # if playsr.get(move_otherside, -1) == -1:
                #     playsr[move_otherside] = 0
                #     winsr[move_otherside] = 0
                if move_otherside is not False:
                    k = isEnd(move_otherside)
                    if k:
                        times = t
                        break
                NL = []
                NL.append(move_otherside)
                availables, count = getTheNextStepStatus_updata_run_sim_net(NL, count)
                del NL

                for move in availables:
                    if move is not False:
                        k = isEnd(move)
                        if k:
                            # move_choose = move
                            times = t
                            break
                if k:
                    times = t
                    break

                datanew = get_alldata_5(move_otherside)
                datanew = torch.tensor(np.array(datanew)).cuda(device)
                datanew = Variable(torch.unsqueeze(datanew, dim=0).float(), requires_grad=False)
                outnew = net(datanew)

                prenew = outnew
                out = torch.zeros(25)
                prenew = prenew.squeeze()
                # out = self.trans(out)
                out = out.cuda(device)
                # print(out)
                for i in range(0, 6):
                    oneStatus = prenew[i * 4:(i + 1) * 4]
                    out[i * 4:(i + 1) * 4] = 3 * torch.softmax(oneStatus, 0)
                    for index, value in enumerate(out[i * 4:(i + 1) * 4]):
                        if index < 3:
                            if value < 0.2:
                                out[i * 4 + index] = 0.2
                out = out.tolist()
                # probli = []
                consider = {}
                max_consider = []
                max_problis = []

                for move in availables:
                    move_p = move.pPawn
                    move_to = move.pMove
                    problis = out[(move_p - 1) * 4:move_p * 4 - 1]  # move = ['right', 'down', 'rightdown']
                    probli = problis[move_dict[move_to]]
                    max_problis.append(probli)
                    consider[move] = probli
                list_consider_value = list(consider.values())
                list_consider_keys = list(consider.keys())
                for one_pro in max_problis:
                    position = list_consider_value.index(one_pro)
                    max_consider.append(list_consider_keys[position])
                max_problis = torch.tensor(max_problis)

                move_choose = random.choices(max_consider, torch.softmax(max_problis / temperature, 0), k=1)

                move_choose = move_choose[0]

                Qsr[move_choose] = valuenet(datanew).tolist()[0][0]
                # Qsr[move_choose] = 0
                visited_red.add(move_choose)

                # 激活已选节点在playsr中，winsr中
                # if playsr.get(move_choose, -1) == -1:
                #     playsr[move_choose] = 0
                #     winsr[move_choose] = 0
                visited_states.add(move_choose)
                k = 0
                if move_choose is not False:
                    k = isEnd(move_choose)
                    if k:
                        times = t
                        break

                """
                    此处是要更新新的局面数据，以传入网络进行估值
                """

                """
                    把移动过的点加入进访问元组中，方便日后查询
                """

    return Qsr, The_total_choose, k, visited_states, visited_red, times