import numpy as np
import heapq
import copy
import time

class Node:
    def __init__(self, state : np.ndarray, parent = None , move : int = -1, depth : int = 0):
        #父节点的作用是不断向上回溯找到最佳路径
        self.state = state
        self.parent = parent
        self.move = move
        self.depth = depth
        self.manhattan_distance = self.calculate_manhattan_distance()
    
    def calculate_manhattan_distance(self) -> int:
        distance = 0
        h2 = 0
        for i in range(4):
            for j in range(4):
                if self.state[i * 4 + j] != 0:
                    row_goal = (self.state[i * 4 + j] - 1) // 4
                    col_goal = (self.state[i * 4 + j] - 1) % 4
                    distance += abs(i - row_goal) + abs(j - col_goal)
                    if j == col_goal:
                        for k in range(j + 1, 4):
                            if self.state[i * 4 + k] != 0 and (self.state[i * 4 + k] - 1) // 4 == i and (
                                    self.state[i * 4 + k] - 1) % 4 < j:  # 同一行的右边某个点正确位置是在他的正左边
                                h2 += 2
                    if i == row_goal:
                        for k in range(i + 1, 4):
                            if self.state[k * 4 + j] != 0 and (self.state[k * 4 + j] - 1) % 4 == j and (
                                    self.state[k * 4 + j] - 1) // 4 < i:  # 同一列的下边某个点正确位置是在他的正上边
                                h2 += 2
        return distance + h2
    

    def __lt__(self, other):
        return (self.depth + self.manhattan_distance) < (other.depth + other.manhattan_distance)
    
    def __eq__(self, other):
        return self.state == other.state

def a_star(initial_state : np.ndarray, goal_state : np.ndarray) -> Node:
    #维护优先队列  open_list  和  已经扩展的节点集合  closed_set
    initial_node = Node(initial_state, None, -1, 0)
    open_list = [initial_node]
    closed_set = set()

    while open_list:
        current_node = heapq.heappop(open_list)
        if current_node.manhattan_distance == 0:        #检查是否到达终点
            return current_node
        closed_set.add(tuple(current_node.state))               #扩展过的节点加入closed_set
        for move, new_state in get_moves(current_node):
            if tuple(new_state) in closed_set:                  #环检测
                continue
            new_node = Node(new_state, current_node, move, current_node.depth + 1)
            heapq.heappush(open_list, new_node)

    return None




def IDA_star(initial_state : np.ndarray, goal_state : np.ndarray) -> Node:
    #每次迭代更新cost_limit的值
    cost_limit = 1
    while cost_limit != 0:
        best_cost = float('inf')
        open_stack = [Node(initial_state, None, -1, 0)]
        while open_stack:
            current_node = open_stack.pop()
            if current_node.manhattan_distance == 0:          #找到最终解
                return current_node
            for move, new_state in get_moves(current_node):
                new_node = Node(new_state, current_node, move, current_node.depth + 1)
                if new_node.manhattan_distance + new_node.depth <= cost_limit:           #未越界的情形
                    open_stack.append(new_node)
                else:
                    best_cost = min(best_cost, new_node.manhattan_distance + new_node.depth)        #越界的情形
        if open_stack == [] and best_cost == float('inf'):
            return None
        if open_stack == [] and best_cost != float('inf'):
            cost_limit = best_cost
            
def get_moves(current_node : Node) -> list:
    state = current_node.state
    moves = []
    empty_index = -1
    for i in range(16):
        if state[i] == 0:
            empty_index = i
            break
    row, col = divmod(empty_index, 4)                       #找到移动前空格所在的行和列
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]         # Up, Down, Left, Right
    for dr, dc in directions:
        new_row, new_col = row + dr, col + dc               #移动后空格所在的行和列
        if 0 <= new_row < 4 and 0 <= new_col < 4:
            new_state = copy.deepcopy(state)
            swapedNum = state[new_row * 4 + new_col]
            if swapedNum == current_node.move:              #如果移动步数相同，跳过该状态
                continue
            new_state[new_row * 4 + new_col] = 0
            new_state[row * 4 + col] = swapedNum
            moves.append((swapedNum, new_state))
    return moves

def read_file(file_path : str) -> np.ndarray:
    file_content = ''
    res = np.zeros((16), int)
    with open(file_path, 'r') as file:
        file_content = file.read()
    lines = file_content.strip().split('\n')
    for i in range(4):
        line = lines[i].strip().split(' ')
        for j in range(4):
            res[i * 4 + j] = int(line[j])
    return res

def print_state(state : np.ndarray):
    for i in range(4):
        print(state[i], end = ' ')
    print()
    for i in range(4,8):
        print(state[i], end = ' ')
    print()
    for i in range(8,12):
        print(state[i], end = ' ')
    print()
    for i in range(12,16):
        print(state[i], end = ' ')
      
def printSolution(solution : Node):
    if solution:
        path = []
        while solution:
            path.insert(0, (solution.move, solution.state))
            solution = solution.parent                          #沿着父节点向上回溯
        print(f'steps = {len(path) - 1}')
        print('initial state:')
        print_state(path[0][1])
        print()
        print("Action sequences: ")
        for i in range(1, len(path)):
            print(path[i][0], end = ' ')
        print()
        for move, state in path:
            if move == -1:
                continue
            print(f'moved num: {move}')
            print_state(state)
            print('\n---------------')
        
    else:
        print("No solution found.")

goal_state = np.array([1, 2, 3, 4,
                       5, 6, 7, 8,
                       9, 10, 11, 12,
                       13, 14, 15, 0])


def A_star_test():
    start_time = time.time()
    file_path = r'1.txt'
    print(f'Test using A_star for {file_path}:')
    initial_state = read_file(file_path)
    solution = a_star(initial_state, goal_state)
    end_time = time.time()
    print(f'running time = {end_time - start_time} seconds')
    printSolution(solution)


    
def IDA_star_test_without_detection():
    start_time = time.time()
    file_path = r'1.txt'
    print(f'Test using IDA_star_without_detection for {file_path}:')
    initial_state = read_file(file_path)
    solution = IDA_star(initial_state, goal_state)
    end_time = time.time()
    print(f'running time = {end_time - start_time} seconds')
    printSolution(solution)
    

A_star_test()
IDA_star_test_without_detection()
