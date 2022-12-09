# 3주차

# 트리 
# 설명참고 


##########################
# 힙
#힙은 데이터에서 최대값과 최소값을 빠르게 찾기 위해 고안된 완전 이진 트리

# 맥스 힙에 원소 추가
class MaxHeap:
    def __init__(self):
        self.items = [None]

    def insert(self, value):
        # 구현해보세요!
        self.items.append(value) #노드를 맨 마지막에 넣기
        cur_index = len(self.items) - 1 #가장 마지막에 넣은 인덱스

        while cur_index > 1: #cur_index 가 1이 되면 정상을 찍은것, 다른 것과 비교하지 않아도 됨
            parent_index = cur_index // 2 #부모 index 찾기
            if self.items[parent_index] < self.items[cur_index]: #부모 보다 넣은 노드가 크면 바꿔주기
                self.items[parent_index], self.items[cur_index] = self.items[cur_index], self.items[parent_index]
                cur_index = parent_index #그래서 cur_index 는 parent_index로 바꿔치기
            else:
                break
        return


max_heap = MaxHeap()
max_heap.insert(3)
max_heap.insert(4)
max_heap.insert(2)
max_heap.insert(9)
print(max_heap.items)  # [None, 9, 4, 2, 3] 가 출력되어야 합니다!


#맥스 힙의 원소 제거
class MaxHeap:
    def __init__(self):
        self.items = [None]

    def insert(self, value):
        self.items.append(value)
        cur_index = len(self.items) - 1

        while cur_index > 1:  # cur_index 가 1이 되면 정상을 찍은거라 다른 것과 비교 안하셔도 됩니다!
            parent_index = cur_index // 2
            if self.items[parent_index] < self.items[cur_index]:
                self.items[parent_index], self.items[cur_index] = self.items[cur_index], self.items[parent_index]
                cur_index = parent_index
            else:
                break

    def delete(self):
        # 구현해보세요!
        self.items[1], self.items[-1] = self.items[-1], self.items[1] #맨 끝과 맨 앞에 루트 노드의 자리를 바꿔주기
        prev_max = self.items.pop() #맨뒤의 원소를 뽑아서 담는 변수
        cur_index = 1 #루트 노드의 위치 인덱스

        while cur_index <= len(self.items) -1:  #cur_index가 현재 가지고 있는 노드들의 index에서 벗어나지 않을때까지
            left_child_index = cur_index * 2
            right_child_index = cur_index * 2 + 1
            max_index = cur_index

            if left_child_index <= len(self.items) - 1 and self.items[left_child_index] > self.items[max_index]:
                #왼쪽에 자식이 있고, 부모보다 크면
                max_index = left_child_index

            if right_child_index <= len(self.items) - 1 and self.items[right_child_index] > self.items[max_index]:
                #오른쪽에 자식이 있고, 부모보다 크면
                max_index = right_child_index

            if max_index == cur_index: #현재 있는 노드가 자식들보다 크다. while문 멈추기
                break

            self.items[cur_index], self.items[max_index] = self.items[max_index], self.items[cur_index]
            cur_index = max_index


        return prev_max  # 8 을 반환해야 합니다.


max_heap = MaxHeap()
max_heap.insert(8)
max_heap.insert(6)
max_heap.insert(7)
max_heap.insert(2)
max_heap.insert(5)
max_heap.insert(4)
print(max_heap.items)  # [None, 8, 6, 7, 2, 5, 4]
print(max_heap.delete())  # 8 을 반환해야 합니다!
print(max_heap.items)  # [None, 7, 6, 4, 2, 5]



##########################
# 그래프
# 연결되어 있는 정점와 정점간의 관계를 표현할 수 있는 자료구조
#설명 참고



##########################
# DFS & BFS
# Depth First Search => 구현하기 - 재귀함수

graph = {
    1: [2, 5, 9],
    2: [1, 3],
    3: [2, 4],
    4: [3],
    5: [1, 6, 8],
    6: [5, 7],
    7: [6],
    8: [5],
    9: [1, 10],
    10: [9]
}
visited = []


def dfs_recursion(adjacent_graph, cur_node, visited_array):
    # 구현해보세요!
    visited_array.append(cur_node)
    for adjacent_node in adjacent_graph[cur_node]:
        if adjacent_node not in visited_array: #인접한 노드가 없다면, 방문하지 않았다면 == 탈출 조건
            dfs_recursion(adjacent_graph, adjacent_node, visited_array)

dfs_recursion(graph, 1, visited)  # 1 이 시작노드입니다!
print(visited)  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 이 출력되어야 합니다!


# Depth First Search => 구현하기 - 스택
# 위의 그래프를 예시로 삼아서 인접 리스트 방식으로 표현했습니다!
graph = {
    1: [2, 5, 9],
    2: [1, 3],
    3: [2, 4],
    4: [3],
    5: [1, 6, 8],
    6: [5, 7],
    7: [6],
    8: [5],
    9: [1, 10],
    10: [9]
}


def dfs_stack(adjacent_graph, start_node):
    # 구현해보세요!
    stack = [start_node] #시작하는 노드를 넣기
    visited = []
    while stack: #stack이 비지 않을때까지
        current_node = stack.pop() #현재 stack의 노드를 빼서 담기
        visited.append(current_node) #visited 배열에 위에 뺀 것을 담는 배열
        for adjacent_node in adjacent_graph[current_node]: #그래프에서 current_node의 인접한 노드를 꺼내서 반복문
            if adjacent_node not in visited: #노드가 방문한 목록에 없다면
                stack.append(adjacent_node) 

    return visited


print(dfs_stack(graph, 1))  # 1 이 시작노드입니다!
# [1, 9, 10, 5, 8, 6, 7, 2, 3, 4] 이 출력되어야 합니다!



##########################
# DFS & BFS
# BFS - 큐
# 위의 그래프를 예시로 삼아서 인접 리스트 방식으로 표현했습니다!
graph = {
    1: [2, 3, 4],
    2: [1, 5],
    3: [1, 6, 7],
    4: [1, 8],
    5: [2, 9],
    6: [3, 10],
    7: [3],
    8: [4],
    9: [5],
    10: [6]
}


def bfs_queue(adj_graph, start_node):
    # 구현해보세요!
    queue = [start_node] #시작 노드
    visited = []
    while queue: #queue 가 비지 않을때까지
        current_node = queue.pop(0) #0번째 원소 빼주기
        visited.append(current_node) 
        for adj_node in adj_graph[current_node]: # 인접 그래프에서 인접한 노드들만 반복문
            if adj_node not in visited: #방문했는지 확인
                queue.append(adj_node)
    return visited
    


print(bfs_queue(graph, 1))  # 1 이 시작노드입니다!
# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 이 출력되어야 합니다!




##########################
# Dynamic Programming == 동적 계획법

# 피보나치 수열 - 재귀 함수
# 수학에서 피보나치 수는 첫째 및 둘째 항이 1이며 그 뒤의 모든 항은 바로 앞 두 항의 합인 수열
# 단점 => 수가 크면 클수록 결과값이 나오는데 오래걸림
input = 20


def fibo_recursion(n):
    # 구현해보세요!
    if n == 1 or n == 2: #1이나 2 번째 1 탈출 조건
        return 1
    return fibo_recursion(n - 1) + fibo_recursion(n - 2)


print(fibo_recursion(input))  # 6765

# 동적 계획법이란?
# 복잡한 문제를 간단한 여러 개의 문제로 나누어 푸는 방법
input = 50

# memo 라는 변수에 Fibo(1)과 Fibo(2) 값을 저장해놨습니다!
memo = {
    1: 1,
    2: 1
}


def fibo_dynamic_programming(n, fibo_memo):
    # 구현해보세요!
    if n in fibo_memo: #n이 메모에 있으면
        return fibo_memo[n] #바로 반환

    nth_fibo = fibo_dynamic_programming(n - 1, fibo_memo) + fibo_dynamic_programming(n - 2, fibo_memo) #재귀적으로 찾아주기
    fibo_memo[n] = nth_fibo #찾는 메모를 위에 값으로 넣어주기
    return nth_fibo


print(fibo_dynamic_programming(input, memo))




##########################
#숙제

#농심 라면 공장
import heapq

ramen_stock = 4
supply_dates = [4, 10, 15]
supply_supplies = [20, 5, 10]
supply_recover_k = 30


def get_minimum_count_of_overseas_supply(stock, dates, supplies, k):
    # 풀어보세요!
    answer = 0 #가장 최소로 공급 받을 수 있는 stock
    last_added_date_index = 0 #가장 마지막에 더했던 날짜의 인덱스
    max_heap = []

    while stock <= k: #stock이 k보다 낮을때까지
        while last_added_date_index < len(dates) and dates[last_added_date_index] <= stock: 
            heapq.heappush(max_heap, -supplies[last_added_date_index]) #0번째 인덱스에 supplies를 max_heap에 담음
            last_added_date_index += 1

        answer += 1
        heappop = heapq.heappop(max_heap) #max_heap에서 가작 최솟값을 뽑음 <-> -를 통해 최댓값으로 변환
        stock += -heappop 

    return answer


print(get_minimum_count_of_overseas_supply(ramen_stock, supply_dates, supply_supplies, supply_recover_k))
print("정답 = 2 / 현재 풀이 값 = ", get_minimum_count_of_overseas_supply(4, [4, 10, 15], [20, 5, 10], 30))
print("정답 = 4 / 현재 풀이 값 = ", get_minimum_count_of_overseas_supply(4, [4, 10, 15, 20], [20, 5, 10, 5], 40))
print("정답 = 1 / 현재 풀이 값 = ", get_minimum_count_of_overseas_supply(2, [1, 10], [10, 100], 11))


#샤오미 로봇 청소기
current_r, current_c, current_d = 7, 4, 0
current_room_map = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

# 북 동 남 서
dr = [-1, 0, 1, 0] #row
dc = [0, 1, 0, -1] #column

# 방향 전환
def get_left(d):
    return (d + 3) % 4

# 후진
def get_back(d):
    return (d + 2) % 4


def get_count_of_departments_cleaned_by_robot_vacuum(r, c, d, room_map):
    n = len(room_map) #room_map의 길이
    m = len(room_map[0]) #room_map의 0번째 원소(첫번째 칸의 칼럼들)
    count_cleaned = 1 #청소하는 칸의 개수
    room_map[r][c] = 2 #r이라는 row와 c라는 column 2로 업데이트 => 행렬
    queue = list([[r, c, d]]) #현재 위치와 방향을 저장

    # queue가 비어지면 종료
    while queue:
        r, c, d = queue.pop(0)
        temp_d = d #현재의 방향

        for i in range(4): #모든 방향에 대해서 회전
            temp_d = get_left(temp_d) #현재의 방향에서 왼쪽으로 회전하는 값
            new_r, new_c = r + dr[temp_d], c + dc[temp_d] #왼쪽에서 가는 방향 => 새로운 r, c 값

                # a
            if 0 <= new_r < n and 0 <= new_c < m and room_map[new_r][new_c] == 0: #갈 수 있는지 없는지에 대한 수식 => 벽이 아닌지, 청소하지 않은 칸인지
                count_cleaned += 1
                room_map[new_r][new_c] = 2 #해당 칸을 청소하였는지 기록
                queue.append([new_r, new_c, temp_d]) #위치와 방향 정보를 queue에 저장
                break #다음 탐색 시작

            # c
            elif i == 3:  # 갈 곳이 없었던 경우(후진) => range 0-3 인데 마지막 까지 갔으면 갈 곳이 없음
                new_r, new_c = r + dr[get_back(d)], c + dc[get_back(d)] #현재 r, c 에서 get_back을 추가
                queue.append([new_r, new_c, d]) #r, c에 적용

                # d
                if room_map[new_r][new_c] == 1:  # 뒤가 벽인 경우
                    return count_cleaned


# 57 가 출력되어야 합니다!
print(get_count_of_departments_cleaned_by_robot_vacuum(current_r, current_c, current_d, current_room_map))
current_room_map2 = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]
print("정답 = 29 / 현재 풀이 값 = ", get_count_of_departments_cleaned_by_robot_vacuum(6,3,1,current_room_map2))
current_room_map3 = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 1, 0, 1],
    [1, 0, 0, 1, 0, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]
print("정답 = 33 / 현재 풀이 값 = ", get_count_of_departments_cleaned_by_robot_vacuum(7,4,1,current_room_map3))
current_room_map4 = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 1, 0, 1],
    [1, 0, 0, 1, 0, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]
print("정답 = 25 / 현재 풀이 값 = ", get_count_of_departments_cleaned_by_robot_vacuum(6,2,0,current_room_map4))


# #CGV 극장 좌석 자리 구하기
seat_count = 9
vip_seat_array = [4, 7]


memo = {
    1: 1, 
    2: 2
}

def fibo_dynamic_programming(n, fibo_memo):
    if n in fibo_memo:
        return fibo_memo[n]

    nth_fibo = fibo_dynamic_programming(n - 1, fibo_memo) + fibo_dynamic_programming(n - 2, fibo_memo)
    fibo_memo[n] = nth_fibo
    return nth_fibo

# F(N) = N 명의 사람들을 좌석에 배치하는 방법
#        N-1 명의 사람들을 좌석에 배치하는 방법 + N-2 명의 사람들을 좌석에 배치하는 방법
#        = F(N - 1) + F(N -2)

def get_all_ways_of_theater_seat(total_count, fixed_seat_array): 
    all_ways = 1 #아무자리도 아님 = 1
    current_index = 0 #맨 처음에 시작하는 index
    for fixed_seat in fixed_seat_array: 
        fixed_seat_index = fixed_seat - 1 #인덱스를 번호로 만들기 위해 -1 => [4, 7]
        count_of_ways = fibo_dynamic_programming(fixed_seat_index - current_index, memo) #사이에 있는 좌석의 갯수
        all_ways *= count_of_ways #곱연산
        current_index = fixed_seat_index + 1 #다음 인덱스를 보게하기 위함
    
    count_of_ways = fibo_dynamic_programming(total_count - current_index, memo) #뒤에 남아있을 좌석들
    all_ways *= count_of_ways #곱연산

    return all_ways


# 12가 출력되어야 합니다!
print(get_all_ways_of_theater_seat(seat_count, vip_seat_array))
