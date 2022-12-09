# 5주차

# 2019년 상반기 LINE 인턴 시험
# 나 잡아 봐라 => 모든 경우의 수 나열 (== BFS)
# 브라운이 코니를 잡기 (코니는 반복문으로 가능 브라운의 위치가 어려움)

# 규칙적 => 배열, 자유자재 => 딕셔너리

# 각 시간마다 브라운이 갈 수 있는 위치를 저장
# [{}] => 배열 안에 딕셔너리

from collections import deque

c = 11
b = 2


def catch_me(cony_loc, brown_loc):
    # 구현해보세요!
    time = 0 #같은 시간대에 만나야하기 때문에 시간이라는 개념이 필요함
    queue = deque() #위에 queue 가져오기
    queue.append((brown_loc, 0)) #초기값 => 위치와 시간을 담아주기 
    visited = [{} for _ in range(200001)] #배열 안에 200001 값을 무시하기 위함 => []위치 {}시간
    #visited[위치][시간] => visited 배열의 위치 안에 시간

    while cony_loc <= 200000: #탈출 조건
        cony_loc += time #코니의 위치 : 시간만큼 더해주기 +1 +2 +3...
        if time in visited[cony_loc]: #같은 시간대에 방문했는지 확인
            return time

        for i in range(0, len(queue)): #while을 안쓰는 이유는? :
            current_position, current_time = queue.popleft() #맨 왼쪽, 현재 위치와 현재 시간

            new_time = current_time + 1 #새로운 시간
            new_position = current_position - 1 #새로운 위치
            if 0 <= new_position <= 200000: #0 < n <= 20000 조건 => 실패하는 경우 / 성공은 시간과 위치가 동일
                visited[new_position][new_time] = True
                queue.append((new_position, new_time)) 

            new_position = current_position + 1
            #if new_position < 20001 and new_time not in visited[new_position]:
            if 0 <= new_position <= 200000:
                visited[new_position][new_time] = True
                queue.append((new_position, new_time))

            new_position = current_position * 2
            if 0 <= new_position <= 200000:
                visited[new_position][new_time] = True
                queue.append((new_position, new_time))

        time += 1


print(catch_me(c, b))  # 5가 나와야 합니다!

print("정답 = 3 / 현재 풀이 값 = ", catch_me(10,3))
print("정답 = 8 / 현재 풀이 값 = ", catch_me(51,50))
print("정답 = 28 / 현재 풀이 값 = ", catch_me(550,500))


##########################
# 2020 카카오 신입 개발자 블라인드 채용 - 1
# 문자열 압축
input = "abcabcabcabcdededededede"


def string_compression(string):
    n = len(string) #문자열의 길이
    compression_length_array = []

    for split_size in range(1, n // 2 + 1): #0부터 반복할 필요가 없으니 1, +1을 해야지 N // 2까지 감
        splited=[#배열로 만듬
            string[i:i + split_size] for i in range(0, n ,split_size) #2단위로 쪼갤수 있음 / 0부터 n까지 split_size까지 추가
            #for문을 뒤에 사용 가능
        ] 
        compressed = ""
        count = 1 #초기값은 이전값과 본인을 비교하기때문에 나와있는 본인 포함 1
        
        for j in range(1, len(splited)): 
            prev, cur = splited[j - 1], splited[j] #이전값과 현재값을 비교
            if prev == cur: #둘이 같으면
                count += 1
            else: #이전 문자와 다르면
                if count > 1:
                    compressed += (str(count) + prev)
                else: # 문자가 반복되지 않아 한번만 나타난 경우 1은 생략함
                    compressed += prev
                count = 1
        if count > 1:
            compressed += (str(count) + splited[-1]) #이전이 없기 때문에 -1번째 원소
        else: 
            compressed += splited[-1]
        compression_length_array.append(len(compressed))
        
    return min(compression_length_array) #최솟값 리턴


print(string_compression(input))  # 14 가 출력되어야 합니다!

print("정답 = 3 / 현재 풀이 값 = ", string_compression("JAAA"))
print("정답 = 9 / 현재 풀이 값 = ", string_compression("AZAAAZDWAAA"))
print("정답 = 12 / 현재 풀이 값 = ", string_compression('BBAABAAADABBBD'))



##########################
# 2020 카카오 신입 개발자 블라인드 채용 - 2
from collections import deque

balanced_parentheses_string = "()))((()"

def is_correct_parentheses(string):
    stack = []
    for s in string: #문자열 하나하나 비교
        if s == '(': #열린건지 비교
            stack.append(s) #추가
        elif stack:
            stack.pop() #빼주기
    return len(stack) == 0 #아무것도 안남아 있다면 올바른 문자

def reverse_parentheses(string): #v안에 있는 애 정리해주기
    reversed_string = ""
    for char in string[1:-1]: #u를 첫번째부터 마지막 하나 제거한데까지 반복
        if char == '(':
            reversed_string += ')'
        else:
            reversed_string += '('
    return reversed_string

def separate_u_v(string):# u, v로 분리
    queue = deque(string)
    left, right = 0, 0 #초기화
    u, v = "", ""

    while queue:
        char = queue.popleft()
        u += char
        if char == '(': #열린거라면
            left += 1 #왼쪽 괄호열 추가
        else: #아니라면
            right += 1
        if left == right: #균형 잡힌 괄호 문자열일때
            break

    v = ''.join(list(queue)) #문자열 합치기 ''안에 괄호 안에 모두 합쳐주기
    return u, v

def change_to_correct_parentheses(string): #균형잡힌 괄호 문자열을 올바른 괄호 문자열로 바꾸는 함수
    if string == '': #1번
        return ''
    u, v =separate_u_v(string)

    if is_correct_parentheses(u):
        return u + change_to_correct_parentheses(v)
    else:                
        return '(' + change_to_correct_parentheses(v) + ')' + reverse_parentheses(u[1:-1])


def get_correct_parentheses(balanced_parentheses_string):
    if is_correct_parentheses(balanced_parentheses_string): #애초부터 올바른 문자열이라면
        return balanced_parentheses_string #그대로 반환
    else: 
        return change_to_correct_parentheses(balanced_parentheses_string)


print(get_correct_parentheses(balanced_parentheses_string))  # "()(())()"가 반환 되어야 합니다!

print("정답 = (((()))) / 현재 풀이 값 = ", get_correct_parentheses(")()()()("))
print("정답 = ()()( / 현재 풀이 값 = ", get_correct_parentheses("))()("))
print("정답 = ((((()())))) / 현재 풀이 값 = ", get_correct_parentheses(')()()()(())('))



##########################
# 삼성 역량 테스트 - 1
# 힌트
# 말은 순서대로 이동합니다 -> 반복문
# 말이 쌓일 수 있습니다. -> 쌓이는 것을 저장
# 쌓인 순서대로 이동 -> 스택

k = 4  # 말의 개수

chess_map = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
]
start_horse_location_and_directions = [
    [0, 0, 0],
    [0, 1, 0],
    [0, 2, 0],
    [2, 2, 2]
]
# 이 경우는 게임이 끝나지 않아 -1 을 반환해야 합니다!
# 동 서 북 남
# →, ←, ↑, ↓
dr = [0, 0, -1, 1]
dc = [1, -1, 0, 0]

# 파란칸 방향 전환
def get_d_index_when_go_back(d):
    if d % 2 == 0:
        return d + 1
    else:
        return d - 1


def get_game_over_turn_count(horse_count, game_map, horse_location_and_directions):
    n = len(game_map)
    turn_count = 1
    current_stacked_horse_map = [[[] for _ in range(n)]for _ in range(n)] #빈 배열 안에 배열, 안에 있는 배열이 반복되어서 겉에 배열에 들어가고 겉에 배열도 반복 
                                                                            # -> 배열이 n개있고 배열이 n개있는 배열인 3차원 배열
    for i in range(horse_count): #현재 쌓여있는 말들의 배열 current_stacked_horse_map에 추가
        r, c, d = horse_location_and_directions[i] #i 번째의 위치와 방향을 r, c, d에 뽑기
        current_stacked_horse_map[r][c].append(i) #위치만 저장

    while turn_count <= 1000:
        for horse_index in range(horse_count):
            n = len(game_map)
            r, c, d = horse_location_and_directions[horse_index] #이동했을 때
            new_r = r + dr[d] #directions의 변화량을 저장
            new_c = c + dc[d]

            # 3) 파란색인 경우에는 A번 말의 이동 방향을 반대로 하고 한칸 이동한다.
            if not 0 <= new_r < n or not 0 <= new_c < n or game_map[new_r][new_c] == 2: #범위를 벗어나거나 파란색 칸인 경우에는
                new_d = get_d_index_when_go_back(d) # 새로운 방향

                horse_location_and_directions[horse_index][2] = new_d #방향을 옮겼으니 새로 업데이트
                new_r = r + dr[new_d] #directions의 변화량을 저장
                new_c = c + dc[new_d]

                #방향을 바꾼 후에 이동하는 칸이 파란색이거나 맵을 나갔을 때
                if not 0 <= new_r < n or not 0 <= new_c < n or game_map[new_r][new_c] == 2:
                    continue

            
            moving_horse_index_array = [] #이동하는 말들을 저장하기 위한 배열
            for i in range(len(current_stacked_horse_map[r][c])): #1번 조건 -> 말을 이동하기 위함
                current_stacked_horse_index = current_stacked_horse_map[r][c][i]
                #여기서 이동해야 하는 애들은 현재 옮기는 말 위의 말들
                if horse_index == current_stacked_horse_index:
                    moving_horse_index_array = current_stacked_horse_map[r][c][i:] #i번째 원소부터 넣어주어야함
                    current_stacked_horse_map[r][c] = current_stacked_horse_map[r][c][:i] #index 앞에 있는 애들만 남기 위해서 i전까지의 배열
                    break

            # 2) 빨간색인 경우에는 이동한 후에 A번 말과 그 위에 모든 말의 순서를 뒤집는다.
            if game_map[new_r][new_c] == 1: #새로운 이동한곳에서 업데이트 하기 때문에
                moving_horse_index_array = reversed(moving_horse_index_array) #뒤집기

            for moving_horse_index in moving_horse_index_array:
                current_stacked_horse_map[new_r][new_c].append(moving_horse_index) #새로운 위치에 업데이트
                #horse_location_and_directions에 이동한 말들의 위치를 업데이트 -> row, culumn
                horse_location_and_directions[moving_horse_index][0],horse_location_and_directions[moving_horse_index][1] = new_r, new_c


            #턴이 진행 되는중 말이 4개 이상 쌓이면 게임 종료
            if len(current_stacked_horse_map[new_r][new_c]) >= 4:
                return turn_count

        turn_count += 1

    return -1


print(get_game_over_turn_count(k, chess_map, start_horse_location_and_directions))  # 2가 반환 되어야합니다

start_horse_location_and_directions = [
    [0, 1, 0],
    [1, 1, 0],
    [0, 2, 0],
    [2, 2, 2]
]
print("정답 = 9 / 현재 풀이 값 = ", get_game_over_turn_count(k, chess_map, start_horse_location_and_directions))

start_horse_location_and_directions = [
    [0, 1, 0],
    [0, 1, 1],
    [0, 1, 0],
    [2, 1, 2]
]
print("정답 = 3 / 현재 풀이 값 = ", get_game_over_turn_count(k, chess_map, start_horse_location_and_directions))



##########################
# 삼성 역량 테스트 - 2
# 구슬 탈출
# 공이 2개 -> 4차원 배열 사용 -> 빨간공 r,c + 파란공 r,c
from collections import deque #queue 이용

# . 은 빈 칸, #은 공이 이동 할 수 없는 장애물 또는 벽, O는 구멍의 위치, B는 파란 구슬
game_map = [
    ["#", "#", "#", "#", "#"],
    ["#", ".", ".", "B", "#"],
    ["#", ".", "#", ".", "#"],
    ["#", "R", "O", ".", "#"],
    ["#", "#", "#", "#", "#"],
]

dr = [-1, 0, 1, 0]
dc = [0, 1, 0, -1]

#이동 방향을 받는 함수
def move_until_wall_or_hole(r, c, diff_r, diff_c, game_map): #game_map을 받는 이유는 벽인지 구멍인지 알기 위해
    move_count = 0 #이동한 칸 수
    #다음 이동이 벽이거나 구멍이 아닐 때까지
    while game_map[r + diff_r][c + diff_c] != '#' and game_map[r][c] != 'O':
        r += diff_r
        c += diff_c
        move_count += 1
    return r, c, move_count


def is_available_to_take_out_only_red_marble(game_map):
    # 구현해보세요!
    n, m = len(game_map), len(game_map[0]) #행과 열의 크기
    visited = [[[[False] * m for _ in range(n)] for _ in range(m)] for _ in range(n)]
    queue = deque()
    red_row, red_col, blue_row, blue_col = -1, -1, -1, -1 #임의 지정
    for i in range(n): #게임 맵 돌기
        for j in range(m):
            if game_map[i][j] == "R": #R 이면 red
                red_row, red_col = i, j
            elif game_map[i][j] == "B": #B이면 blue
                blue_row, blue_col == i, j

    queue.append((red_row, red_col, blue_row, blue_col, 1)) #탐색 횟수가 정해져있으니 현재 탐색하는 숫자 '1' 넣기
    visited[red_row][red_col][blue_row][blue_col] = True #현재 조회했기 때문에 True

    while queue:
        red_row, red_col, blue_row, blue_col, try_count = queue.popleft() #try_count  시도하는 횟수
        if try_count > 10: #10 이하여야 한다.
            break

        for i in range(4): #4방향
            next_red_row, next_red_col, r_count = move_until_wall_or_hole(red_row, red_col, dr[i], dc[i], game_map)
            next_blue_row, next_blue_col, b_count = move_until_wall_or_hole(blue_row, blue_col, dr[i], dc[i], game_map)

            if game_map[next_blue_row][next_blue_col] == 'O': # 파란 구슬이 구멍에 떨어지지 않으면(실패 X)
                continue
            if game_map[next_red_row][next_red_col] == 'O': # 빨간 구슬이 구멍에 떨어진다면(성공)
                return True
            if next_red_row == next_blue_row and next_red_col == next_blue_col: # 빨간 구슬과 파란 구슬이 동시에 같은 칸에 있을 수 없다.
                if r_count > b_count: #이동 거리가 많은 구슬을 한칸 뒤로
                    next_red_row -= dr[i] #움직이기로 했던 거리만큼 떨어트리기
                    next_red_col -= dc[i]
                else:
                    next_blue_row -= dr[i]
                    next_blue_col -= dc[i]

            #BFS 탐색 마치고, 방문 여부 확인
            if not visited[next_red_row][next_red_col][next_blue_row][next_blue_col]:
                visited[next_red_row][next_red_col][next_blue_row][next_blue_col] = True
                queue.append((next_red_row, next_red_col, next_blue_row, next_blue_col, try_count + 1)) #try_count 몇번 시도했는지
    return False


print(is_available_to_take_out_only_red_marble(game_map))  # True 를 반환해야 합니다



game_map = [
    ["#", "#", "#", "#", "#", "#", "#", "#", "#", "#"],
    ["#", ".", "O", ".", ".", ".", ".", "R", "B", "#"],
    ["#", "#", "#", "#", "#", "#", "#", "#", "#", "#"]
]
print("정답 = False / 현재 풀이 값 = ", is_available_to_take_out_only_red_marble(game_map))


game_map = [
["#", "#", "#", "#", "#", "#", "#"],
["#", ".", ".", "R", "#", "B", "#"],
["#", ".", "#", "#", "#", "#", "#"],
["#", ".", ".", ".", ".", ".", "#"],
["#", "#", "#", "#", "#", ".", "#"],
["#", "O", ".", ".", ".", ".", "#"],
["#", "#", "#", "#", "#", "#", "#"]
]
print("정답 = True / 현재 풀이 값 = ", is_available_to_take_out_only_red_marble(game_map))



##########################
# 삼성 역량 테스트 - 3
# 치킨 배달
# 여러 개 중에서 특정 개수를 뽑는 경우의 수
# 모든 경우의 수를 다 구해야 합 ---> 조합 사용해야 함

import itertools, sys
# 조합을 얻으려면 itertools 모듈의 combinations 이용
# 최댓값을 얻으려면 -> 시스템상 최댓값을 min의 초깃값으로 설정 -> sys.maxsize 이용

n = 5
m = 3

city_map = [
    [0, 0, 1, 0, 0],
    [0, 0, 2, 0, 1],
    [0, 1, 2, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 2],
]


def get_min_city_chicken_distance(n, m, city_map):
    chicken_location_list = [] #치킨 위치
    home_location_list = [] #집 위치
    for i in range(n): #n 행과 열의 길이
        for j in range(n): #이중 반복
            if city_map[i][j] == 1: #1 이면 집
                home_location_list.append([i, j])
            if city_map[i][j] == 2: #2 이면 치킨집
                chicken_location_list.append([i, j])
    # 치킨집 중에 M개 고르기(조합)
    chicken_location_m_combinations = list(itertools.combinations(chicken_location_list, m)) #m개를 뽑아서 list에 담아주기
    min_distance_of_m_combinations = sys.maxsize #최소 도시 치킨 거리 구하기
    for chicken_location_m_combination in chicken_location_m_combinations: #치킨 거리 구하기 반복문
        distance = 0 #각 집들의 치킨거리 
        for home_r, home_c in home_location_list: #집 위치를 하나하나 뽑기
            min_home_chicken_distance = sys.maxsize #다시 한번 최솟값을 담기 위해서
            for chicken_location in chicken_location_m_combination: #가장 가까운 치킨집
                min_home_chicken_distance = min( #어느 경우에 치킨 거리가 가장 짧은지
                    min_home_chicken_distance,
                    abs(home_r - chicken_location[0]) + abs(home_c - chicken_location[1])
                )
            distance += min_home_chicken_distance #최소 치킨 거리 += 집에 최소 치킨거리
        min_distance_of_m_combinations = min(min_distance_of_m_combinations, distance) #combinations 돌면서 최솟값을 뽑아내기
    return min_distance_of_m_combinations


# 출력
print(get_min_city_chicken_distance(n, m, city_map))  # 5 가 반환되어야 합니다!


city_map = [
    [1, 2, 0, 0, 0],
    [1, 2, 0, 0, 0],
    [1, 2, 0, 0, 0],
    [1, 2, 0, 0, 0],
    [1, 2, 0, 0, 0]
]
print("정답 = 11 / 현재 풀이 값 = ", get_min_city_chicken_distance(5,1,city_map))


city_map = [
    [0, 2, 0, 1, 0],
    [1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [2, 0, 0, 1, 1],
    [2, 2, 0, 1, 2]
]
print("정답 = 10 / 현재 풀이 값 = ", get_min_city_chicken_distance(5,2,city_map))
print("hihihi")