# 3주차

# 정렬 - 1
# 버블 정렬
# input = [4, 6, 2, 9, 1]


# def bubble_sort(array):
#     # 이 부분을 채워보세요!
#     n = len(array)
#     for i in range(n - 1):         # 4번만 비교하면 되니까!
#             for j in range(n - i - 1): # Q. 여기서 왜 5 - i - 1 일까요? => i의 다음 자리?
#                 if array[j] > array[j + 1]:
#                     array[j], array[j + 1] = array[j + 1], array[j] #Swap a, b = b, a

#     return array


# bubble_sort(input)
# print(input)  # [1, 2, 4, 6, 9] 가 되어야 합니다!

# print("정답 = [1, 2, 4, 6, 9] / 현재 풀이 값 = ",bubble_sort([4, 6, 2, 9, 1]))
# print("정답 = [-1, 3, 9, 17] / 현재 풀이 값 = ",bubble_sort([3,-1,17,9]))
# print("정답 = [-3, 32, 44, 56, 100] / 현재 풀이 값 = ",bubble_sort([100,56,-3,32,44]))


##########################
# 정렬 - 2
# 선택 정렬 => 최솟값을 찾아서 변경
# input = [4, 6, 2, 9, 1]


# def insertion_sort(array):
#     # 이 부분을 채워보세요!
#     n = len(array)
#     for i in range(n - 1):   #맨마지막에 하나 남은 원소를 비교하지 않음
#         min_index = i
#         for j in range(n - i): #점점 줄어들기 때문에 i가 줄어듬
#             if array [i + j] < array[min_index]:
#                 min_index = i + j

#         array[i], array[min_index] = array[min_index], array[i]

#     return array


# insertion_sort(input)
# print(input) # [1, 2, 4, 6, 9] 가 되어야 합니다!


# 삽입 정렬 => 전체에서 하나씩 올바른 위치에 삽입
# input = [4, 6, 2, 9, 1]


# def insertion_sort(array):
#     # 이 부분을 채워보세요!
#     n = len(array)
#     for i in range(1, n):
#         for j in range(i):
#             if array[i - j - 1] > array[i - j]: #앞에있는 애가 뒤에있는 애보다 크다면 변경!
#                 array[i - j - 1], array[i - j] = array[i - j], array[i - j - 1]
#             else:
#                 break #더 작지 않을 경우에는 반복문 실행 X => ex.9

#     return


# insertion_sort(input)
# print(input) # [1, 2, 4, 6, 9] 가 되어야 합니다!

# print("정답 = [4, 5, 7, 7, 8] / 현재 풀이 값 = ",insertion_sort([5,8,4,7,7]))
# print("정답 = [-1, 3, 9, 17] / 현재 풀이 값 = ",insertion_sort([3,-1,17,9]))
# print("정답 = [-3, 32, 44, 56, 100] / 현재 풀이 값 = ",insertion_sort([100,56,-3,32,44]))


##########################
# 정렬 - 3
# 병합 정렬 - merge
# array_a = [1, 2, 3, 5]
# array_b = [4, 6, 7, 8]


# def merge(array1, array2):
#     # 이 부분을 채워보세요!
#     array_c = []
#     array1_index = 0
#     array2_index = 0
#     while array1_index < len(array1) and array2_index < len(array2): #각 array 의 길이까지
#         if array1[array1_index] < array2[array2_index]: #각 index 값을 비교
#             array_c.append(array1[array1_index])    #c에다가 append
#             array1_index += 1 #다음것으로
#         else:
#             array_c.append(array2[array2_index])
#             array2_index += 1

#     if array1_index == len(array1): #array1이 끝까지 갔다 == array2가 남아있다
#         while array2_index < len(array2):
#             array_c.append(array2[array2_index])
#             array2_index += 1

#     if array2_index == len(array2):
#         while array1_index < len(array1):
#             array_c.append(array1[array1_index])
#             array1_index += 1
#     return array_c


# print(merge(array_a, array_b))  # [1, 2, 3, 4, 5, 6, 7, 8] 가 되어야 합니다!


# 병합 정렬 - mergeSort =>: 할 정복의 개념
# MergeSort(시작점, 끝점)

# 그러면
# MergeSort(0, N) = Merge(MergeSort(0, N/2) + MergeSort(N/2, N))

# array = [5, 3, 2, 1, 6, 8, 7, 4]


# def merge_sort(array):
#     # 이 곳을 채워보세요!
#     if len(array) <= 1:
#         return array
#     mid = len(array) // 2
#     left_array = merge_sort(array[:mid])   # 왼쪽 부분을 정렬하고 => :mid 0부터 mid 까지
#     right_array = merge_sort(array[mid:])  # 오른쪽 부분을 정렬한 다음에
#     merge(left_array, right_array)         # 합치면서 정렬
#     return merge(merge_sort(left_array), merge_sort(right_array))
#     #return array


# def merge(array1, array2):
#     result = []
#     array1_index = 0
#     array2_index = 0
#     while array1_index < len(array1) and array2_index < len(array2):
#         if array1[array1_index] < array2[array2_index]:
#             result.append(array1[array1_index])
#             array1_index += 1
#         else:
#             result.append(array2[array2_index])
#             array2_index += 1

#     if array1_index == len(array1):
#         while array2_index < len(array2):
#             result.append(array2[array2_index])
#             array2_index += 1

#     if array2_index == len(array2):
#         while array1_index < len(array1):
#             result.append(array1[array1_index])
#             array1_index += 1

#     return result


# print(merge_sort(array))  # [1, 2, 3, 4, 5, 6, 7, 8] 가 되어야 합니다!


##########################
# 스택 => 한쪽 끝으로만 자료를 넣고 뺄 수 있는 자료 구조. ex) 빨래 바구니
# 스택의 구현
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class Stack:
    def __init__(self):
        self.head = None

    def push(self, value): # 현재 [4] 밖에 없다면 #맨 앞에 데이터 넣기
        # 어떻게 하면 될까요?
        new_head = Node(value)             # [3] 을 만들고!
        new_head.next = self.head          # [3] -> [4] 로 만든다음에
        self.head = new_head               # 현재 head의 값을 [3] 으로 바꿔준다.
        return

    # pop 기능 구현
    def pop(self): #맨 앞의 데이터 뽑기
        # 어떻게 하면 될까요?
        if self.is_empty():                  # 만약 비어있다면 에러!
            return "Stack is empty!"
        delete_head = self.head
        self.head = self.head.next
        return delete_head

    def peek(self): #맨 앞의 데이터 보기
        # 어떻게 하면 될까요?
        if self.is_empty():                  # 만약 비어있다면 에러!
            return "Stack is empty!"
        return self.head.data
        return

    # isEmpty 기능 구현
    def is_empty(self): #스택이 비었는지 안 비었는지 여부 반환해주기
        # 어떻게 하면 될까요?
        return self.head is None

# ex)
stack = Stack()
stack.push(3)
print(stack.peek())
stack.push(4)
print(stack.peek())
print(stack.pop().data)
print(stack.peek())
print(stack.is_empty())


# 문제 풀기 - 탑 (어렵네;)
top_heights = [6, 9, 5, 7, 4]


def get_receiver_top_orders(heights):
    answer = [0] * len(heights)
    while heights:  # heights가 빈상태가 아닐때까지
        height = heights.pop()  # hegihts에서 맨앞
        # heights의 현재 길이 첫 -1 => index번호이기 때문에, 마지막 -1은 0꺼자 -1씩 줄이기
        for idx in range(len(heights) - 1, 0, -1):
            if heights[idx] > height:
                answer[len(heights)] = idx + 1  # +1의 이유는 index가 아닌 위치를 알려주기 위함
                break
    return answer


print(get_receiver_top_orders(top_heights))  # [0, 0, 2, 2, 4] 가 반환되어야 한다!

print("정답 = [0, 0, 2, 2, 4] / 현재 풀이 값 = ",get_receiver_top_orders([6, 9, 5, 7, 4]))
print("정답 = [0, 0, 0, 3, 3, 3, 6] / 현재 풀이 값 = ",get_receiver_top_orders([3, 9, 9, 3, 5, 7, 2]))
print("정답 = [0, 0, 2, 0, 0, 5, 6] / 현재 풀이 값 = ",get_receiver_top_orders([1, 5, 3, 6, 7, 6, 5]))


##########################
# 큐 => 한쪽 끝으로 자료를 넣고, 반대쪽에서는 자료를 뺄 수 있는 선형구조. ex) 놀이기구줄
# ===> 순서대로 처리되어야 하는 일에 필요

class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class Queue:
    def __init__(self):
        self.head = None
        self.tail = None

    def enqueue(self, value):  # 맨 뒤에 데이터 추가하기
        # 어떻게 하면 될까요?
        new_node = Node(value)
        if self.is_empty():                # 만약 비어있다면,
            self.head = new_node           # head 에 new_node를
            self.tail = new_node           # tail 에 new_node를 넣어준다.
            return

        self.tail.next = new_node
        self.tail = new_node
        return

    def dequeue(self):  # 맨 앞의 데이터 뽑기
        # 어떻게 하면 될까요?
        if self.is_empty():
            return "Queue is empty!"

        delete_head = self.head
        self.head = self.head.next

        return delete_head

    def peek(self): #맨 앞의 데이터 보기
        # 어떻게 하면 될까요?
        if self.is_empty():
            return "Queue is empty!"
    
        return self.head.data

    def is_empty(self): #큐가 비었는지 안 비었는지 여부 반환해주기
        # 어떻게 하면 될까요?
        return self.head is None

#ex)
queue = Queue()
queue.enqueue(3) #맨 뒤에 3
print(queue.peek()) #맨 앞을 보면 3뿐이니 3
queue.enqueue(4) #맨 뒤에 4
print(queue.peek()) #맨 앞 3
queue.enqueue(5) #맨 뒤에 5
print(queue.peek()) #맨 앞 3
print(queue.dequeue()) # 맨 앞 데이터 뽑기
print(queue.peek()) #맨 앞 3 뽑혀서 4



##########################
# 해쉬 - 1
# 딕셔너리의 구현
class Dict:
    def __init__(self):
        self.items = [None] * 8

    def put(self, key, value):
        # 구현해보세요!
        index = hash(key) % len(self.items) #현재 배열의 최대 길이로 hash => 인덱스의 범위에 맞춘 뒤 그 값을 담는다
        self.items[index] = value
        return

    def get(self, key):
        # 구현해보세요!
        index = hash(key) % len(self.items)
        return self.items[index]


my_dict = Dict()
my_dict.put("test", 3)
print(my_dict.get("test"))  # 3이 반환되어야 합니다!

# =====> 충돌 링크드 리스트로 딕셔너리로 해결 (추가되는 녀석 뒤에 붙이기)
class LinkedTuple:
    def __init__(self):
        self.items = list()

    def add(self, key, value):
        self.items.append((key, value))

    def get(self, key):
        for k, v in self.items:
            if key == k:
                return v

class LinkedDict:
    def __init__(self):
        self.items = [] #[LinkedTuple(), LinkedTuple(), LinkedTuple()...]
        for i in range(8):
            self.items.append(LinkedTuple())

    def put(self, key, value):
        index = hash(key) % len(self.items)
        self.items[index].add(key, value)

    def get(self, key):
        index = hash(key) % len(self.items)
        return self.items[index].get(key)



##########################
# 해쉬 - 2
# 해쉬 테이블(Hash Table) => "키" 와 "데이터"를 저장함으로써 즉각적으로 데이터를 받아오고 업데이트하고 싶을 때 사용하는 자료구조
# ========================> 딕셔너리와 동일

#해쉬 함수(Hash Function) =>  임의의 길이를 갖는 메시지를 입력하여 고정된 길이의 해쉬값을 출력하는 함수

#만약, 해쉬 값 혹은 인덱스가 중복되어 충돌이 일어난다면? 체이닝과 개방 주소법 방법으로 해결

### 출석체크 문제
# all_students = ["나연", "정연", "모모", "사나", "지효", "미나", "다현", "채영", "쯔위"]
# present_students = ["정연", "모모", "채영", "쯔위", "사나", "나연", "미나", "다현"]
# #정열이 되어있지 않으니 반복문 돌려보기


def get_absent_student(all_array, present_array):
    # 구현해보세요!
    dict = {}
    for key in all_array:
        dict[key] = True #아무 값이나 넣기

    for key in present_array: #dict에서 key를 하나씩 없애기
        del dict[key]
    
    for key in dict.keys(): #key 중에 하나를 반환
        return key


print(get_absent_student(all_students, present_students))

print("정답 = 예지 / 현재 풀이 값 = ",get_absent_student(["류진","예지","채령","리아","유나"],["리아","류진","채령","유나"]))
print("정답 = RM / 현재 풀이 값 = ",get_absent_student(["정국","진","뷔","슈가","지민","RM"],["뷔","정국","지민","진","슈가"]))



##########################
#숙제

#쓱 최대로 할인 적용하기
shop_prices = [30000, 2000, 1500000]
user_coupons = [20, 40]


def get_max_discounted_price(prices, coupons):
    # 이 곳을 채워보세요!
    prices.sort(reverse=True)
    coupons.sort(reverse=True)
    price_index = 0
    coupon_index = 0
    max_discount_price = 0

    while price_index < len(prices) and coupon_index < len(coupons):
        max_discount_price += prices[price_index] * (100 - coupons[coupon_index]) / 100
        price_index += 1
        coupon_index += 1

    while price_index < len(prices):
        max_discount_price += prices[price_index]
        price_index += 1

    return max_discount_price


print("정답 = 926000 / 현재 풀이 값 = ", get_max_discounted_price([30000, 2000, 1500000], [20, 40]))
print("정답 = 485000 / 현재 풀이 값 = ", get_max_discounted_price([50000, 1500000], [10, 70, 30, 20]))
print("정답 = 1550000 / 현재 풀이 값 = ", get_max_discounted_price([50000, 1500000], []))
print("정답 = 1458000 / 현재 풀이 값 = ", get_max_discounted_price([20000, 100000, 1500000], [10, 10, 10]))


#올바른 괄호
def is_correct_parenthesis(string):
    # 구현해보세요!
    stack = []

    for i in range(len(string)):
        if string[i] == '(':
            stack.append(i)
        elif string[i] == ")":
            if len(stack) == 0:
                return False
            stack.pop()

    if len(stack) != 0:
        return False
    else:
        return True


print("정답 = True / 현재 풀이 값 = ", is_correct_parenthesis("(())"))
print("정답 = False / 현재 풀이 값 = ", is_correct_parenthesis(")"))
print("정답 = False / 현재 풀이 값 = ", is_correct_parenthesis("((())))"))
print("정답 = False / 현재 풀이 값 = ", is_correct_parenthesis("())()"))
print("정답 = False / 현재 풀이 값 = ", is_correct_parenthesis("((())"))


# #멜론 베스트 앨범 뽑기 (제일 어려움;)
def get_melon_best_album(genre_array, play_array):
    n = len(genre_array)
    genre_total_play_dict = {}
    genre_index_play_array_dict = {}
    for i in range(n):
        genre = genre_array[i]
        play = play_array[i]
        if genre not in genre_total_play_dict:
            genre_total_play_dict[genre] = play
            genre_index_play_array_dict[genre] = [[i, play]]
        else:
            genre_total_play_dict[genre] += play
            genre_index_play_array_dict[genre].append([i, play])

    sorted_genre_play_array = sorted(genre_total_play_dict.items(), key=lambda item: item[1], reverse=True)
    result = []
    for genre, _value in sorted_genre_play_array:
        index_play_array = genre_index_play_array_dict[genre]
        sorted_by_play_and_index_play_index_array = sorted(index_play_array, key=lambda item: item[1], reverse=True)
        for i in range(len(sorted_by_play_and_index_play_index_array)):
            if i > 1:
                break
            result.append(sorted_by_play_and_index_play_index_array[i][0])
    return result


print("정답 = [4, 1, 3, 0] / 현재 풀이 값 = ",   get_melon_best_album(["classic", "pop", "classic", "classic", "pop"], [500, 600, 150, 800, 2500]))
print("정답 = [0, 6, 5, 2, 4, 1] / 현재 풀이 값 = ", get_melon_best_album(["hiphop", "classic", "pop", "classic", "classic", "pop", "hiphop"], [2000, 500, 600, 150, 800, 2500, 2000]))