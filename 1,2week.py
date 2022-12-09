# 알파벳 최빈값 찾기 방법 1

input = "hello my name is sparta"


def find_max_occurred_alphabet(string):
    # 이 부분을 채워보세요!
    alphabet_array = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "x", "y", "z"]
    max_occurrence = 0
    max_alphabet = alphabet_array[0]

    for alphabet in alphabet_array :
        occurrence = 0
        for char in string:
            if char == alphabet:
                occurrence += 1

        if occurrence > max_occurrence:
            max_alphabet = alphabet
            max_occurrence = occurrence
    return max_alphabet


result = find_max_occurred_alphabet(input)
print(result)

#############################################

# 알파벳 최빈값 찾기 방법 2
input = "hello my name is sparta"


def find_max_occurred_alphabet(string):
    # 이 부분을 채워보세요!

    alphabet_occurrence_array = [0] * 26

    # 이 부분을 채워보세요!

    for char in string: #012345678910~
        if not char.isalpha():
            continue
        arr_index = ord(char) - ord("a") #배열에 인덱스 값?
        alphabet_occurrence_array[arr_index] += 1

    max_occurrence = 0
    max_alphabet_index = 0
    for index in range(len(alphabet_occurrence_array)):
        alphabet_occurrence = alphabet_occurrence_array[index]

        if alphabet_occurrence > max_occurrence:
            max_alphabet_index = index
            max_occurrence = alphabet_occurrence

    return chr(max_alphabet_index + ord("a"))
    #아스킷 코드로 변환했기 때문에 다시 돌려놓기


result = find_max_occurred_alphabet


#############################################


def find_alphabet_occurrence_array(string):
    alphabet_occurrence_array = [0] * 26

    for char in string :
        if not char.isalpha() : #알파벳이 아니면
            continue #반복문 계속 출력
        arr_index = ord(char) - ord("a") #알파벳 아스키 코드 a = 97, b = 98
        alphabet_occurrence_array[arr_index] += 1
    return alphabet_occurrence_array


print(find_alphabet_occurrence_array("hello my name is sparta"))

#############################################

def find_alphabet_occurrence_array(string):
    alphabet_occurrence_array = [0] * 26

    # 이 부분을 채워보세요!

    for char in string: #012345678910~
        if not char.isalpha():
            continue
        arr_index = ord(char) - ord("a") #배열에 인덱스 값?
        alphabet_occurrence_array[arr_index] += 1

    return alphabet_occurrence_array


print(find_alphabet_occurrence_array("hello my name is sparta"))

#############################################


input = [3, 5, 6, 1, 2, 4]


def is_number_exist(number, array):
    # 이 부분을 채워보세요!
    for num in array:
        if num == number:
            return True
    return False


result = is_number_exist(3, input)
print(result)

#############################################

input = [0, 3, 5, 6, 1, 2, 4]


def find_max_plus_or_multiply(array):
    # 이 부분을 채워보세요!
    multiply_num = 0
    for num in array:
        if num <= 1 or multiply_num <= 1:
            multiply_num += num
        else:
            multiply_num *= num
    return multiply_num


result = find_max_plus_or_multiply(input)
print(result)

#############################################
input = "abadabac"

def find_not_repeating_character(string):
    # 이 부분을 채워보세요!
    alphabet_occurrence_array = [0] * 26

    for char in string:
        if not char.isalpha():
            continue
        arr_index = ord(char) - ord("a")
        alphabet_occurrence_array[arr_index] += 1

    not_occurrence = []
    for index in range(len(alphabet_occurrence_array)):
        alphabet_occurrence = alphabet_occurrence_array[index]

        if alphabet_occurrence == 1:
            not_occurrence.append(chr(index + ord("a")))

    for char in string:
        if char in not_occurrence:
            return char

    return "_"


result = find_not_repeating_character(input)
print(result)

#############################################

# 소수 찾기
input = 20


def find_prime_list_under_number(number):
    # 이 부분을 채워보세요!
    prime_array = []

    for n in number:
        if n % 2 == 0:
            break
        else:
            prime_array.append(n)

    return [prime_array]


result = find_prime_list_under_number(input)
print(result)

#############################################

def summarize_string(target_string):
    # 이 부분을 채워보세요!
    n = len(target_string) #8
    count = 0
    result_str = ''

    for i in range(n - 1): #7 => 0,1,2,3,4,5,6
        if target_string[i] == target_string[i + 1]:
            count += 1
        else:
            result_str += target_string[i] + str(count + 1) + '/'
            count = 0

    result_str += target_string[n - 1] + str(count + 1)

    return result_str


input_str = "acccdeeed"

print(summarize_string(input_str))


#############################################
# 2주차

# 클래스
class Person:
    def __init__(self, param_name):
        print("i am created!", self)
        self.name = param_name

    def talk(self):
        print("안녕하세요, 제 이름은", self.name, "입니다")

person_1 = Person("유재석") #() => 생성자, init이랑 같음
print(person_1.name)
print(person_1)
person_1.talk()
person_2 = Person("박명수")
print(person_2.name)
print(person_2)
person_2.talk()


#############################################

# 링크드 리스트 - 1
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self, value):
        self.head = Node(value)

    def append(self, value):
        cur = self.head
        while cur.next is not None:
            cur = cur.next
        cur.next = Node(value)

    def print_all(self):
        cur = self.head
        while cur is not None:
            print(cur.data)
            cur = cur.next

linked_list = LinkedList(5)
linked_list.append(12)
linked_list.print_all()


#############################################

# 링크드 리스트 - 2
# 원소 찾기 / 원소 추가
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    def __init__(self, value):
        self.head = Node(value)

    def append(self, value):
        cur = self.head
        while cur.next is not None:
            cur = cur.next
        cur.next = Node(value)

    def print_all(self):
        cur = self.head
        while cur is not None:
            print(cur.data)
            cur = cur.next

    def get_node(self, index): #원소 찾기
        node = self.head
        count = 0
        while count < index:
            node = node.next
            count += 1
        return node

    def add_node(self, index, value): #원소 추가
        new_node = Node(value)
        if index == 0: #index - 1 에서 0 일때의 조건
            new_node.next = self.head
            self.head = new_node
            return

        node = self.get_node(index - 1)
        next_node = node.next
        node.next = new_node
        new_node.next = next_node

    def delete_node(self, index): #원소 삭제
        if index == 0: #index - 1 에서 0 일때의 조건
            self.head = self.head.next
            return

        node = self.get_node(index - 1)
        node.next = node.next.next
        return "index 번째 Node를 제거해주세요!"


linked_list = LinkedList(5)
linked_list.append(12)
linked_list.add_node(0, 3)
linked_list.delete_node(0)
linked_list.print_all()


#############################################

# 링크드 리스트 문제
# 두 링크드 리스트의 합 계산
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    def __init__(self, value):
        self.head = Node(value)

    def append(self, value):
        cur = self.head
        while cur.next is not None:
            cur = cur.next
        cur.next = Node(value)


# def get_linked_list_sum(linked_list_1, linked_list_2):
#     # 구현해보세요!
#     sum_1 = 0
#     head_1 = linked_list_1.head
#     while head_1 is not None:
#         sum_1 = sum_1 * 10 + head_1.data
#         head_1 = head_1.next

#     sum_2 = 0
#     head_2 = linked_list_2.head
#     while head_2 is not None:
#         sum_2 = sum_2 * 10 + head_2.data
#         head_2 = head_2.next

#     return sum_1 + sum_2

def get_linked_list_sum(linked_list_1, linked_list_2):
    sum_1 = _get_linked_list_sum(linked_list_1)
    sum_2 = _get_linked_list_sum(linked_list_2)

    return sum_1 + sum_2


def _get_linked_list_sum(linked_list):
    sum = 0
    head = linked_list.head
    while head is not None:
        sum = sum * 10 + head.data
        head = head.next
    return sum


linked_list_1 = LinkedList(6)
linked_list_1.append(7)
linked_list_1.append(8)

linked_list_2 = LinkedList(3)
linked_list_2.append(5)
linked_list_2.append(4)

print(get_linked_list_sum(linked_list_1, linked_list_2))


#############################################

# 이진 탐색
finding_target = 14
finding_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


def is_existing_target_number_binary(target, array):
    # 구현해보세요!
    cur_min = 0 #최솟값
    cur_max = len(array) - 1 #최댓값
    cur_guess = (cur_min + cur_max) // 2 #8

    while cur_min <= cur_max:
        if array[cur_guess] == target:
            return True
        elif array[cur_guess] < target:
            cur_min = cur_guess + 1
        else:
            cur_max = cur_guess - 1
        cur_guess = (cur_min + cur_max) // 2
    return False


result = is_existing_target_number_binary(finding_target, finding_numbers)
print(result)


#############################################

# 이진 탐색 - 무작위 수 찾기 => 불가능!!(일정한 규칙으로 정렬되어 있는 데이터일때만 이진 탐색이 가능)
finding_target = 2
finding_numbers = [0, 3, 5, 6, 1, 2, 4]

def is_exist_target_number_binary(target, numbers):
    # 이 부분을 채워보세요!
    cur_min = 0 #최솟값
    cur_max = len(array) - 1 #최댓값
    cur_guess = (cur_min + cur_max) // 2 #8

    while cur_min <= cur_max:
        if array[cur_guess] == target:
            return True
        elif array[cur_guess] < target:
            cur_min = cur_guess + 1
        else:
            cur_max = cur_guess - 1
        cur_guess = (cur_min + cur_max) // 2
    return False
    return 1


result = is_exist_target_number_binary(finding_target, finding_numbers)
print(result)


#############################################

# 재귀(Recursion)은 어떠한 것을 정의할 때 자기 자신을 참조하는 것을 뜻한다. [위키백과]
# => 자기 자신을 호출하는 함수, 특정 구조가 반복된다면 재귀함수로 문제를 풀어보도록 하기! ☆반드시 탈출조건 필요☆
# 재귀 함수 - 보신각 카운트다운
def count_down(number):
    if number < 0:         # 만약 숫자가 0보다 작다면, 빠져나가자!
        return

    print(number)          # number를 출력하고
    count_down(number - 1) # count_down 함수를 number - 1 인자를 주고 다시 호출한다!


count_down(60)


#############################################

# 재귀 함수 -2 팩토리얼
# 팩토리얼은 1부터 어떤 양의 정수 n까지의 정수를 모두 곱한 것을 의미
def factorial(n):
    # 이 부분을 채워보세요!
    if n == 1:
        return 1
    return n * factorial(n - 1)


print(factorial(5))


#############################################

# 재귀 함수 -2 회문 검사
input = "abcba"


def is_palindrome(string):
    n = len(string)
    for i in range(n):
        if string[i] != string [n - 1 - i]:
            return false
    return True


print(is_palindrome(input))

# 위에 문제 재귀 함수로 풀기
input = "abcba"


def is_palindrome(string):
    if len(string) <= 1: #if문은 탈출, 조건 1보다 작을때
        return True
    if string[0] != string[-1]: #문자열 맨 앞과 맨 뒤가 다를 때 False
        return False
    return is_palindrome(string[1:-1])


print(is_palindrome(input))


#############################################
# 2주차 숙제
# 링크드 리스트 끝에서 K번째 값 출력하기
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    def __init__(self, value):
        self.head = Node(value)

    def append(self, value):
        cur = self.head
        while cur.next is not None:
            cur = cur.next
        cur.next = Node(value)

    def get_kth_node_from_last(self, k):
        # 구현해보세요!
        length = 1
        cur = self.head
        while cur.next is not None:
            cur = cur.next
            length += 1

        end_length = length - k
        cur = self.head
        for i in range(end_length):
            cur = cur.next
        return cur

linked_list = LinkedList(6)
linked_list.append(7)
linked_list.append(8)

print(linked_list.get_kth_node_from_last(2).data)  # 7이 나와야 합니다!


# 배달의 민족 배달 가능 여부
shop_menus = ["만두", "떡볶이", "오뎅", "사이다", "콜라"]
shop_orders = ["오뎅", "콜라", "만두"]


def is_available_to_order(menus, orders):
    # 이 부분을 채워보세요!
    menus_set = set(menus)
    for order in orders:
        if order not in menus_set:
            return False
    return True


# result = is_available_to_order(shop_menus, shop_orders)
# print(result)

# 더하거나 빼거나
numbers = [2, 3, 1]
target_number = 0
result_count = 0  # target 을 달성할 수 있는 모든 방법의 수를 담기 위한 변수


def get_count_of_ways_to_target_by_doing_plus_or_minus(array, target, current_index, current_sum):
    if current_index == len(array):  # 탈출조건!
        if current_sum == target:
            global result_count
            result_count += 1  # 마지막 다다랐을 때 합계를 추가해주면 됩니다.
        return
    get_count_of_ways_to_target_by_doing_plus_or_minus(array, target, current_index + 1, current_sum + array[current_index])
    get_count_of_ways_to_target_by_doing_plus_or_minus(array, target, current_index + 1, current_sum - array[current_index])


get_count_of_ways_to_target_by_doing_plus_or_minus(numbers, target_number, 0, 0)
# current_index 와 current_sum 에 0, 0을 넣은 이유는 시작하는 총액이 0, 시작 인덱스도 0이니까 그렇습니다!
print(result_count)  # 2가 반환됩니다!