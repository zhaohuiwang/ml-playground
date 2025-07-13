
def mask_ccard(card_number: int) -> str:
    """This funciton takes in credit card numbers as an argument, output a string showing the last fout digit and the rest are replaced with *.
    card_numger: str
    
      Parameters
      ----------
      card_numger: str
        The input card numbe.
      Returns
      -------
      str
        The masked card number.
    """
    return "*"*(len(card_number)-4)+card_number[-4:]

import math
def is_pronic(n: int) -> bool:
    """A pronic number is a number that is the product of two consective numbers"""
    sqrt_nf = math.floor(math.sqrt(n))
    return sqrt_nf*(sqrt_nf+1) == n

def is_disarium(n: int) -> bool:
    """
    A number is considered Disarium when the sum of its digits, each raised to the power of their respective postions, equals the number itself.
    FOr example, 175 = 1**1 + 7**2 + 5**3
    """
    return sum([int(str(n)[i-1])**i for i in range(1,len(str(n))+1)]) == n

def sort_string_to_ginorts(s: str) -> str:
    """
    In ginortS, all sorted lowercase letters are ahead of uppercase letters, all sorted uppercase letters are ahead of digit, and all sorted odd numbers are ahead of even numbers.
    """
    c_upper, c_lower, d_odd, d_even = '', '', '', ''
    for i in sorted(s):
        # resulting in [*digits, *upper_characters, *lower_characters]
        if i.isalpha():
            if i.isupper():
                c_upper += i
            else:
                c_lower += i
        elif i.isdigit():
            if int(i)%2 != 0:
                d_odd += i
            else:
                d_even += i
        else:
            continue
    return  c_lower + c_upper + d_odd + d_even

def move_zero_to_end(n: int) -> int:
    """ """
    new_str, c_zeros = '', 0
    for i in str(n):
        if i == '0':
            c_zeros += 1
        else:
            new_str += i
    return int(new_str + '0'*c_zeros)



from typing import List

def _get_change_making_matrix(set_of_coins: List[int], r: int) -> List[List]:
    """
    An matrix with orderd coin values as row indices and ordered integer number from 0 to r as the column indices 
    """
    m = [[0 for _ in range(r + 1)] for _ in range(len(set_of_coins) + 1)]
    for i in range(1, r + 1):
        m[0][i] = float("inf")  # By default there is no way of making change
    return m

def change_making(coins: List[int], n: int) -> int:
    """This function assumes that all coins are available infinitely.
    if coins are only to be used once, change m[c][r - coin] to m[c - 1][r - coin].
    n is the number to obtain with the fewest coins.
    coins is a list or tuple with the available denominations.
    For example, coins = [1,2,5] n = 6:
        0   1   2   3   4   5   6   
    0   0   inf inf inf inf inf inf
    1   0   1   2   3   4   5   6
    2   0   1   1   2   2   3   3
    5   0   1   1   2   2   1   2
    """
    m = _get_change_making_matrix(coins, n)
    for c, coin in enumerate(coins, 1):
        for r in range(1, n + 1):
            # Just use the coin
            if coin == r:
                m[c][r] = 1
            # coin cannot be included. Use the previous solution for making r,excluding coin. When the coin value (row index) is bigger than that of the amount (column index), we can not use the current coin instead use the set of coins with smaller face values. 
            elif coin > r:
                m[c][r] = m[c - 1][r]
            # coin can be used. Decide which one of the following solutions is the best: 1. Not using the current coin, the previous solution for making r with smaller value coin sets. 2. Using the current coin (count is 1) and other set of coins required to cover the remaining amount (r-coin, which is the column index).
            else:
                m[c][r] = min(m[c - 1][r], 1 + m[c][r - coin])
    return m[-1][-1]
    # return m

cnt = change_making(coins=[1,2,5], n=6) 


from typing import List

def coin_change_options(coins: List[int], amount: int) -> int:
    """
    Given a set of coin denominations and a total amount, determine the number of ways to make that amount using the available coins.
    """
    # Create a Dynamic Programming Table to store the number of ways to make change for each value up to the amount
    dp = [0] * (amount + 1)
    # dp = [0 for _ in range(amount + 1)]
    
    # There is one way to make the amount 0, by choosing no coins
    dp[0] = 1
    
    # For each coin, update the dp table
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]

    return dp[amount]


def coin_change_minimum_count(coins: List[int], amount: int) -> int:
    """
    You are given an array of coins with varying denominations and an integer sum representing the total amount of money; you must return the fewest coins required to make up that sum; if that sum cannot be constructed, return -1.
    """
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    # [0, inf, inf, inf,..., inf]

    for coin in coins:
        for j in range(coin, amount + 1):
            dp[j] = min(dp[j], dp[j - coin] + 1)
            # dp[j - coin] + 1 specified the current coin count as 1 plus the the number of coin required to cover the remaining amount (index is j-coin) 

    if dp[amount] == float('inf'):
        return -1
    else:
        return dp[amount]
    

# Calculate the nth Fibonacci number using recursion.
from typing import List, Optional

def fibonacci_r(n: int) -> int:
    """    
    Solving Fibonacci Numbers using recursion only
    n (int): The position in the Fibonacci sequence to calculate.
    Fibonacci sequence is a sequence in which each element is the sum of the two elements that precede it. It usually Starts from 0 and 1, althought alternative versions starts from 1 and 1 or 1 and 3 are also seen.
    The bigger the n, the closer the ratio of fibonacci(n-1)/fibonacci(n) is to the Golden Ratio.
    
    Returns:
    int: The nth Fibonacci number.
    """
    # Base case: return n if it is 0 or 1
    if n <= 1:
        result =  n
    else:
        # Recursive case: sum of the two preceding numbers
        result = fibonacci(n - 1) + fibonacci(n - 2)

    return result


memo = {}

def fibonacci_rm(n: int) -> int:
    """    
    Solving Fibonacci Numbers using recursion and memoization
    n (int): The position in the Fibonacci sequence to calculate.
    Fibonacci sequence is a sequence in which each element is the sum of the two elements that precede it. It usually Starts from 0 and 1, althought alternative versions starts from 1 and 1 or 1 and 3 are also seen.
    The bigger the n, the closer the ratio of fibonacci(n-1)/fibonacci(n) is to the Golden Ratio.
    
    Returns:
    int: The nth Fibonacci number.
    """
    if n in memo:
        return memo[n]
    
    # Base case: return n if it is 0 or 1
    if n <= 1:
        result =  n
    else:
        # Recursive case: sum of the two preceding numbers
        result = fibonacci(n - 1) + fibonacci(n - 2)
    memo[n] = result

    return result



def fibonacci_dp(n : int) -> int:
    """
    Solving Fibonacci Numbers using Dynamic Programming
    Dynamic programming is a method for solving a complex problem by breaking it up into smaller subproblems, and store the results of the subproblems for later use (to reduce duplication).
    """
    memo = {}
    for i in range(0, n+1):
        if i < 2:
            result = i
        else:
            result = memo[i-1] + memo[i-2]
        
        memo[i] = result
    
    return memo[n]


# Example usage
if __name__ == "__main__":
    # Change this value to compute a different Fibonacci number
    position = 10
    result = fibonacci(position)
    print(f"The {position}th Fibonacci number is: {result}")


"""
def factorial(n: int) -> int:
    if n <= 1:
        return 1
    return n * factorial(n - 1)
    
"""
from typing import List
def binary_search(arr: List[int], target: int, low: int, high: int) -> int:
    """
    A function searchs for the target in the sorted array, and returns the founded target index.
    Recursive Binary Search: the algorithm narrows down the search space by repeatedly dividing it in half until the target is found or the search space is empty.
    """
    # Base case: target not in array
    if low > high:
        return -1
    # Recursive case
    else:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            return binary_search(arr, target, mid + 1, high)
        else:
            return binary_search(arr, target, low, mid - 1)
        
arr = [5, 3, 8, 4, 2, 7]
sorted_arr = arr.sort()
target = 4
low= 0
high = len(arr) -1
binary_search(sorted(arr), target, low, high)

from typing import Any
def linear_search(list1: List[Any], target: Any) -> int:
        """ Linear search algorithm """
        for index, value in enumerate(list1):
            if value == target:
                return index
        return -1  # Indicate that the target is not found





def bfs(visited, graph, node): 
  """
  Function for Breadth-first Search or BFS
  """
  visited.append(node)
  queue.append(node)

  while queue:          # Creating loop to visit each node
    m = queue.pop(0) 
    print (m, end = " ") 

    for neighbour in graph[m]:
      if neighbour not in visited:
        visited.append(neighbour)
        queue.append(neighbour)


# Define a tree data structure
graph = {
    # 5 is the value in the vertix/node, 3 and 7 are its neighbours  
    '5' : ['3','7'], 
    '3' : ['2', '4'],
    '7' : ['8'],
    '2' : [], # a leaf node, the end of branch with no children
    '4' : ['8'],
    '8' : []
}

visited = []    # List for visited nodes.
queue = []      # Initialize a queue
# Driver Code
print("Following is the Breadth-First Search")
bfs(visited, graph, '5')    # function calling

