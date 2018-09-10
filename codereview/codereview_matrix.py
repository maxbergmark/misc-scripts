import sys
import time

R = 100
C = 100

def minCost2(cost, m, n):
 
    # Instead of following line, we can use int tc[m+1][n+1] or
    # dynamically allocate memoery to save space. The following
    # line is used to keep te program simple and make it working
    # on all compilers.
    tc = [[0 for x in range(C)] for x in range(R)]
 
    tc[0][0] = cost[0][0]
 
    # Initialize first column of total cost(tc) array
    for i in range(1, m+1):
        tc[i][0] = tc[i-1][0] + cost[i][0]
 
    # Initialize first row of tc array
    for j in range(1, n+1):
        tc[0][j] = tc[0][j-1] + cost[0][j]
 
    # Construct rest of the tc array
    for i in range(1, m+1):
        for j in range(1, n+1):
            tc[i][j] = min(tc[i-1][j-1], tc[i-1][j], tc[i][j-1]) + cost[i][j]
 
    return tc[m][n]

# Returns cost of minimum cost path from (0,0) to (m, n) in mat[R][C]
def minCost(cost, m, n):
    if (n < 0 or m < 0):
        return sys.maxsize
    elif (m == 0 and n == 0):
        return cost[m][n]
    else:
        return cost[m][n] + min( minCost(cost, m-1, n-1),
                                minCost(cost, m-1, n),
                                minCost(cost, m, n-1) )
 
#A utility function that returns minimum of 3 integers */
def min(x, y, z):
    if (x < y):
        return x if (x < z) else z
    else:
        return y if (y < z) else z

def test(target_matrix, cost, i, j, m, n):
    if (i == m and j == n):
        return cost
    if i+1 > m:
        cost += target_matrix[i][j+1] 
        return test(target_matrix, cost, i, j+1, m, n)
    if j+1 > n:
        cost += target_matrix[i+1][j] 
        return test(target_matrix, cost, i+1, j, m, n)
    if (i+1 <= m and j+1 <= n):
        ret_cost, i, j = min2(target_matrix[i+1][j], target_matrix[i][j+1], target_matrix[i+1][j+1], i, j)
        cost +=ret_cost
        return test(target_matrix, cost, i, j, m, n)


def min2(x, y, z, i, j):
    if (x < y):
        if (x < z):
            return x, i+1, j
        else:
            return z, i+1, j+1
    else:
        if (y < z):
            return y, i, j+1
        else:
            return z, i+1, j+1


if __name__ == '__main__':
    # inp = [
            # [11,9, 3],
            # [3, 1, 0],
            # [1, 3, 2]
            # ]
    inp = [[i+j for i in range(10)] for j in range(10)]
    [print(i) for i in inp]
    t0 = time.clock()
    res = test(inp, inp[0][0], 0, 0, 8, 8)
    t1 = time.clock()
    res2 = minCost(inp, 8, 8)
    t2 = time.clock()
    res3 = minCost2(inp, 8, 8)
    t3 = time.clock()
    print(res, res2, res3)
    print(t1-t0)
    print(t2-t1)
    print(t3-t2)
