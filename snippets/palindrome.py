import time

def largestPalindromeProduct(digits):
    num=int(digits*"9")
    maxPalindrome=0
    i=int(num)
    floor=int(num/10) if digits > 1 else -1
    c = 0
    while i > floor:
        j=i
        while j > floor:
            val=str(i*j)
            c += 1
            if val==val[::-1]:
                floor=j
                maxPalindrome=max(maxPalindrome,i*j)
            j-=1
        i-=1
    # print("count1", c)
    return maxPalindrome

def largestPalindromeProduct2(digits):
    num = 10**digits-1
    maxPalindrome = 0
    i = int(num)
    floor = num//10
    while i > floor:
        j = i
        while j > floor:
            val=str(i*j)
            if val==val[::-1]:
                floor=j
                maxPalindrome=max(maxPalindrome,i*j)
            j-=1
        i-=1
    return maxPalindrome

def palindrome(digits):
    maxval = 0
    for i in range(10**digits-1, max(0, 10**(digits)-10**(digits/2)-1), -1):
        for j in range(i, max(0, i-1000-10**(digits/2)-1), -1):
            prod = i*j
            sprod = str(prod)
            if (sprod == sprod[::-1]):
                maxval = max(maxval, prod)
                break
    return maxval

def palindrome2(digits):
    if (digits % 2 == 0):
        floor = 10**digits-10**(digits//2)+1
        # print("floor", floor)
        maxPalindrome = floor*(10**digits-1)
        # return maxPalindrome
    else:
        floor = 10**digits-10**(digits//2+1)+1
        maxPalindrome=0
    i = 10**digits-1
    c = 0
    while i > floor:
        j = i
        while j > floor:
            prod = i*j
            if prod < maxPalindrome:
                break
            val=str(prod)
            c += 1
            if val == val[::-1]:
                floor=j
                # print(i, j, prod)
                maxPalindrome=max(maxPalindrome,prod)
                break
            j-=2
        i-=2
    # print("count3", c)
    return maxPalindrome

def palindrome3(digits):
    if (digits == 1):
        return 9
    ceil = 10**digits-1
    floor = ceil//10
    for i in range(ceil-1, floor, -1):
        palin = int(str(i)+str(i)[::-1])
        print(palin)
        j = ceil
        while j*j >= palin:
            if palin % j == 0:
                return palin
            j -= 1

for i in range(2, 4, 2):
    print(i)

    # t0 = time.clock()
    # n = largestPalindromeProduct(i)
    # t1 = time.clock()
    # print(n, t1-t0)

    # t0 = time.clock()
    # n = largestPalindromeProduct2(i)
    # t1 = time.clock()
    # print(n, t1-t0)

    t0 = time.clock()
    n = palindrome2(i)
    t1 = time.clock()
    print(n, t1-t0)

    t0 = time.clock()
    n = palindrome3(i)
    t1 = time.clock()
    print(n, t1-t0)