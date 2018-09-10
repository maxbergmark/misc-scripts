from time import time
import numpy as np

def main():
    start = time()
    primes = prime_generator(10**4)
    result = highly_divisible_triangular_number(primes)
    print("Highly divisible triangular number is", result)
    print("It took {} seconds.".format(time() - start))

def prime_generator(max_prime):

    composites = set()
    primes = []

    for number in range(2, max_prime):
        if number not in composites:
            primes.append(number)
            composites.update(range(number**2, max_prime, number))

    return primes

def find_prime_factor(num, primes, indexes):

    for i, div in enumerate(primes):

        if div >= num:
            indexes.append(i)
            return indexes
        if num % div == 0:
            div_2 = num / div
            indexes.append(i)
            find_prime_factor(div_2, primes, indexes)
            break

    return indexes

def highly_divisible_triangular_number(primes):

    n = 4

    while True:

        triangular_number = n*(n - 1)/2
        indexes = find_prime_factor(triangular_number, primes, [])

        count = np.ones(max(indexes) + 1)

        for j in indexes:
            count[j] += 1

        div_num = count.prod()

        if div_num >= 500:
            return triangular_number

        n += 1

main()