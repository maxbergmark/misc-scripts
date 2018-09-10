n=int(input())
a=[' '*(n+~i)+'+'*(2*i+1)for i in range(n)]
print('\n'.join(['+'*n]+a+a[::-1]))
# a=lambda l:l+l[::-1]
# f=lambda n:'\n'.join(['+'*n]+a([(' '*n+'+'*n*2)[2:8]for i in range(n)]))
# print(f(5))