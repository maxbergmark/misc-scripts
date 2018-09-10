'''
import numpy as n
def f(x,y):
 a,i,j,k=n.array([[-i-j for i in range(x)]for j in range(y)]),1,0,1
 while i<=x*y:
  a[a==j],i,j,k=n.arange(i,i+k),i+k,j-1,n.sum(a==j-1)
 return a

import numpy as n
def g(x,y):
 a,i,k=n.array([[i+j for i in range(x)]for j in range(y)]),1,2
 for j in range(x+y-1):
  a[a==j],i,k=-n.arange(i,k),k,k+n.sum(a==j+1)
 return-a

import numpy
def m(x,y):
 a,i,k=numpy.array([[-i-j for i in range(x)]for j in range(y)]),1,2
 for j in range(x+y-1):
  a[a==-j],i,k=range(i,k),k,k+n.sum(a==-j-1)
 return a
'''
from numpy import*
r=range
def h(x,y):
 a,i,k,j=-array([i//y+i%y for i in r(x*y)]),1,2,0
 while j<x+y:a[a==-j],i,k,j=r(i,k),k,k+sum(a==~j),j+1
 a.shape=x,y;return a

x = 5
y = 5
# a = f(x,y)
# b = g(x,y)
c = h(x,y)
# print(a)
# print(b)
print(c)
# print(n.all(a==b), n.all(a==c))
