import re
# s=input()
s="aqrst,ukaei,ffooo"
w=s.find(",")+1
v='[aeiou]'*2
m=re.search(v+'.'*(w-2)+v,s)
a=m.start();print m and(a/w,a%w)or'not found'
print m and divmod(m.start(),w)or'not found'