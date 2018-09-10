p = 9**9
while 1:
    for i in range(2,p):
        if not p%i:
            p-=1
            break
    