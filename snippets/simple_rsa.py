
def egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)

def modinv(a, m):
    g, x, y = egcd(a, m)
    if g != 1:
        raise Exception('modular inverse does not exist')
    else:
        return x % m

p = 102013
q = 103979
n = p * q
phi = (p-1)*(q-1)
e = 98251*98867

# e = 5
d = modinv(e, phi)

inp = int(input())
crypto = pow(inp, e, n)
print(crypto)
decrypto = pow(crypto, d, n)
print(decrypto)
