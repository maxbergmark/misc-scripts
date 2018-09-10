class ASDF:
    def __getattr__(self, obj):
        obj = obj+1
        return obj

    def __init__(self):
        self.a = 2
        print(self.a)

    def __repr__(self):
        return self.a
    def __str__(self):
        return self.a
a = ASDF()
print(a)

