class test:
    def __init__(self):
        self.a = [0,1,2,3,4,5,6,7,8,9]
        self.n = 10
        self.i = 0

    def __iter__(self):
        """
        enumerate all edeges
        """
        #self.i = 0
        return self

    def __next__(self):
        """
        work with __iter__
        """
        if self.i == self.n:
            self.i = 0
            raise StopIteration
        fml = self.a[self.i]
        a = self.i + 1
        self.i = a
        return fml

a = test()
for i in a:
    if i == 6:
        break
    print(i)

for i in a:
    print(i)
