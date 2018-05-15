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
    print(i)

for i in a:
    print(i)

        # #Compute beta
        # #b         = beta * A
        # #E[X]      = β0     +β1E[U1]     +. . .+βkE[Uk].
        # #E[X · Ui] = β0E[Ui]+β1E[U1 · Ui]+. . .+βkE[Uk · Ui].
        # #A = np.matrix(np.zeros((Kp+1, Kp+1)))
        # A = np.array(np.zeros((Kp+1, Kp+1)))
        # b = np.zeros(Kp+1)
        # #for the first equation
        # A[0, 0] = 1
        # b[0] = deepcopy(self.get_E(X))
        # for k in range(Kp):
        #     U_k = parents[k]
        #     A[0, k+1] = deepcopy(self.get_E(U_k))
        
        # #for the rest equations
        # for i in range(Kp):# for row i+1
        #     U_i = parents[i]
        #     A[i+1, 0] = deepcopy(self.get_E(U_i))
        #     b[i+1]    = deepcopy(self.get_E((X, U_i)))
        #     for k in range(Kp):
        #         U_k = parents[k]
        #         A[i+1, k+1] = deepcopy(self.get_E((U_k, U_i)))
        # beta = np.linalg.solve(A, b)
        # #Compute var
        # #Cov[X;X] = E[X · X]−E[X] · E[X]
        # #Cov[X;Ui] = E[X · Ui]−E[X] · E[Ui]
        # #var = Cov[X;X]−SIGMA{βiβjCov[Ui;Uj]}
        # var = self.get_E((X, X)) - self.get_E(X)**2
        # #print("var=",var)
        # for i in range(Kp):
        #     U_i = parents[i]
        #     for j in range(Kp):
        #         U_j = parents[j]
        #         var = var - beta[i+1] * beta[j+1] * (self.get_E((U_i, U_j)) - self.get_E(U_i) * self.get_E(U_j))
        # var = var + 1.0e-5
        # assert(var >= 0)
