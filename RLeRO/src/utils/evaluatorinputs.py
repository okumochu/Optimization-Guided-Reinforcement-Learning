
class EvaluatorInput:
    def __init__(self):
        self.__sets()
        self.__params()

    def __sets(self):
        self.I = None
        self.P_hat = None

    def __params(self):
        self.C_T = None
        self.D_hat = None
        self.C_S = None
        self.C_L = None
        self.A_hat = None
        self.V = None
        self.S_I = None
        self.X = None
