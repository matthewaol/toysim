import numpy as np


class TestSomething:

    def main(self):
        self.a = 1
        self.b = 2
        self.n = 10

    def test_alpha(self):
        self.main()
        c = self.a+self.b
        assert c == 3

    def test_beta(self):
        self.main()
        u = np.zeros(self.n**2)
        v = u.reshape((self.n,self.n))
        assert u.size == v.size
