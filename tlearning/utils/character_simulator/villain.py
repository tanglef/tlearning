import numpy as np
import torch


class Villain:
    def __init__(self, nclasses):
        """The villain is the adversarious attacks labeller, they will not tell
        the truth. The strategy matrix is:
            P[i,j] = P(ŷ=i|y=j) = 1/K

        Args:
            nclasses (int): number of possible classes. In a binary setting
                1 (resp -1) becomes -1 (resp 1). In multiclass settings, the
                class answered is drawn with uniform probability over all
                categories.
        """
        self.nclasses = nclasses
        assert type(self.nclasses) is int

    def answer(self, y):
        if self.nclasses < 2:
            raise ValueError("We should have at least two classes.")
        elif self.nclasses == 2:
            ans = -y
            return -y
        else:
            ans = torch.zeros_like(y)
            for i in range(len(y)):
                r = list(range(1, self.nclasses))
                ans[i] = np.random.choice(r)
            return ans
