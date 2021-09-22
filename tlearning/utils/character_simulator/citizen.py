import numpy as np
import torch


class Citizen():
    def __init__(self, nclasses, strategy):
        """The citizen is quite naive: if the image is easy to classify they
        will most likely get the right answer, o.w they will choose a category
        from which confusion is possible.
        The creator gives the strategy matrix (technically a citizen could be
        one of the other characters hidden).

        Args:
            nclasses (int): possible number of classes to choose from.
            strategy (ndarray): prior distribution P of size
                (nclasses, nclasses) with P[i,j] = P(ŷ=i|y=j).
        """
        self.nclasses = nclasses
        assert type(self.nclasses) is int
        self.strategy = strategy

    def answer(self, y):
        ans = torch.zeros_like(y)
        for i in range(len(y)):
            val = np.random.choice(
                self.nclasses, p=self.strategy[y.item(), :])
            ans[i] = val
        return ans
