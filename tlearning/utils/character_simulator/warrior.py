import torch


class Warrior():
    def __init__(self, nclasses, strategy, selection=0):
        """The warrior is a brute force spammer: his answers are independant
        of the given input.
        The strategy matrix is:
            P[i,j] = P(ŷ=i|y=j) = 1(i=c) if strategy is duel (c is chosen spam)
                                = 1/K if strategy is brawl.

        Args:
            nclasses (int): possible number of classes to choose from.
            strategy (str): either 'duel' or 'brawl':
                - duel: always answer the same category,
                - brawl: answer any category with uniform probability.
            selection (int): number of the spam class in duel strategy.
        """
        self.nclasses = nclasses
        assert type(self.nclasses) is int
        self.strategy = strategy
        assert strategy in ["duel", "brawl"]
        self.selection = selection

    def answer(self, y):
        self.device = y.device
        if self.strategy == "duel":
            ans = torch.cat(
                [len(y)*torch.Tensor([self.selection]).type(torch.int)])
        else:
            ans = torch.randint(size=(len(y), ), low=0,
                                high=self.nclasses)
        return ans.type(torch.int).to(self.device)
