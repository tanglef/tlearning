import torch


class Wizard():
    def __init__(self, nclasses, strategy):
        """The wizard is a know it all. If a seer, his answers are always
        correct.
        If peerpressured, then all wizards answer after other participants and
        choosethe most voted class.
        Note that this removes the hypothesis that labellers are independant
        but is closer to reality where people can change their choice in a poll
        based on the results they see.
        The strategy matrix is:
            P[i,j] = P(ŷ=i|y=j) = 1(i=j) if strategy is seer,
                                = 1(argmax_j others=i) else.

        Args:
            nclasses (int): possible number of classes to choose from.
            strategy (str): either 'seer' or 'peerpressure':
                - seer: always answer the right category,
                - pressured: answer the category of the majority.
            others_result (Tensor): Tensor of shape (batch_size, n_classes)
                containing the number of votes for each class by other voters.
        """
        self.nclasses = nclasses
        assert type(self.nclasses) is int
        self.strategy = strategy
        assert strategy in ["seer", "pressured"]

    def answer(self, y, others_result=None):
        self.others = others_result
        if self.strategy == "pressured" and self.others is None:
            raise ValueError(
                "Others result is needed with peerpressure strategy")
        if self.strategy == "seer":
            return y
        else:
            assert len(y) == len(self.others)
            return torch.argmax(self.others, dim=1)
