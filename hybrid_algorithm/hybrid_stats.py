"""
The statistic module to estimate hybrid diagnosers.
"""

class hybrid_stats:
    """
    This class defines multipule standards to estimate hybrid diagnosers.
    """
    def __init__(self):
        #The real labels
        self.labels = []
        #labels predicted
        self.tank = {}

    def add_diagnoser(self, name):
        """
        add a diagnoser named by name
        """
        if name not in self.tank:
            self.tank[name] = []

    def append_label(self, label):
        """
        append label at the end of self.labels
        """
        self.labels.append(label)

    def append_predicted(self, name, label):
        """
        append predicted labels at the end of the tank
        """
        assert name in self.tank
        assert len(self.labels) == (len(self.tank[name]) + 1)
        self.tank[name].append(label)

    def stats(self, name):
        """
        print the statistic informations one diagnoser
        """
        assert name in self.tank
        _accuracy = self.__accuracy(name)
        _correctness = self.__correctness(name)
        print(name, ":\naccuracy =", _accuracy, ", correctness =", _correctness)

    def print_stats(self):
        """
        print all the statistic information for each diagnoser
        """
        for name in self.tank:
            self.stats(name)

    def __accuracy(self, name):
        """
        accuracy = (number of predicted labels==real labels) / (label number)
        """
        assert name in self.tank
        label_number = len(self.labels)
        counter = 0
        predicted = self.tank[name]
        for i, pred in zip(range(len(self.labels)), predicted):
            for j in pred:
                if (self.labels[i] == j).all():
                    counter = counter + 1
                    break
        accuracy = counter / label_number
        return accuracy

    def __correctness(self, name):
        """
        correntess = (number of not wrong predicted labels) / (predicted number)
        for example:
            real label      = [0, 1, 1, 0, 0, 0]
            predicted label = [0, 1, 0, 0, 0, 0]
            The predicted label is not accurate but correct
        """
        assert name in self.tank
        label_number = len(self.labels)
        counter = 0
        predicted = self.tank[name]
        num = len(predicted[0])
        predicted_number = num * label_number
        for i, pred in zip(range(label_number), predicted):
            for j in range(num):
                if (self.labels[i] >= pred[j]).all():
                    counter = counter + 1

        correctness = counter / predicted_number
        return correctness
