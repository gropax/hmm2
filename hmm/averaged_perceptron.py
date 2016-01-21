from __future__ import division, print_function, unicode_literals, with_statement
from collections import defaultdict

class AveragedPerceptron:
    """
    A simple multi-class AveragedPerceptron
    """

    def __init__(self, default_label="NOUN"):

        # self.weights[class][features] --> importance of `feature` to
        # predict `class`
        self.weights = defaultdict(lambda: defaultdict(float))
        # the accumulated value for a pair class/feature
        self._cached = defaultdict(lambda: defaultdict(float))
        # number of instances seen
        self.n_updates = 0.0

    def __perceptron_update(self, f, l, how_much):
        self.weights[l][f] += how_much
        self._cached[l][f] += self.n_updates * how_much

    def update(self, true_label, guessed_label, true_features, guessed_features=None):
        """
        Implements the perceptron rule to update weight vector:

            w_{t+1} = w_t + phi(true_label, true_features) - phi(guessed_label, guessed_features)

        If `guessed_features` is `None`, the feature vector is assumed
        to be the same for the true and the predicted label.
        """

        self.n_updates += 1

        if true_label == guessed_label:
            return

        if guessed_features is None:
            for f in true_features:
                self.__perceptron_update(f, true_label, +1)
                self.__perceptron_update(f, guessed_label, -1)
        else:
            for f in true_features:
                self.__perceptron_update(f, true_label, +1)
            for f in guessed_features:
                self.__perceptron_update(f, guessed_label, -1)

    def score(self, features, labels=None):

        if labels is None:
            labels = self.weights.keys()

        scores = defaultdict(float)
        for c in labels:
            scores[c] = sum(features[f] * self.weights[c][f] for f in features)

        return scores

    def predict(self, features, possible_labels=None):
        """
        Predict the label associated to a feature vector

        Parameters
        ----------
        - features, a dictionnary
            a dictionnary mapping feature names to their value
        - possible_labels, an iterable
            if not None, only the set of possible labels (i.e. the argmax of
            the decision rule) is reduced to this set
        """
        scores = self.score(features)
        if possible_labels is not None:
            scores = {k: v for k, v in scores.items() if k in possible_labels}

        return max(scores, key=lambda label: (scores[label], label))

    def average_weights(self):

        new_weigths = defaultdict(lambda: defaultdict(float))
        t = 1.0 / self.n_updates
        for label in self.weights:
            for f in self.weights[label]:
                self.weights[label][f] -= t * self._cached[label][f]


    def decision_function(self, features, label):
        return sum(features[f] * self.weights[label][f] for f in features)

