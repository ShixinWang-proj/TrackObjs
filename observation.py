import numpy as np


class Observation:

    def __init__(self, observation):
        self.observation = observation
        self. center = np.mean(observation)
        self.width = 0.5 * (observation[1] - observation[0])
        self.targets = 1
        self.L = int(2 * self.width)

    def predict(self, v):
        return self.observation + v

    def get_e(self, sig):
        e = np.sum(sig[self.observation[0]:self.observation[1]]**2).round(0)
        return e

