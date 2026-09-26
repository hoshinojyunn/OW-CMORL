import numpy as np
from abc import abstractmethod
import torch

from .temporal import adapt_weights, normalize_weights

class ScalarizationFunction():
    def __init__(self, num_objs, weights = None):
        self.num_objs = num_objs
        if weights is not None:
            self.weights = torch.Tensor(weights)
        else:
            self.weights = None
    
    def update_weights(self, weights):
        if weights is not None:
            self.weights = torch.Tensor(weights)

    @abstractmethod
    def evaluate(self, objs):
        pass

class WeightedSumScalarization(ScalarizationFunction):
    def __init__(self, num_objs, weights = None):
        super(WeightedSumScalarization, self).__init__(num_objs, weights)
    
    def update_z(self, z):
        pass

    def evaluate(self, objs):
        return (objs * self.weights).sum(axis = -1)


class DynamicWeightedSumScalarization(WeightedSumScalarization):
    def __init__(self, num_objs, weights=None):
        super().__init__(num_objs, weights)
        self.base_weights = None if weights is None else np.asarray(weights, dtype=np.float64)
        self.drift_score = 0.0
        self.knee_score = 0.0

    def update_dynamic_state(self, gap_vector, drift_score, knee_score):
        if self.base_weights is None:
            self.base_weights = normalize_weights(np.ones(self.num_objs))
        self.drift_score = float(drift_score)
        self.knee_score = float(knee_score)
        self.update_weights(
            adapt_weights(self.base_weights, np.asarray(gap_vector), self.drift_score, self.knee_score)
        )
