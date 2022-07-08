import numpy as np
from pybasicdtw.core import Core, DistanceMetric, StepPattern

class DTW(Core):
    def __init__(self, x:np.ndarray, y:np.ndarray, distanceMetric:DistanceMetric = DistanceMetric.EUCLIDEAN, stepPattern:StepPattern = StepPattern.CLASSIC, stepWeights:np.ndarray=np.array([]), dimensionWeights:np.ndarray=np.array([])) -> None:
        # Initalise CORE
        super().__init__(distanceMetric=distanceMetric, stepPattern=stepPattern, stepWeights=stepWeights)
        # compute match
        self.__lCost, self.__aCost = self.CostMatrix(x, y, dimensionWeights)
        # calculate optimal path
        self.__path, self.__totalCost = self.WarpingPath(self.__aCost, self.__lCost)

    @property
    def AccumulatedCostMatrix(self):
        return self.__aCost

    @property
    def LocalCostMatrix(self):
        return self.__lCost

    @property
    def MatchPath(self):
        return self.__path

    @property
    def TotalCost(self):
        return self.__totalCost