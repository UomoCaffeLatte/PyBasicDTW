from typing import Tuple
import numpy as np
from types import LambdaType
from enum import Enum

class DistanceMetric(Enum):
    EUCLIDEAN = 1
    ABSOLUTE = 2

class StepPattern(Enum):
    # NOTE: Ensure ordering is largest step to smallest as this affects the selection when all steps present same cost.
    CLASSIC = np.array([(1,1),(1,0),(0,1)])

class Core:
    def __init__(self, distanceMetric:DistanceMetric = DistanceMetric.EUCLIDEAN, stepPattern:StepPattern = StepPattern.CLASSIC, stepWeights:np.ndarray=np.array([])) -> None:
        self.distanceMetrics:dict = {"EUCLIDEAN":lambda x,y: np.square(x,y), "ABSOLUTE":lambda x,y: np.abs(x-y)}
        self.metric = None
        self.stepPattern = None
        self.stepWeights = None
        self.dimWeights = None
        self.localCost = None
        self.accumulatedCost = None
        # type Check DistanceMetric & StepPattern
        if not isinstance(distanceMetric, DistanceMetric): raise TypeError("Distance metric must be of DistanceMetric type.")
        if not isinstance(stepPattern, StepPattern): raise TypeError("Step pattern must be of StepPattern type.")
        self.stepPattern = stepPattern.value
        self.metric = self.distanceMetrics[distanceMetric.name]
        # if stepweights are not provided, create default
        if stepWeights.size == 0: self.stepWeights = np.ones((self.stepPattern.shape[0]))
        # if stepweights are provided, check they match the stepPattern
        if stepWeights.size > 0:
            if not isinstance(stepWeights, np.ndarray): raise TypeError("StepWeights must be of numpy array (ndarray) type.")
            if stepWeights.ndim != 1: raise ValueError("StepWeights must be a 1 dimensional numpy array.")
            if stepWeights.shape[0] != self.stepPattern.shape[0]: raise ValueError(f"StepWeights do not match StepPattern, SW:{stepWeights.shape[0]} != SP:{self.stepPattern.shape[0]}")
            self.stepWeights = stepWeights

    def CostMatrix(self, x:np.ndarray, y:np.ndarray, dimensionWeights:np.ndarray=np.array([]), sdtw:bool=False) -> Tuple:
        # NOTE: x and y arrays must be given in the list format where each item corresponds to the values at some time t.
        # NOTE: x is the sequence to find within y for sdtw
        ## VALIDATION ##
        # validate both sequences have the same dimensions
        if not isinstance(x, np.ndarray): raise TypeError("x must be of numpy array (ndarray) type.")
        if not isinstance(y, np.ndarray): raise TypeError("y must be of numpy array (ndarray) type.")
        if x.shape[1] != y.shape[1]: raise ValueError("x and y must have equal dimensions along axis 1.")
        # if dimensionWeights are not provided, create default
        if dimensionWeights.size == 0: self.dimWeights = np.ones((x.shape[1]))
        # if dimension weights are provided, check they match dimensions
        if dimensionWeights.size > 0:
            if not isinstance(dimensionWeights, np.ndarray): raise TypeError("DimensionWeights must be of numpy array (ndarray) type.")
            if dimensionWeights.ndim != 1: raise ValueError("DimensionWeights must be a 1 dimensional numpy array.")
            if dimensionWeights.shape[0] != x.shape[1]: raise ValueError(f"DimensionWeights do not match dimensions: DW:{dimensionWeights.shape[0]} != D:{x.shape[1]}.")
            self.dimWeights = dimensionWeights
        ## /VALIDATION ##
        ## COSTCALC ##
        # To speed up the cost calculation we will be using vectorisation methods.
        # 1. If x and y are not equal shapes, increase dimension of smaller array so that numpy can broadcast.
        if x.shape[0] > y.shape[0]: y = y[:,np.newaxis]
        if y.shape[0] > x.shape[0]: x = x[:,np.newaxis]
        # 2. calculate distance metric using numpy broadcasting
        self.localCost = self.metric(x,y) * self.dimWeights
        # 3. Sum each index using numpy broadcasting, each row corresponds to one x value along all y values 
        self.localCost = np.sum(self.localCost, axis=2)
        # tranpose matrix to negative effects of broadcasting when x is larger than y
        if x.shape[0] > y.shape[0]: self.localCost = np.transpose(self.localCost)
        # The accumulated cost matrix is more complicated to calculate.
        # 1. Create an empty extended accumulated cost matrix dependent on the StepPattern
        ## 1a. Find the maximum step in x and y direction
        maxStep = np.array([self.stepPattern[np.argmax(self.stepPattern[:,0]),0], self.stepPattern[np.argmax(self.stepPattern[:,1]),1]])
        ## 1b. Create an empty matrix that is padded with this maxStep
        self.accumulatedCost = np.zeros(shape=( (self.localCost.shape[0]+maxStep[0]), (self.localCost.shape[1]+maxStep[1]) ))
        ## 1c. Record the true start index due to the padding 
        realZeroIndex = maxStep
        ## 1d. Set default values for padded areas
        self.accumulatedCost[:realZeroIndex[0],:] = np.inf
        self.accumulatedCost[:,:realZeroIndex[1]] = np.inf
        self.accumulatedCost[:realZeroIndex[0],:realZeroIndex[1]] = 0
        # 1e. If sdtw allow for multiple starting points along y
        if sdtw: self.accumulatedCost[:realZeroIndex[0], realZeroIndex[1]:] = 0
        # 2. Calculate accumulated cost
        for r in range(realZeroIndex[0], self.accumulatedCost.shape[0]):
            for c in range(realZeroIndex[1], self.accumulatedCost.shape[1]):
                # create mask from step choices
                mask = np.zeros(self.accumulatedCost.shape, bool)
                stepChoices = (r,c) - self.stepPattern
                mask[stepChoices[:,0],stepChoices[:,1]] = True
                # obtain lexi arg min value from mask
                self.accumulatedCost[r,c] = self.accumulatedCost[mask][self.LexiMin(self.accumulatedCost[mask])] + self.localCost[r-realZeroIndex[0],c-realZeroIndex[1]]
        # 3. Remove padding for final complete accumulated cost matrix
        self.accumulatedCost = self.accumulatedCost[realZeroIndex[0]:, realZeroIndex[1]:]
        return self.localCost, self.accumulatedCost

        ## /COSTCALC ##
    def WarpingPath(self) -> Tuple:
        pass

    def LexiMin(self, list:np.ndarray, invert:bool=False) -> int:
        """ Get the lexiographical minimum in a 1D array.

        Lexiographical: generalisation of alphabetical order of the dictionaries to sequence of ordered elements.        

        If invert is False: if there are two minimmums, the minimum with the smallest index will be chosen.
        If invert is True : if there are two minimums, the minimum with the largest index will be chosen.

        Args:
            list (np.ndarray): 1D numpy array of elements.
            invert (bool, optional): Invert lexiographical order of non-unique minimums. Defaults to False.

        Returns:
            int: Index of lexiographical minimum value in list.
        """
        # find all occurences which has the lowest cost and flatten to 1D array
        minimums = np.argwhere(list == np.amin(list)).flatten()
        # if only one minimum fond / UNIQUE
        if len(minimums) == 1: return minimums[0]
        # if more than one minimum found / NON-UNIQUE
        if invert: return np.amax(minimums)
        return np.amin(minimums)

