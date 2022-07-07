import numpy as np
from PyBasicDTW.core import Core, DistanceMetric, StepPattern

class NeighbourExclusion:
    @staticmethod
    def Match(targetIndex:int, searchArray:np.ndarray, matchLength:int) -> None:
        # Exclude points within the current matched period. As numpy arrays pass by reference, the resulting list does not need to be passed back.
        pass

    @staticmethod
    def Distance(targetIndex:int, searchArray:np.ndarray, distance:int) -> None:
        # Exclude points within set distance from center. As numpy arrays pass by reference, the resulting list does not need to be passed back.
        # ensure searchArray is a float
        if not searchArray.dtype == np.float64: raise TypeError("SearchArray must of be of a float64 numpy array type.")
        # 1. create a mask
        mask = np.zeros(searchArray.shape, bool)
        leftMostIdx = targetIndex - distance
        if leftMostIdx < 0: leftMostIdx = 0
        rightMostIdx = targetIndex + distance + 1
        # rightMostIdx always is 1+ count for slicing, as the end index is not included.
        if rightMostIdx > searchArray.shape[0]: rightMostIdx = searchArray.shape[0]
        mask[leftMostIdx:rightMostIdx] = True
        # 2. set masked items to np.inf
        searchArray[mask] = np.inf

    @staticmethod
    def LocalMaximum(targetIndex:int, searchArray:np.ndarray) -> None:
        # Exclude points within the local maximuim on either side of the center point. As numpy arrays pass by reference, the resulting list does not need to be passed back.
        # ensure searchArray is a float
        if not searchArray.dtype == np.float64: raise TypeError("SearchArray must of be of a float64 numpy array type.")
        ## 1. Find neighbour diff, complete left and right sepearte to allow negtaive index
        leftArray = searchArray[:targetIndex+1]
        # flip left array to allow negative diff
        leftDiff = np.flip(np.diff(np.flip(leftArray)))
        rightArray = searchArray[targetIndex:]
        rightDiff = np.diff(rightArray)
        differences = np.concatenate((leftDiff, np.array([0]), rightDiff))
        ## 2. Create a mask to see any postivie gradients
        maximums = np.argwhere(differences>0).flatten()
        ## 3. Find first left most maximum
        leftMaximum = np.argwhere(maximums<targetIndex).flatten()
        leftMaxIdx = None
        if leftMaximum.shape[0] > 0: leftMaxIdx = maximums[leftMaximum[0]]+1
        if leftMaximum.shape[0] == 0: leftMaxIdx = 0
        ## 4. Find first right most maximum
        rightMaximum = np.argwhere(maximums>targetIndex).flatten()
        rightMaxIdx = None
        if rightMaximum.shape[0] > 0: rightMaxIdx = maximums[rightMaximum[0]]
        if rightMaximum.shape[0] == 0: rightMaxIdx = searchArray.shape[0]
        ## 3. create mask
        mask = np.zeros(searchArray.shape, bool)
        mask[leftMaxIdx:rightMaxIdx] = True
        # 2. Set masked items to np.inf
        searchArray[mask] = np.inf

class SDTW(Core):
    def __init__(self, x:np.ndarray, y:np.ndarray, distanceMetric:DistanceMetric = DistanceMetric.EUCLIDEAN, stepPattern:StepPattern = StepPattern.CLASSIC, stepWeights:np.ndarray=np.array([]), dimensionWeights:np.ndarray=np.array([])) -> None:
        # Initalise CORE
        super().__init__(distanceMetric=distanceMetric, stepPattern=stepPattern, stepWeights=stepWeights)
        # compute match
        self.__lCost, self.__aCost = self.CostMatrix(x, y, dimensionWeights)
        # collate all possible end points
        self.__endPoints = np.copy(self.__aCost[-1,:])
        # variable to store ordered matches with each match a tuple in the format (path, totalCost)
        self.__matches:list = []

    @property
    def AccumulatedCostMatrix(self):
        return self.__aCost

    @property
    def LocalCostMatrix(self):
        return self.__lCost
