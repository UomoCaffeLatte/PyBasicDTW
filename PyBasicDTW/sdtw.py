from pybasicdtw.dtw import DTW
from typing import Tuple, Callable
import numpy as np
import warnings

class NeighbourExclusion:
    @classmethod
    def Distance(cls,targetIndex:int, searchArray:np.ndarray, **kwargs) -> None:
        # Exclude points within set distance from center. As numpy arrays pass by reference, the resulting list does not need to be passed back.
        # ensure searchArray is a float
        if not searchArray.dtype == np.float64: raise TypeError("SearchArray must of be of a float64 numpy array type.")
        if kwargs.get("distance","None") == "None": raise ValueError("Distance argument missing.")
        # 1. create a mask
        mask = np.zeros(searchArray.shape, bool)
        leftMostIdx = targetIndex - kwargs.get("distance")
        if leftMostIdx < 0: leftMostIdx = 0
        rightMostIdx = targetIndex + kwargs.get("distance") + 1
        # rightMostIdx always is 1+ count for slicing, as the end index is not included.
        if rightMostIdx > searchArray.shape[0]: rightMostIdx = searchArray.shape[0]
        mask[leftMostIdx:rightMostIdx] = True
        # 2. set masked items to np.inf
        searchArray[mask] = np.inf

    @classmethod
    def LocalMaximum(cls,targetIndex:int, searchArray:np.ndarray, **kwargs) -> None:
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

class SDTW(DTW):
    def __init__(self, **kwargs) -> None:
        super().__init__(sdtw=True, **kwargs)
        # copy of original accumulated cost matrix
        self.__originalAccumulatedCost = np.copy(self.accumulatedCost)
        # retrieve all possible end points list
        self.__endPoints = np.copy(self.accumulatedCost[-1,:])
        self.__matches:list = []
    
    @property
    def OriginalAccumulatedCost(self):
        return self.__originalAccumulatedCost
    
    @property
    def matches(self):
        return self.__matches
    
    def GetEndCost(self, path:np.ndarray) -> float:
        return self.__originalAccumulatedCost[path[0][0],path[0][1]]

    def __Match(self,targetIndex:int, 
              searchArray:np.ndarray, 
              matchTimeLength:int) -> None:
        # Exclude points within the current matched period. As numpy arrays pass by reference, the resulting list does not need to be passed back.
        searchArray[targetIndex-matchTimeLength:targetIndex+1] = np.inf

    def FindMatch(self, neighbourExclusion:Callable=NeighbourExclusion.Distance, 
                  overlapMatches:bool=True, 
                  invertEndPointSelection:bool=True,
                  **kwargs) -> Tuple:
        # check neighbour exclusion is callable
        if not callable(neighbourExclusion): raise TypeError("NeighbourExclusion must be a callable type.")
        # Check if any more matches can be found
        if np.all(self.__endPoints == np.inf):
            warnings.warn("No more matches can be found.")
            return None, None
        # Find optimum end point, check if right or leftmost match to be chosen in non-unique scenario.
        optimumEndPointIdx = self._DTW__LexiMin(self.__endPoints, invert=invertEndPointSelection)
        # Find warping path
        path, totalCost = self.WarpingPath(self._DTW__aCost, 
                                           self._DTW__lCost,
                                           (self._DTW__lCost.shape[0]-1,optimumEndPointIdx))
        # Perform neighbourhood exclusion
        neighbourExclusion(optimumEndPointIdx, self.__endPoints, **kwargs)
        # add match overlap feature onto accumulated cost matrix
        if not overlapMatches: 
            self.__Match(optimumEndPointIdx, 
                                     self.__endPoints, 
                                     (path[0][1] - path[-1][1]))
            self._DTW__aCost[:,path[-1][1]:path[0][1]+1] = np.inf 
            self._DTW__lCost[:,path[-1][1]:path[0][1]+1] = np.inf

        self.__matches.append([path, totalCost])
        return path, totalCost