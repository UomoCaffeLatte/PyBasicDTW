import numpy as np
from types import LambdaType
from typing import Tuple, List
import logging as lg

class DTW:
    ''' Performs cost matrix and optimal warping path calculations.'''

    def __init__(self,
                 x:np.ndarray,
                 y=np.ndarray,
                 sdtw:bool=False,
                 dimensionWeights:np.ndarray=np.array([]),
                 distanceMetric:LambdaType=lambda x,y: np.square(x-y),
                 stepPattern:np.ndarray=np.array([(1,1),(1,0),(0,1)]),
                 stepWeights:np.ndarray=np.array([])) -> None:
        # All logic is seperated for debugWalkThrough functionality
        self.__x:np.ndarray = x
        self.__y:np.ndarray = y
        self.__sdtw:bool = sdtw
        self.__dimWeights:np.ndarray = dimensionWeights
        self.__metric:LambdaType = distanceMetric
        self.__stepPattern:np.ndarray = stepPattern
        self.__stepWeights:np.ndarray = stepWeights
        self.__lCost:np.ndarray = np.array([])
        self.__aCost:np.ndarray = np.array([])
        # check x and y
        if self.__x.shape[1] != self.__y.shape[1]: raise ValueError("x and y must be equal dimensions along axis 1.")
        # check distance metric
        if not isinstance(distanceMetric, LambdaType): raise TypeError("DistanceMetric must be a Callable type.")
        if distanceMetric.__code__.co_argcount != 2: raise ValueError("DistanceMetric Callable must have only two inputs.")
        # fill empty stepPattern, stepWeights, and dimensionWeights
        if self.__stepPattern.size == 0: self.__stepPattern = np.array([(1,1),(1,0),(0,1)]) # Classic step pattern
        if self.__stepWeights.size == 0: self.__stepWeights = np.ones((self.__stepPattern.shape[0]))
        if self.__dimWeights.size == 0: self.__dimWeights = np.ones((x.shape[1]))
        # check stepWeights & dimWeights
        if self.__stepWeights.shape[0] != self.__stepPattern.shape[0]: raise ValueError(f"StepWeights do not match StepPattern, StepWeights:{self.__stepWeights.shape[0]} != StepPatten:{self.__stepPattern.shape[0]}")
        if self.__dimWeights.shape[0] != self.__x.shape[1]: raise ValueError("DimWeights length do not match number of dimensions.")
        
        # Compute cost matrix
        self.__CostMatrix()
        
    @property
    def stepPattern(self):
        return self.__stepPattern
    
    @property
    def stepWeights(self):
        return self.__stepWeights
    
    @property
    def dimWeights(self):
        return self.__dimWeights
    
    @property
    def distanceMetric(self):
        return self.__metric

    @property
    def localCost(self):
        return self.__lCost
    
    @property
    def accumulatedCost(self):
        return self.__aCost
        
    def __CostMatrix(self) -> None:
        # Calculate Local Cost Matrix
        self.__lCost = self.__CalculateLocalCost()

        # Check Size
        if self.__lCost.shape[0] != self.__x.shape[0]: raise ValueError("Mis-shaped local cost matrix rows.")
        if self.__lCost.shape[1] != self.__y.shape[0]: raise ValueError("Mis-shaped local cost matrix cols.")
        
        # Calculate Accumulated Cost Matrix
        self.__aCost = self.__CalculateAccumulatedCost()
        
        # Check Size
        if self.__aCost.shape[0] != self.__x.shape[0]: raise ValueError("Mis-shaped accumulated cost matrix rows.")
        if self.__aCost.shape[1] != self.__y.shape[0]: raise ValueError("Mis-shaped accumulated cost matrix cols.")
    
    def __CalculateLocalCost(self) -> np.ndarray:
        # speed up using numpy broadcasting
        x = self.__x
        y = self.__y
        
        # 1. If x and y are not equal shapes, increase dimension of smaller array so that numpy can broadcast.
        if x.shape[0] > y.shape[0]: y = y[:,np.newaxis]
        if y.shape[0] > x.shape[0]: x = x[:,np.newaxis]
        if x.shape[0] == y.shape[0]: x = x[:,np.newaxis] # to ensure proper broadcasting
        
        # 2. calculate distance metric using numpy broadcasting
        localCost = self.__metric(x,y) * self.__dimWeights
        
        # 3. Sum each index using numpy broadcasting, each row corresponds to one x value along all y values 
        localCost = np.sum(localCost, axis=2)
        
        # 4. tranpose matrix to negative effects of broadcasting when x is larger than y
        if x.shape[0] > y.shape[0]: localCost = np.transpose(localCost)
        return localCost
    
    def __CalculateAccumulatedCost(self) -> np.ndarray:
        if self.__lCost.size == 0: raise ValueError("Cannot calculate accumulated cost without local cost matrix.")
        # 1. Create an empty extended accumulated cost matrix dependent on the StepPattern
        ## 1a. Find the maximum step in x and y direction
        maxStep = np.array([self.__stepPattern[np.argmax(self.__stepPattern[:,0]),0], self.__stepPattern[np.argmax(self.__stepPattern[:,1]),1]])
        
        ## 1b. Create an empty matrix that is padded with this maxStep
        accumulatedCost = np.zeros(shape=( (self.__lCost.shape[0]+maxStep[0]), (self.__lCost.shape[1]+maxStep[1]) ))
        
        ## 1c. Record the true start index due to the padding 
        realZeroIndex = maxStep
        
        ## 1d. Set default values for padded areas
        accumulatedCost[:realZeroIndex[0],:] = np.inf
        accumulatedCost[:,:realZeroIndex[1]] = np.inf
        accumulatedCost[:realZeroIndex[0],:realZeroIndex[1]] = 0
        
        # 1e. If sdtw allow for multiple starting points along y
        if self.__sdtw: accumulatedCost[:realZeroIndex[0], realZeroIndex[1]:] = 0
        
        # 2. Calculate accumulated cost
        for r in range(realZeroIndex[0], accumulatedCost.shape[0]):
            for c in range(realZeroIndex[1], accumulatedCost.shape[1]):
                # create mask from step choices
                mask = np.zeros(accumulatedCost.shape, bool)
                stepChoices = (r,c) - self.__stepPattern
                mask[stepChoices[:,0],stepChoices[:,1]] = True
                
                # obtain lexi arg min value from mask
                minCostIndex = self.__LexiMin(accumulatedCost[mask])
                accumulatedCost[r,c] = (accumulatedCost[mask][minCostIndex]*self.__stepWeights[minCostIndex]) + self.__lCost[r-realZeroIndex[0],c-realZeroIndex[1]]
        
        # 3. Remove padding for final complete accumulated cost matrix
        accumulatedCost = accumulatedCost[realZeroIndex[0]:, realZeroIndex[1]:]
        return accumulatedCost
    
    def WarpingPath(self,
                    endIndex:Tuple=None) -> Tuple:
        forceStart = True
        if self.__sdtw: forceStart = False
        # value checking
        if endIndex == None and self.__sdtw == True: raise ValueError("Please enter an endIndex for sdtw case.")
        if self.__sdtw == False: 
            if endIndex != None:
                lg.warn("For classical dtw the endIndex arg is not required")
            # calculate end point for classical dtw
            endIndex = (self.__aCost.shape[0]-1, self.__lCost.shape[1]-1)

        # find warping path
        path = [endIndex]
        startReached = False
        while startReached == False:
            # calculate all potential step choices
            stepChoices = path[-1] - self.__stepPattern
            stepCosts = []
            for step in stepChoices:
                stepCosts.append(self.__aCost[step[0],step[1]])
            stepCosts = np.array(stepCosts)

            # Find lowest costing step and ignore negative stepChoices by setting aCost to INF
            # find index of elements with negative index and set cost to inf
            negativeIndices = np.argwhere(stepChoices<0)[:,0]
            for idx in negativeIndices: 
                stepCosts[idx] = np.inf
            optimalStep = stepChoices[self.__LexiMin(stepCosts)]
            path.append(optimalStep)
            
            # check condition x == 0
            if optimalStep[1] == 0 and optimalStep[0] != 0:
                # can only traverse down from x==0 so rest of warping path is known
                for i in range(optimalStep[0]-1, -1, -1):
                    path.append((i,0))
                break
                
            # check if start point reached
            if forceStart and (path[-1] == (0,0)).all(): startReached = True
            if not forceStart and path[-1][0] == 0: startReached = True
            
        # calculate total dtw cost
        path = np.array(path)
        mask = np.zeros(self.__lCost.shape, bool)
        mask[path[:,0],path[:,1]] = True
        totalCost = np.sum(self.__lCost[mask])
        return path, totalCost
        
    def __LexiMin(self, list:np.ndarray, invert:bool=False) -> int:
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