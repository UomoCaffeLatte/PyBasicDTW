import unittest
import numpy as np
from PyBasicDTW import SDTW, NeighbourExclusion

class NeighbourExclusion_unitTests(unittest.TestCase):

    def test_distanceExclusion_Leftmost(self):
        # Arrange
        endPoints = np.array([50,80,90,58,100,20,30,40,50,60], dtype="float64")
        # Act
        NeighbourExclusion.Distance(targetIndex=0, searchArray=endPoints, distance=3)
        # Assert
        self.assertTrue(np.all(endPoints[0:4] == np.inf))
        self.assertTrue(endPoints[endPoints == np.inf].shape[0] == 4)

    def test_distanceExclusion_Rightmost(self):
        # Arrange
        endPoints = np.array([50,80,90,58,100,20,30,40,50,60], dtype="float64")
        # Act
        NeighbourExclusion.Distance(targetIndex=endPoints.shape[0]-1, searchArray=endPoints, distance=3)
        # Assert
        self.assertTrue(np.all(endPoints[-3:endPoints.shape[0]] == np.inf))
        self.assertTrue(endPoints[endPoints == np.inf].shape[0] == 4)

    def test_distanceExclusion_Random(self):
        for _ in range(0,100):
            # Arrange
            endPoints = np.random.default_rng().uniform(0,1000,50)
            distance = np.random.randint(0,50)
            targetIndex = np.random.randint(0,50)
            # Act
            NeighbourExclusion.Distance(targetIndex=targetIndex, searchArray=endPoints, distance=distance)
            # Assert
            startIndex = targetIndex - distance
            if startIndex < 0: startIndex = 0
            endIndex = targetIndex + distance + 1
            if endIndex > endPoints.shape[0]: endIndex = endPoints.shape[0]
            self.assertTrue(np.all(endPoints[startIndex:endIndex] == np.inf))
            self.assertTrue(endPoints[endPoints == np.inf].shape[0] == (endIndex - startIndex))

    def test_distanceExclusion_typeFail(self):
        # Arrange
        endPoints = np.array([50,80,90,58,100,20,30,40,50,60], dtype="int")
        # Act and Assert
        with self.assertRaises(TypeError):
            NeighbourExclusion.Distance(targetIndex=0, searchArray=endPoints, distance=3)
        