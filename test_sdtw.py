import unittest
import unittest.mock as mock
import numpy as np
from pybasicdtw import SDTW, NeighbourExclusion

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
    
    def test_localMaximumExclusion_Leftmost(self):
        # Arrange
        endPoints = np.array([5,5,5,10,4,5,6,10,20,30,40], dtype="float64")
        # Act
        NeighbourExclusion.LocalMaximum(targetIndex=3, searchArray=endPoints)
        # Assert
        self.assertTrue(np.all(endPoints[0:5] == np.inf))
        self.assertTrue(endPoints[endPoints == np.inf].shape[0] == 5)

    def test_localMaximumExclusion_Rightmost(self):
        # Arrange
        endPoints = np.array([5,5,5,10,4,5,6,10,20,15,10], dtype="float64")
        # Act
        NeighbourExclusion.LocalMaximum(targetIndex=8, searchArray=endPoints)
        # Assert
        self.assertTrue(np.all(endPoints[6:endPoints.shape[0]] == np.inf))
        self.assertTrue(endPoints[endPoints == np.inf].shape[0] == 7)

    def test_localMaximumExclusion_Center(self):
        # Arrange
        endPoints = np.array([5,5,5,10,4,5,6,10,20,15,10], dtype="float64")
        # Act
        NeighbourExclusion.LocalMaximum(targetIndex=5, searchArray=endPoints)
        # Assert
        self.assertTrue(np.all(endPoints[4:6] == np.inf))
        self.assertTrue(endPoints[endPoints == np.inf].shape[0] == 2)

    def test_localMaximumExclusion_typeFail(self):
        # Arrange
        endPoints = np.array([50,80,90,58,100,20,30,40,50,60], dtype="int")
        # Act and Assert
        with self.assertRaises(TypeError):
            NeighbourExclusion.LocalMaximum(targetIndex=5, searchArray=endPoints)

    def test_MatchExclusion_Random(self):
        # Arrange
        endPoints = np.random.default_rng().uniform(0,1000,50)
        matchTimeLength = np.random.randint(0,9)
        targetIndex = np.random.randint(10,50) 
        # Act
        NeighbourExclusion.Match(targetIndex=targetIndex, searchArray=endPoints, matchTimeLength=matchTimeLength)
        # Assert
        self.assertTrue(np.all(endPoints[targetIndex-matchTimeLength:targetIndex+1] == np.inf))
        self.assertTrue(endPoints[endPoints == np.inf].shape[0] == matchTimeLength+1)

class SDTW_unitTests(unittest.TestCase):

    @mock.patch("pybasicdtw.core.Core.CostMatrix", return_value=[None, np.array([[100],[100],[100]])])
    def test_init_aCostCopy(self, CostMatrixMock):
        # Arrange and Act
        sdtw = SDTW(None, None)
        # Assert
        self.assertTrue(np.array_equal(sdtw.AccumulatedCostMatrix, np.array([[100],[100],[100]])))
        self.assertTrue(CostMatrixMock.called_once())

    @mock.patch("pybasicdtw.core.Core.CostMatrix", return_value=[None, np.array([[100],[100],[100]])])
    def test_init_endPoints(self,CostMatrixMock):
        # Arrange and Act
        sdtw = SDTW(None, None)
        # Assert
        self.assertTrue(np.array_equal(sdtw._SDTW__endPoints, np.array([100])))
        self.assertTrue(CostMatrixMock.called_once())

    @mock.patch("pybasicdtw.core.Core.CostMatrix", return_value=[np.array([[50],[50],[50]]), np.array([[100],[100],[100]])])
    def test_init_costMatrix(self,CostMatrixMock):
        # Arrange and Act
        sdtw = SDTW(None, None)
        # Assert
        self.assertTrue(np.array_equal(np.array([[50],[50],[50]]), sdtw.LocalCostMatrix))
        self.assertTrue(np.array_equal(np.array([[100],[100],[100]]), sdtw.AccumulatedCostMatrix))
        self.assertTrue(CostMatrixMock.called_once())
    
    @mock.patch("pybasicdtw.core.Core.CostMatrix", return_value=[np.array([[50],[50],[50]]), np.array([[100],[100],[100]])])
    def test_getEndCost(self, CostMatrixMock):
        # Arrange
        sdtw = SDTW(None,None)
        # Act
        endCost = sdtw.GetEndCost(np.array([[0,0]]))
        # Assert
        self.assertTrue(endCost == 100)
        self.assertTrue(CostMatrixMock.called_once())

    @mock.patch("pybasicdtw.core.Core.CostMatrix", return_value=[np.array([[50],[50],[50]]), np.array([[100],[100],[100]])])
    def test_findMatch_typeError(self, CostMatrixMock):
        # Arrange
        sdtw = SDTW(None, None)
        # Act and Assert
        with self.assertRaises(TypeError):
            sdtw.FindMatch(neighbourExclusion="STR")

    @mock.patch("pybasicdtw.core.Core.CostMatrix", return_value=[np.array([[50],[50],[50]]), np.array([[100],[100],[100]])])
    def test_findMatch_valueErrorClass(self, CostMatrixMock):
        # Arrange
        sdtw = SDTW(None, None)
        def TestFunc(): return None
        # Act and Assert
        with self.assertRaises(ValueError) as ve:
            sdtw.FindMatch(TestFunc)
        self.assertEqual("NeighbourExclusion must be a method from the NeighbourExclusion class.", str(ve.exception))

    @mock.patch("pybasicdtw.core.Core.CostMatrix", return_value=[np.array([[50],[50],[50]]), np.array([[100],[100],[100]])])
    def test_findMatch_valueErrorMethod(self, CostMatrixMock):
        # Arrange
        sdtw = SDTW(None, None)
        # Act and Assert
        with self.assertRaises(ValueError) as ve:
            sdtw.FindMatch()
        self.assertEqual("For NeighbourhoodExclusion.Distance please provide a distance using the keyword arg 'distance'.", str(ve.exception))

    @mock.patch("pybasicdtw.core.Core.CostMatrix", return_value=[np.array([[8,8,8,8,8], [6,6,6,6,6], [4,4,4,4,4]], dtype="float"), np.array([[8,8,8,8,8], [14,14,14,14,14], [18,18,18,18,18]], dtype="float")])
    @mock.patch("pybasicdtw.core.Core.WarpingPath", return_value=[np.array([(2,4),(1,3),(0,2)]), 18])
    @mock.patch("pybasicdtw.sdtw.NeighbourExclusion.Match", return_value=[None])
    def test_findmatch_overlap(self, CostMatrixMock, WarpingPathMock, MatchMock):
        # Arrange
        sdtw = SDTW(None, None)
        # Act
        path, totalCost = sdtw.FindMatch(distance=1)
        # Assert
        self.assertTrue(np.array_equal(np.array([(2,4),(1,3),(0,2)]), path))
        self.assertEqual(totalCost, 18)

    @mock.patch("pybasicdtw.core.Core.CostMatrix", return_value=[np.array([[8,8,8,8,8], [6,6,6,6,6], [4,4,4,4,4]], dtype="float"), np.array([[8,8,8,8,8], [14,14,14,14,14], [18,18,18,18,18]], dtype="float")])
    @mock.patch("pybasicdtw.core.Core.WarpingPath", return_value=[np.array([(2,4),(1,3),(0,2)]), 18])
    @mock.patch("pybasicdtw.sdtw.NeighbourExclusion.Match", return_value=[None])
    def test_findmatch_matches(self, CostMatrixMock, WarpingPathMock, MatchMock):
        # Arrange
        sdtw = SDTW(None, None)
        # Act
        _, _ = sdtw.FindMatch(distance=1)
        _, _ = sdtw.FindMatch(distance=1)
        # Assert
        self.assertTrue(np.array_equal(np.array([(2,4),(1,3),(0,2)]), sdtw.Matches[0][0]))
        self.assertEqual(sdtw.Matches[0][1], 18)
        self.assertTrue(np.array_equal(np.array([(2,4),(1,3),(0,2)]), sdtw.Matches[1][0]))
        self.assertEqual(sdtw.Matches[1][1], 18)