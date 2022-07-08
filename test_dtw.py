from pybasicdtw import Core, DistanceMetric, StepPattern, DTW

import unittest
import unittest.mock as mock

import numpy as np

class DTW_UnitTests(unittest.TestCase):
    def OneDSequenceSameLength(self):
        x = np.array([[1],[2],[3]], dtype="float")
        y = np.array([[1],[2],[3]], dtype="float")
        LCost = np.array([[0,1,2],[1,0,1],[2,1,0]], dtype="float")
        ACost = np.array([[0,1,3],[1,0,1],[3,1,0]], dtype="float")
        return x,y, LCost, ACost

    def TwoDSequenceSameLength(self):
        x = np.array([[1,1],[2,2],[3,3]], dtype="float")
        y = np.array([[1,1],[2,2],[3,3]], dtype="float")
        LCost = np.array([[0,2,4],[2,0,2],[4,2,0]], dtype="float")
        ACost = np.array([[0,2,6],[2,0,2],[6,2,0]], dtype="float")
        return x,y, LCost, ACost

    def OneDSequences(self):
        x = np.array([[1],[2],[3]], dtype="float")
        y = np.array([[5],[5],[5],[5],[5]], dtype="float")
        LCost = np.array([[4,4,4,4,4], [3,3,3,3,3], [2,2,2,2,2]], dtype="float")
        ACost = np.array([[4,8,12,16,20], [7,7,10,13,16], [9,9,9,11,13]], dtype="float")
        return x,y,LCost,ACost

    def TwoDSequences(self):
        x = np.array([[1,1],[2,2],[3,3]], dtype="float")
        y = np.array([[5,5],[5,5],[5,5],[5,5],[5,5]], dtype="float")
        LCost = np.array([[8,8,8,8,8], [6,6,6,6,6], [4,4,4,4,4]], dtype="float")
        ACost = np.array([[8,16,24,32,40],[14,14,20,26,32],[18,18,18,22,26]], dtype="float")
        return x,y, LCost, ACost

    def test_DTW1D(self):
        # Arrange
        x,y,LCost,ACost = self.OneDSequences()
        # Act
        dtw = DTW(x,y, DistanceMetric.ABSOLUTE)
        # Assert
        self.assertEqual(dtw.TotalCost, 13)
        self.assertTrue(np.array_equal(dtw.MatchPath, np.array([(2,4), (2,3), (2,2), (1, 1), (0, 0)])))
        self.assertTrue(np.array_equal(dtw.accumulatedCost, ACost))
        self.assertTrue(np.array_equal(dtw.localCost, LCost))

    def test_DTW1DSameLength(self):
        # Arrange
        x, y, LCost, ACost = self.OneDSequenceSameLength()
        # Act
        dtw = DTW(x,y, DistanceMetric.ABSOLUTE)
        # Assert
        self.assertEqual(dtw.TotalCost, 0)
        self.assertTrue(np.array_equal(dtw.MatchPath, np.array([(2,2),(1,1),(0,0)])))
        self.assertTrue(np.array_equal(dtw.accumulatedCost, ACost))
        self.assertTrue(np.array_equal(dtw.localCost, LCost))

    def test_DTW1DReverse(self):
        # Arrange
        x,y,LCost, _ = self.OneDSequences()
        LCost = LCost.transpose()
        ACost = np.array([[4,7,9],[8,7,9],[12,10,9],[16,13,11],[20,16,13]])
        # Act
        dtw = DTW(y,x, DistanceMetric.ABSOLUTE)
        # Assert
        self.assertTrue(np.array_equal(dtw.accumulatedCost, ACost))
        self.assertTrue(np.array_equal(dtw.localCost, LCost))
        self.assertTrue(np.array_equal(dtw.MatchPath, np.array([[4,2],[3,2],[2,2],[1,1],[0,0]])))
        self.assertEqual(dtw.TotalCost, 13)

    def test_DTW2D(self):
        # Arrange
        x,y,LCost, ACost = self.TwoDSequences()
        # Act
        dtw = DTW(x,y, DistanceMetric.ABSOLUTE)
        # Assert
        self.assertTrue(np.array_equal(dtw.accumulatedCost, ACost))
        self.assertTrue(np.array_equal(dtw.localCost, LCost))
        self.assertTrue(np.array_equal(dtw.MatchPath, np.array([[2,4],[2,3],[2,2],[1,1],[0,0]])))
        self.assertEqual(dtw.TotalCost, 26)

    def test_DTW2DSameLength(self):
        # Arrange
        x,y,LCost, ACost = self.TwoDSequenceSameLength()
        # Act
        dtw = DTW(x,y, DistanceMetric.ABSOLUTE)
        # Assert
        self.assertTrue(np.array_equal(dtw.accumulatedCost, ACost))
        self.assertTrue(np.array_equal(dtw.localCost, LCost))
        self.assertTrue(np.array_equal(dtw.MatchPath, np.array([(2,2),(1,1),(0,0)])))
        self.assertEqual(dtw.TotalCost, 0)

    def test_DTW2DReverse(self):
        # Arrange
        x,y,LCost,_ = self.TwoDSequences()
        LCost = LCost.transpose()
        ACost = np.array([[8,14,18],[16,14,18],[24,20,18],[32,26,22],[40,32,26]])
        # Act
        dtw = DTW(y,x, DistanceMetric.ABSOLUTE)
        # Assert
        self.assertTrue(np.array_equal(dtw.accumulatedCost, ACost))
        self.assertTrue(np.array_equal(dtw.localCost, LCost))
        self.assertTrue(np.array_equal(dtw.MatchPath, np.array([[4,2],[3,2],[2,2],[1,1],[0,0]])))
        self.assertEqual(dtw.TotalCost, 26)
        