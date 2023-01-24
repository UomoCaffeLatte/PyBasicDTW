from pybasicdtw import DTW

import unittest
import unittest.mock as mock

import numpy as np

class DTW_UnitTest(unittest.TestCase):

    def OneDSequenceSameLength(self, sdtw=False):
        x = np.array([[1],[2],[3]], dtype="float")
        y = np.array([[1],[2],[3]], dtype="float")
        LCost = np.array([[0,1,2],[1,0,1],[2,1,0]], dtype="float")
        ACost = np.array([[0,1,3],[1,0,1],[3,1,0]], dtype="float")
        if sdtw: ACost = np.array([[0,1,2],[1,0,1],[3,1,0]], dtype="float")
        return x,y, LCost, ACost

    def TwoDSequenceSameLength(self, sdtw=False):
        x = np.array([[1,1],[2,2],[3,3]], dtype="float")
        y = np.array([[1,1],[2,2],[3,3]], dtype="float")
        LCost = np.array([[0,2,4],[2,0,2],[4,2,0]], dtype="float")
        ACost = np.array([[0,2,6],[2,0,2],[6,2,0]], dtype="float")
        if sdtw: ACost = np.array([[0,2,4],[2,0,2],[6,2,0]], dtype="float")
        return x,y, LCost, ACost

    def OneDSequences(self, sdtw=False, reverse=False):
        x = np.array([[1],[2],[3]], dtype="float")
        y = np.array([[5],[5],[5],[5],[5]], dtype="float")
        LCost = np.array([[4,4,4,4,4], [3,3,3,3,3], [2,2,2,2,2]], dtype="float")
        ACost = np.array([[4,8,12,16,20], [7,7,10,13,16], [9,9,9,11,13]], dtype="float")
        if sdtw: ACost = np.array([[4,4,4,4,4], [7,7,7,7,7], [9,9,9,9,9]], dtype="float")
        if sdtw and reverse: ACost = np.array([[4,3,2],[8,6,4],[12,9,6],[16,12,8],[20,15,10]], dtype="float")
        return x,y,LCost,ACost

    def TwoDSequences(self, sdtw=False, reverse=False):
        x = np.array([[1,1],[2,2],[3,3]], dtype="float")
        y = np.array([[5,5],[5,5],[5,5],[5,5],[5,5]], dtype="float")
        LCost = np.array([[8,8,8,8,8], [6,6,6,6,6], [4,4,4,4,4]], dtype="float")
        ACost = np.array([[8,16,24,32,40],[14,14,20,26,32],[18,18,18,22,26]], dtype="float")
        if sdtw: ACost = np.array([[8,8,8,8,8], [14,14,14,14,14], [18,18,18,18,18]], dtype="float")
        if sdtw and reverse: ACost = np.array([[8,6,4],[16,12,8],[24,18,12],[32,24,16],[40,30,20]], dtype="float")
        return x,y, LCost, ACost

    # INIT
    @mock.patch("pybasicdtw.dtw.DTW._DTW__CostMatrix", return_value=[])
    def test_init_defaultStepWeights(self, mockCostMatrix):
        # Arrange and Act
        dtw = DTW(x=np.array([[0],[0]]),y=np.array([[0],[0]]))
        # Assert
        self.assertTrue(np.array_equal(np.array([1,1,1]),dtw.stepWeights))

    @mock.patch("pybasicdtw.dtw.DTW._DTW__CostMatrix", return_value=[])
    def test_init_customStepWeights(self, mockCostMatrix):
        # Arrange and Act
        dtw = DTW(x=np.array([[0],[0]]),y=np.array([[0],[0]]), stepWeights=np.array([2,3,4]))
        # Assert
        self.assertTrue(np.array_equal(np.array([2,3,4]), dtw.stepWeights))
    
    @mock.patch("pybasicdtw.dtw.DTW._DTW__CostMatrix", return_value=[])
    def test_init_customStepWeights_failStepPatternMatch(self, mockCostMatrix):
        # Arrange Act Assert
        with self.assertRaises(ValueError) as ve:
            dtw = DTW(x=np.array([[0],[0]]),y=np.array([[0],[0]]),stepWeights=np.array([1]))
            self.assertEqual(str(ve.exception),"StepWeights do not match StepPattern, StepWeights:1 != StepPatten:3")
            
    @mock.patch("pybasicdtw.dtw.DTW._DTW__CostMatrix", return_value=[])
    def test_init_defaultDimWeights(self, mockCostMatrix):
        # Arrange and Act
        dtw = DTW(x=np.array([[0],[0]]),y=np.array([[0],[0]]))
        # Assert
        self.assertTrue(np.array_equal(np.array([1]),dtw.dimWeights))

    @mock.patch("pybasicdtw.dtw.DTW._DTW__CostMatrix", return_value=[])
    def test_init_customDimWeights(self, mockCostMatrix):
        # Arrange and Act
        dtw = DTW(x=np.array([[0],[0]]),y=np.array([[0],[0]]), dimensionWeights=np.array([2]))
        # Assert
        self.assertTrue(np.array_equal(np.array([2]), dtw.dimWeights))
    
    @mock.patch("pybasicdtw.dtw.DTW._DTW__CostMatrix", return_value=[])
    def test_init_customDimWeights_failDimMatch(self, mockCostMatrix):
        # Arrange Act Assert
        with self.assertRaises(ValueError) as ve:
            dtw = DTW(x=np.array([[0],[0]]),y=np.array([[0],[0]]),dimensionWeights=np.array([1,2]))
            self.assertEqual(str(ve.exception),"DimWeights length do not match number of dimensions.")

    @mock.patch("pybasicdtw.dtw.DTW._DTW__CostMatrix", return_value=[])
    def test_init_distanceMetric(self, mockCostMatrix):
        # Arrange # Act
        dtw1 = DTW(x=np.array([[0],[0]]),y=np.array([[0],[0]]),distanceMetric=lambda x,y: np.square(x-y))
        dtw2 = DTW(x=np.array([[0],[0]]),y=np.array([[0],[0]]),distanceMetric=lambda x,y: np.abs(x-y))
        # Assert
        self.assertEqual((lambda x,y: np.square(x-y)).__code__.co_code, dtw1.distanceMetric.__code__.co_code)  
        self.assertEqual((lambda x,y: np.abs(x-y)).__code__.co_code, dtw2.distanceMetric.__code__.co_code)  

    @mock.patch("pybasicdtw.dtw.DTW._DTW__CostMatrix", return_value=[])
    def test_init_distanceMetric_failType(self, mockCostMatrix):
        # Arrange
        FAIL = "FAIL"
        # Act and Assert
        with self.assertRaises(TypeError)  as te:
            _ = DTW(x=np.array([[0],[0]]),y=np.array([[0],[0]]), distanceMetric=FAIL)
            self.assertEqual(str(te.exception), "DistanceMetric must be a Callable type.")

    @mock.patch("pybasicdtw.dtw.DTW._DTW__CostMatrix", return_value=[])
    def test_init_distanceMetric_failArgs(self, mockCostMatrix):
        # Arrange # Act
        FAIL = lambda x: x
        # Act and Assert
        with self.assertRaises(ValueError)  as ve:
            _ = DTW(x=np.array([[0],[0]]),y=np.array([[0],[0]]), distanceMetric=FAIL)
            self.assertEqual(str(ve.exception), "DistanceMetric Callable must have two inputs.")

    @mock.patch("pybasicdtw.dtw.DTW._DTW__CostMatrix", return_value=[])
    def test_init_stepPattern(self,mockCostMatrix):
        # Arrange # Act
        dtw = DTW(x=np.array([[0],[0]]),y=np.array([[0],[0]]))
        # Assert
        self.assertTrue(np.array_equal(np.array([(1,1),(1,0),(0,1)]), dtw.stepPattern))

    ## COST MATRIX
    @mock.patch("pybasicdtw.dtw.DTW._DTW__LexiMin", return_value=[])
    def test_costMatrix_xy_fail(self, mockLexiMinIndex):
        # Arrange
        x = np.array([[1,2,3],[1,2,3]])
        y = np.array([[1,2],[1,2],[1,2,]])
        # Act # Assert
        with self.assertRaises(ValueError):
            DTW(x = x, y = y)
        mockLexiMinIndex.assert_not_called()

    @mock.patch("pybasicdtw.dtw.DTW._DTW__LexiMin", return_value=[0])
    def test_costMatrix_defaultDimWeights(self, mockLexiMinIndex):
         # Arrange
        x = np.array([[1,2,3],[1,2,3]])
        y = np.array([[1,2,3],[1,2,3],[1,2,3]])
        # Act
        dtw = DTW(x,y)
        # Assert
        self.assertTrue(np.array_equal(np.array([1,1,1]), dtw.dimWeights))
        mockLexiMinIndex.assert_called()

    @mock.patch("pybasicdtw.dtw.DTW._DTW__LexiMin", return_value=[])
    def test_costMatrix_customDimWeights_fail(self, mockLexiMinIndex):
        # Arrange
        x = np.array([[1,2,3],[1,2,3]])
        y = np.array([[1,2,3],[1,2,3],[1,2,3]])
        dimWeights = np.array([1,1,1,1])
        # Act # Assert
        with self.assertRaises(ValueError):
            dtw = DTW(x,y, dimensionWeights=dimWeights)
        mockLexiMinIndex.assert_not_called()

    ## for all local and accumulated cost matrix tests ensure to test both sdtw and dtw version
    def test_costMatrix_1dDiffLength(self):
        # arrange
        x1, y1, LCostTruth, ACostTruth = self.OneDSequences()
        x2, y2, LSCostTruth, ASCostTruth = self.OneDSequences(sdtw=True)
        # Act
        dtw1 = DTW(x1,y1, distanceMetric=lambda x,y:np.abs(x-y))
        dtw2 = DTW(x2,y2, sdtw=True, distanceMetric=lambda x,y:np.abs(x-y))
        # Assert
        self.assertTrue(np.array_equal(LCostTruth, dtw1.localCost))
        self.assertTrue(np.array_equal(ACostTruth, dtw1.accumulatedCost))
        self.assertTrue(np.array_equal(LSCostTruth, dtw2.localCost))
        self.assertTrue(np.array_equal(ASCostTruth, dtw2.accumulatedCost))

    def test_costMatrix_2dDiffLength(self):
        # arrange
        x1, y1, LCostTruth, ACostTruth = self.TwoDSequences()
        x2, y2, LSCostTruth, ASCostTruth = self.TwoDSequences(sdtw=True)
        # Act
        dtw1 = DTW(x1,y1, distanceMetric=lambda x,y:np.abs(x-y))
        dtw2 = DTW(x2,y2, sdtw=True, distanceMetric=lambda x,y:np.abs(x-y))
        # Assert
        self.assertTrue(np.array_equal(LCostTruth, dtw1.localCost))
        self.assertTrue(np.array_equal(ACostTruth, dtw1.accumulatedCost))
        self.assertTrue(np.array_equal(LSCostTruth, dtw2.localCost))
        self.assertTrue(np.array_equal(ASCostTruth, dtw2.accumulatedCost))

    def test_costMatrix_1dReverseLength(self):
        # arrange
        x1, y1, LCostTruth, ACostTruth = self.OneDSequences()
        x2, y2, LSCostTruth, ASCostTruth = self.OneDSequences(sdtw=True, reverse=True)
        # Act
        dtw1 = DTW(y1,x1, distanceMetric=lambda x,y:np.abs(x-y))
        dtw2 = DTW(y2,x2, sdtw=True, distanceMetric=lambda x,y:np.abs(x-y))
        # Assert
        self.assertTrue(np.array_equal(LCostTruth.transpose(), dtw1.localCost))
        self.assertTrue(np.array_equal(ACostTruth.transpose(), dtw1.accumulatedCost))
        self.assertTrue(np.array_equal(LSCostTruth.transpose(), dtw2.localCost))
        # ASCost wont be symmetric
        self.assertTrue(np.array_equal(ASCostTruth, dtw2.accumulatedCost))

    def test_costMatrix_2dReverseLengh(self):
        # arrange
        x1, y1, LCostTruth, ACostTruth = self.TwoDSequences()
        x2, y2, LSCostTruth, ASCostTruth = self.TwoDSequences(sdtw=True, reverse=True)
        # Act
        dtw1 = DTW(y1,x1, distanceMetric=lambda x,y:np.abs(x-y))
        dtw2 = DTW(y2,x2, sdtw=True, distanceMetric=lambda x,y:np.abs(x-y))
        # Assert
        self.assertTrue(np.array_equal(LCostTruth.transpose(), dtw1.localCost))
        self.assertTrue(np.array_equal(ACostTruth.transpose(), dtw1.accumulatedCost))
        self.assertTrue(np.array_equal(LSCostTruth.transpose(), dtw2.localCost))
        # ASCost wont be symmetric
        self.assertTrue(np.array_equal(ASCostTruth, dtw2.accumulatedCost))

    def test_costMatrix_1dSameLength(self):
        # arrange
        x1, y1, LCostTruth, ACostTruth = self.OneDSequenceSameLength()
        x2, y2, LSCostTruth, ASCostTruth = self.OneDSequenceSameLength(sdtw=True)
        # Act
        dtw1 = DTW(y1,x1, distanceMetric=lambda x,y:np.abs(x-y))
        dtw2 = DTW(y2,x2, sdtw=True, distanceMetric=lambda x,y:np.abs(x-y))
        # Assert
        self.assertTrue(np.array_equal(LCostTruth, dtw1.localCost))
        self.assertTrue(np.array_equal(ACostTruth, dtw1.accumulatedCost))
        self.assertTrue(np.array_equal(LSCostTruth, dtw2.localCost))
        self.assertTrue(np.array_equal(ASCostTruth, dtw2.accumulatedCost))

    def test_costMatrix_2dSameLength(self):
        # arrange
        x1, y1, LCostTruth, ACostTruth = self.TwoDSequenceSameLength()
        x2, y2, LSCostTruth, ASCostTruth = self.TwoDSequenceSameLength(sdtw=True)
        # Act
        dtw1 = DTW(y1,x1, distanceMetric=lambda x,y:np.abs(x-y))
        dtw2 = DTW(y2,x2, sdtw=True, distanceMetric=lambda x,y:np.abs(x-y))
        # Assert
        self.assertTrue(np.array_equal(LCostTruth, dtw1.localCost))
        self.assertTrue(np.array_equal(ACostTruth, dtw1.accumulatedCost))
        self.assertTrue(np.array_equal(LSCostTruth, dtw2.localCost))
        self.assertTrue(np.array_equal(ASCostTruth, dtw2.accumulatedCost))
        
    ## OPTIMAL WARPING PATH
    @mock.patch("pybasicdtw.dtw.DTW._DTW__CostMatrix", return_value=[])
    def test_optimalWarpingPath_1DdtwEndIndex(self, costMatrix):
        # Arrange
        _,_, LCost, ACost = self.OneDSequences()
        # Act
        dtw = DTW(x=np.array([[0],[0]]),y=np.array([[0],[0]]))
        dtw._DTW__aCost = ACost
        dtw._DTW__lCost = LCost
        path, totalCost = dtw.WarpingPath()
        # Assert
        self.assertTrue(np.array_equal(path, np.array([(2,4), (2,3), (2,2), (1, 1), (0, 0)])))
        self.assertEqual(totalCost, 13)

    @mock.patch("pybasicdtw.dtw.DTW._DTW__CostMatrix", return_value=[])
    def test_optimalWarpingPath_1DsdtwEndIndex(self, mockCostMatrix):
        # Arrange
        _,_, LCost, ACost = self.OneDSequences(sdtw=True)
        # Act
        dtw = DTW(x=np.array([[0],[0]]),y=np.array([[0],[0]]), sdtw=True)
        dtw._DTW__aCost = ACost
        dtw._DTW__lCost = LCost
        path1, totalCost1 = dtw.WarpingPath(endIndex=(2,2))
        path2, totalCost2 = dtw.WarpingPath(endIndex=(2,1))
        # Assert
        self.assertTrue(np.array_equal(np.array([(2,2),(1,1),(0,0)]), path1))
        self.assertEqual(totalCost1, 9)
        self.assertTrue(np.array_equal(np.array([(2,1),(1,0),(0,0)]), path2))
        self.assertEqual(totalCost2, 9)

    @mock.patch("pybasicdtw.dtw.DTW._DTW__CostMatrix", return_value=[])
    def test_optimalWarpingPath_1DReverseSdtwEndIndex(self, mockCostMatrix):
        # Arrange
        _,_, LCost, ACost = self.OneDSequences(sdtw=True, reverse=True)
        # Act
        dtw = DTW(x=np.array([[0],[0]]),y=np.array([[0],[0]]), sdtw=True)
        dtw._DTW__aCost = ACost
        dtw._DTW__lCost = LCost.transpose()
        path1, totalCost1 = dtw.WarpingPath(endIndex=(4,1))
        path2, totalCost2 = dtw.WarpingPath(endIndex=(4,0))
        # Assert
        self.assertTrue(np.array_equal(np.array([(4,1),(3,1),(2,1),(1,1),(0,1)]), path1))
        self.assertEqual(totalCost1, 15)
        self.assertTrue(np.array_equal(np.array([(4,0),(3,0),(2,0),(1,0),(0,0)]), path2))
        self.assertEqual(totalCost2, 20)

    @mock.patch("pybasicdtw.dtw.DTW._DTW__CostMatrix", return_value=[])
    def test_optimaWarpingPath_2DdtwEndIndex(self, mockCostMatrix):
        # Arrange
        _,_, LCost, ACost = self.TwoDSequences()
        # Act
        dtw = DTW(x=np.array([[0],[0]]),y=np.array([[0],[0]]))
        dtw._DTW__aCost = ACost
        dtw._DTW__lCost = LCost
        path, totalCost = dtw.WarpingPath()
        # Assert
        self.assertTrue(np.array_equal(path, np.array([(2,4), (2,3), (2,2), (1, 1), (0, 0)])))
        self.assertEqual(totalCost, 26)

    @mock.patch("pybasicdtw.dtw.DTW._DTW__CostMatrix", return_value=[])
    def test_optimalWarpingPath_2DsdtwEndIndex(self, mockCostMatrix):
        # Arrange
        _,_, LCost, ACost = self.TwoDSequences(sdtw=True)
        # Act
        dtw = DTW(x=np.array([[0],[0]]),y=np.array([[0],[0]]), sdtw=True)
        dtw._DTW__aCost = ACost
        dtw._DTW__lCost = LCost
        path1, totalCost1 = dtw.WarpingPath(endIndex=(2,2))
        path2, totalCost2= dtw.WarpingPath(endIndex=(2,1))
        # Assert
        self.assertTrue(np.array_equal(np.array([(2,2),(1,1),(0,0)]), path1))
        self.assertEqual(totalCost1, 18)
        self.assertTrue(np.array_equal(np.array([(2,1),(1,0),(0,0)]), path2))
        self.assertEqual(totalCost2, 18)

    # LEXIARGMIN
    @mock.patch("pybasicdtw.dtw.DTW._DTW__CostMatrix", return_value=[])
    def test_LexiArgMin_Equal(self, mockCostMatrix):
        # arrange
        items = np.array([100,100,100])
        # act
        minIndex = DTW(x=np.array([[0],[0]]),y=np.array([[0],[0]]))._DTW__LexiMin(items)
        # assaert
        self.assertEqual(minIndex, 0)

    @mock.patch("pybasicdtw.dtw.DTW._DTW__CostMatrix", return_value=[])
    def test_LexiArgMin_NonEqual(self, mockCostMatrix):
        # arrange
        items = np.array([50,20,150])
        # act
        minIndex = DTW(x=np.array([[0],[0]]),y=np.array([[0],[0]]))._DTW__LexiMin(items)
        # assaert
        self.assertEqual(minIndex, 1)

    @mock.patch("pybasicdtw.dtw.DTW._DTW__CostMatrix", return_value=[])
    def test_LexiArgMin_EqualInverted(self, mockCostMatrix):
        # arrange
        items = np.array([100,100,100])
        # act
        minIndex = DTW(x=np.array([[0],[0]]),y=np.array([[0],[0]]))._DTW__LexiMin(items, invert=True)
        # assaert
        self.assertEqual(minIndex, 2)

    @mock.patch("pybasicdtw.dtw.DTW._DTW__CostMatrix", return_value=[])
    def test_LexiArgMin_NonEqualInverted(self, mockCostMatrix):
        # arrange
        items = np.array([50,20,150])
        # act
        minIndex = DTW(x=np.array([[0],[0]]),y=np.array([[0],[0]]))._DTW__LexiMin(items, invert=True)
        # assaert
        self.assertEqual(minIndex, 1)
    
    
    # REALCASE TESTS
    def RealOneDSequenceSameLength(self):
        x = np.array([[1],[2],[3]], dtype="float")
        y = np.array([[1],[2],[3]], dtype="float")
        LCost = np.array([[0,1,2],[1,0,1],[2,1,0]], dtype="float")
        ACost = np.array([[0,1,3],[1,0,1],[3,1,0]], dtype="float")
        return x,y, LCost, ACost

    def RealTwoDSequenceSameLength(self):
        x = np.array([[1,1],[2,2],[3,3]], dtype="float")
        y = np.array([[1,1],[2,2],[3,3]], dtype="float")
        LCost = np.array([[0,2,4],[2,0,2],[4,2,0]], dtype="float")
        ACost = np.array([[0,2,6],[2,0,2],[6,2,0]], dtype="float")
        return x,y, LCost, ACost

    def RealOneDSequences(self):
        x = np.array([[1],[2],[3]], dtype="float")
        y = np.array([[5],[5],[5],[5],[5]], dtype="float")
        LCost = np.array([[4,4,4,4,4], [3,3,3,3,3], [2,2,2,2,2]], dtype="float")
        ACost = np.array([[4,8,12,16,20], [7,7,10,13,16], [9,9,9,11,13]], dtype="float")
        return x,y,LCost,ACost

    def RealTwoDSequences(self):
        x = np.array([[1,1],[2,2],[3,3]], dtype="float")
        y = np.array([[5,5],[5,5],[5,5],[5,5],[5,5]], dtype="float")
        LCost = np.array([[8,8,8,8,8], [6,6,6,6,6], [4,4,4,4,4]], dtype="float")
        ACost = np.array([[8,16,24,32,40],[14,14,20,26,32],[18,18,18,22,26]], dtype="float")
        return x,y, LCost, ACost

    def test_DTW1D(self):
        # Arrange
        x,y,LCost,ACost = self.RealOneDSequences()
        # Act
        dtw = DTW(x,y, distanceMetric=lambda x,y: np.abs(x-y))
        path, totalCost = dtw.WarpingPath()
        # Assert
        self.assertEqual(totalCost, 13)
        self.assertTrue(np.array_equal(path, np.array([(2,4), (2,3), (2,2), (1, 1), (0, 0)])))
        self.assertTrue(np.array_equal(dtw.accumulatedCost, ACost))
        self.assertTrue(np.array_equal(dtw.localCost, LCost))

    def test_DTW1DSameLength(self):
        # Arrange
        x, y, LCost, ACost = self.RealOneDSequenceSameLength()
        # Act
        dtw = DTW(x,y, distanceMetric=lambda x,y: np.abs(x-y))
        path, totalCost = dtw.WarpingPath()
        # Assert
        self.assertEqual(totalCost, 0)
        self.assertTrue(np.array_equal(path, np.array([(2,2),(1,1),(0,0)])))
        self.assertTrue(np.array_equal(dtw.accumulatedCost, ACost))
        self.assertTrue(np.array_equal(dtw.localCost, LCost))

    def test_DTW1DReverse(self):
        # Arrange
        x,y,LCost, _ = self.RealOneDSequences()
        LCost = LCost.transpose()
        ACost = np.array([[4,7,9],[8,7,9],[12,10,9],[16,13,11],[20,16,13]])
        # Act
        dtw = DTW(y,x, distanceMetric=lambda x,y: np.abs(x-y))
        path, totalCost = dtw.WarpingPath()
        # Assert
        self.assertTrue(np.array_equal(dtw.accumulatedCost, ACost))
        self.assertTrue(np.array_equal(dtw.localCost, LCost))
        self.assertTrue(np.array_equal(path, np.array([[4,2],[3,2],[2,2],[1,1],[0,0]])))
        self.assertEqual(totalCost, 13)

    def test_DTW2D(self):
        # Arrange
        x,y,LCost, ACost = self.TwoDSequences()
        # Act
        dtw = DTW(x,y, distanceMetric=lambda x,y: np.abs(x-y))
        path, totalCost = dtw.WarpingPath()
        # Assert
        self.assertTrue(np.array_equal(dtw.accumulatedCost, ACost))
        self.assertTrue(np.array_equal(dtw.localCost, LCost))
        self.assertTrue(np.array_equal(path, np.array([[2,4],[2,3],[2,2],[1,1],[0,0]])))
        self.assertEqual(totalCost, 26)

    def test_DTW2DSameLength(self):
        # Arrange
        x,y,LCost, ACost = self.TwoDSequenceSameLength()
        # Act
        dtw = DTW(x,y, distanceMetric=lambda x,y: np.abs(x-y))
        path, totalCost = dtw.WarpingPath()
        # Assert
        self.assertTrue(np.array_equal(dtw.accumulatedCost, ACost))
        self.assertTrue(np.array_equal(dtw.localCost, LCost))
        self.assertTrue(np.array_equal(path, np.array([(2,2),(1,1),(0,0)])))
        self.assertEqual(totalCost, 0)

    def test_DTW2DReverse(self):
        # Arrange
        x,y,LCost,_ = self.TwoDSequences()
        LCost = LCost.transpose()
        ACost = np.array([[8,14,18],[16,14,18],[24,20,18],[32,26,22],[40,32,26]])
        # Act
        dtw = DTW(y,x, distanceMetric=lambda x,y: np.abs(x-y))
        path, totalCost = dtw.WarpingPath()
        # Assert
        self.assertTrue(np.array_equal(dtw.accumulatedCost, ACost))
        self.assertTrue(np.array_equal(dtw.localCost, LCost))
        self.assertTrue(np.array_equal(path, np.array([[4,2],[3,2],[2,2],[1,1],[0,0]])))
        self.assertEqual(totalCost, 26)
        

if __name__ == "__main__":
    unittest.main(verbosity=2)