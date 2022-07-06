from PyBasicDTW import Core, DistanceMetric, StepPattern

import unittest
import unittest.mock as mock

import numpy as np

class Core_UnitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.core = Core(DistanceMetric.ABSOLUTE, StepPattern.CLASSIC)

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

    # def test_init_defaultStepWeights(self):
    #     pass

    # def test_init_customStepWeights(self):
    #     pass

    # def test_init_customStepWeights_fail(self):
    #     pass

    # def test_init_distanceMetric(self):
    #     pass
    
    # def test_init_stepPattern(self):
    #     pass
    
    # # COST MATRIX

    # def test_costMatrix_defaultDimWeights(self):
    #     pass

    # def test_costMatrix_customDimWeights(self):
    #     pass

    # def test_costMatrix_customDimWeights_fail(self):
    #     pass

    ## for all local and accumulated cost matrix tests ensure to test both sdtw and dtw version
    def test_costMatrix_1dDiffLength(self):
        # arrange
        x1, y1, LCostTruth, ACostTruth = self.OneDSequences()
        x2, y2, LSCostTruth, ASCostTruth = self.OneDSequences(sdtw=True)
        # Act
        LCost, ACost = self.core.CostMatrix(x1,y1)
        LSCost, ASCost = self.core.CostMatrix(x2,y2, sdtw=True) 
        # Assert
        self.assertTrue(np.array_equal(LCostTruth, LCost))
        self.assertTrue(np.array_equal(ACostTruth, ACost))
        self.assertTrue(np.array_equal(LSCostTruth, LSCost))
        self.assertTrue(np.array_equal(ASCostTruth, ASCost))


    def test_costMatrix_2dDiffLength(self):
        # arrange
        x1, y1, LCostTruth, ACostTruth = self.TwoDSequences()
        x2, y2, LSCostTruth, ASCostTruth = self.TwoDSequences(sdtw=True)
        # Act
        LCost, ACost = self.core.CostMatrix(x1,y1)
        LSCost, ASCost = self.core.CostMatrix(x2,y2, sdtw=True) 
        # Assert
        self.assertTrue(np.array_equal(LCostTruth, LCost))
        self.assertTrue(np.array_equal(ACostTruth, ACost))
        self.assertTrue(np.array_equal(LSCostTruth, LSCost))
        self.assertTrue(np.array_equal(ASCostTruth, ASCost))

    
    def test_costMatrix_1dReverseLength(self):
        # arrange
        x1, y1, LCostTruth, ACostTruth = self.OneDSequences()
        x2, y2, LSCostTruth, ASCostTruth = self.OneDSequences(sdtw=True, reverse=True)
        # Act
        LCost, ACost = self.core.CostMatrix(y1,x1)
        LSCost, ASCost = self.core.CostMatrix(y2,x2, sdtw=True) 
        # Assert
        self.assertTrue(np.array_equal(LCostTruth.transpose(), LCost))
        self.assertTrue(np.array_equal(ACostTruth.transpose(), ACost))
        self.assertTrue(np.array_equal(LSCostTruth.transpose(), LSCost))
        # ASCost wont be symmetric
        self.assertTrue(np.array_equal(ASCostTruth, ASCost))

    def test_costMatrix_2dReverseLengh(self):
        # arrange
        x1, y1, LCostTruth, ACostTruth = self.TwoDSequences()
        x2, y2, LSCostTruth, ASCostTruth = self.TwoDSequences(sdtw=True, reverse=True)
        # Act
        LCost, ACost = self.core.CostMatrix(y1,x1)
        LSCost, ASCost = self.core.CostMatrix(y2,x2, sdtw=True) 
        # Assert
        self.assertTrue(np.array_equal(LCostTruth.transpose(), LCost))
        self.assertTrue(np.array_equal(ACostTruth.transpose(), ACost))
        self.assertTrue(np.array_equal(LSCostTruth.transpose(), LSCost))
        # ASCost wont be symmetric
        self.assertTrue(np.array_equal(ASCostTruth, ASCost))


    # def test_costMatrix_1dSameLength(self):
    #     pass

    # def test_costMatrix_2dSameLength(self):
    #     pass

    # OPTIMAL WARPING PATH

    # LEXIARGMIN

    if __name__ == "__main__":
        unittest.main(verbosity=2)