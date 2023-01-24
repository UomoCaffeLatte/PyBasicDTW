from pybasicdtw.helper import Helper
import numpy as np
import unittest


class Helper_unitTests(unittest.TestCase):
    
    def test_InputFormat_1D(self):
        # Arrange
        input = np.array([1,2,3,4,5,6,7,8])
        trueOutput = np.array([[1],[2],[3],[4],[5],[6],[7],[8]])
        # Act
        result = Helper.InputFormat(input)
        # Assert
        self.assertTrue(np.array_equal(trueOutput, result))
        
    def test_InputFormat_2D(self):
        # Arrange
        input = np.array([[1,2,3,4,5,6,7,8],[8,7,6,5,4,3,2,1]])
        trueOutput = np.array([[1,8],[2,7],[3,6],[4,5],[5,4],[6,3],[7,2],[8,1]])
        # Act
        result = Helper.InputFormat(input)
        # Assert
        self.assertTrue(np.array_equal(trueOutput, result))

if __name__ == "__main__":
    unittest.main(verbosity=2)