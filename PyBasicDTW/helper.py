import numpy as np

class Helper:
    
    @classmethod
    def PlotCostMatrix(cls, costMatrix:np.ndarray, path:np.ndarray=None, **kwargs) -> None:
        pass
    
    @classmethod
    def PlotFullCostMatrix(cls, x:np.ndarray, y:np.ndarray, **kwargs) -> None:
        pass
    
    @classmethod
    def InputFormat(cls, input:np.ndarray) -> np.ndarray:
        # reformate for 1D and 2D arrays. Where each element represents all dimensions at some time t.
        if input.ndim == 1: 
            input = np.array([input])
        return np.array([input[:, idx] for idx in range(input.shape[1])])