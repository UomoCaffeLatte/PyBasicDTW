# PyBasicDTW
A readable dynamic time warping (dtw) library that performs classical dtw and subsequence dtw.

### Features
- Classical DTW: Finds the similarity between two sequences
- Subsequence DTW: Finds the most similar subsequence in one sequences that matches the other sequence.
- Multidimension sequences supported
- Customisable dimension weighting
- Customisable step pattern and weights
- Customisable distance metrics
- Various plotting (FUTURE WORK)

## Dependencies
- Numpy
- Matplotlib

# Quick Start

First install pybasicdtw and dependencies using pip

```
 pip install pybasicdtw
```

Create a new python script and import the pybasicdtw modules. The following steps will be split between classical and subsequence dtw.

### 1. Classical DTW
Import the required classes.
``` python
    from pybasicdtw import DTW, DistanceMetric, StepPattern
```
**Distance Metric** argument can be of any callable type with two numpy array inputs and one numpy array output. Typically this is a lambda function, which you can create yourself or use ones provided using the DistanceMetric class.

For example, to use the euclidean distance metric you could either pass in:
``` python
    DistanceMetric.EUCLIDEAN
```
or
``` python
    lambda x,y: np.square(x-y)
```

**Step Pattern** argument can be of any numpy array that describes the step pattern to calculate the cost matrix and optimal warping path. Each element in this array describes a step backwards in terms of (row, column).

For example:
``` python
    stepPattern = np.array([(1,1),(1,0),(0,1)])
    # (1,1) take one step backward on the row and one step backwards on the column
    # (1,0) take one step backward on the row
    # (0,00 take one step backward on the column
    # Visual Explanation, imagine you are looking down on a 2D matrix.
    #  (1,0) -------
    #             / |
    #            /  |
    #           /   |
    #     (1,1)/    |
    #               (0,1)
```
This is the equivalent of using the provided stepPattern using the StepPattern class.
``` python
    StepPattern.CLASSIC
```

Now we wil create our own sequences to run through. These can be be multidimensional.
``` python
    import numpy as np
    # 1 Dimension, each element corresponds some value at time t.
    x = np.array([1,2,3])
    y = np.array([1,2,3,4,5,6,7,8])

    # 2 Dimension, each element corresponds to values of a dimension across time t.
    x2D = np.array([[1,2,3],[3,2,1]])
    y2D = np.array([[1,2,3,4,5,6,7,8],[8,7,6,5,4,3,2,1]])
```
The library requires these sequences to be of a specific format. Where each element describes the values of all dimensions at that time t. This can be easily done using the list comprehension.

``` python
    # For 1D array, we need to ensure each element is in its own array.
    x = np.array([[value] for value in range(x.shape[0])])
    # x results: array([[1],[2],[3]])
    y = np.array([[value] for value in range(y.shape[0])])

    # For 2D array, here we iterate through the each dimension and extract the values that correspond to that time t.
    x2D = np.array([x2D[:, t] for t in range(x2D.shape[1])])
    # x2D results: array([[1,3],[2,2],[3,1]])
    y2D = np.array([y2D[:, t] for t in range(y2D.shape[1])])
```
Now we can proceed to calculating the similarity between these two sequences.

The basic function call using default values:
``` python
    # We will be using the 2D array example from now onwards.
    # Default values:
    # DistanceMetric = Euclidean
    # StepPattern = Classic = np.array([(1,1),(1,0),(0,1)])
    # StepPatterWeights = np.array([1,1,1])
    # DimensionWeights = np.array([1,...])
    dtw = DTW(x,y)
```

Accessing properties of the similarity match
``` python
    # Accumulated cost matrix
    dtw.AccumulatedCostMatrixnp.ndarray # an n x m matrix where n = length of x and m = length of y.
    # Local cost matrix
    dtw.LocalCostMatrix:np.ndarray # an n x m matrix where n = length of x and m = length of y.
    # Match path describing the points of similarity between both sequences.
    # Each element of this path represents the index of the matched points, (x,y) is the order of the indices for sequence x and y.
    # NOTE: The path is in reverse order, where element at index 0 is the end point.
    dtw.MatchPath:np.ndarray # e.g. array([(3,3),(2,2),(1,1)])
    # The total local cost of the matched path
    dtw.TotalCost:float # e.g. 10.2

```

### 2. Subsequence DTW 
The steps to find subsequence similarity matches are similar to the Classical DTW with just some extra steps.


Import the required classes.
``` python
    from pybasicdtw import SDTW, DistanceMetric, StepPattern, NeighbourExclusion
```

We will be using the numpy array we generated from the Classical DTW example. If you are unsure what format the inputs need to be please refer to the instructions in the Classical DTW example.

``` python
    import numpy as np
    # 1 Dimension, each element corresponds some value at time t.
    x = np.array([1,2,3])
    y = np.array([1,2,3,4,5,6,7,8])
    # For 1D array, we need to ensure each element is in its own array.
    x = np.array([[value] for value in range(x.shape[0])])
    # x results: array([[1],[2],[3]])
    y = np.array([[value] for value in range(y.shape[0])])
    # y results: array([1],[2],[3],[4],[5],[6],[7],[8])
```

Firstly, we initalise sdtw which creates the cost matrices needed to find similar subsequences. The x argument is the sequence we are trying find, and the y argument is the sequence in which we are trying find subsequences of within it that best match the x sequence.
``` python
    # Optional arguments and their default values are explained in the Classical DTW example
    sdtw = SDTW(x,y)
```

Now we can find the first similar subsequence in sequence y. But before this, lets quickly go through the arguments of this function.

**Neighbour Exclusion** argument describes the method use to exclude neighbouring end points before the next match is found. All matches are created backwards, starting at the end point. There are two types of exclusion methods you can use, Distance and LocalMaximum based exclusion.

The Distance method excludes neighbouring points within a set distance of indices, this can be selected using the NeighbourExclusion class as follows, the distance can be set using the distance keyword argument when called the FindMatch function:
``` python
    NeighbourExclusion.Distance
    # how to specify distance
    sdtw.FindMatch(...., distance=10)
```


The LocalMaximum method excludes neighbouring points up to the next local maximum. This can be selected using the NeighbourExclusion class as follows@
``` python
    NeighbourExclusion.Distance
```

The **OverlapMatches** argument gives you the option to overlap subsequence matches. By default this is false, therefore no two subsequence matches can overlap.

The **InvertEndPointsSelection** argument specifies if there are two non-unique optimum end points which one to choose from. The default is True, hence it will always choose the end point with the largest index. Hence, False is the opposite.

``` python
    # returns a tuple (path, totalCost)
    path, totalCost = sdtw.FindMatch(NeighbourExclusion.Distance, distance=10)
```

Accessing properties of the sdtw class
``` python
    # Accumulated cost matrix
    dtw.AccumulatedCostMatrixnp.ndarray # an n x m matrix where n = length of x and m = length of y.
    # Local cost matrix
    dtw.LocalCostMatrix:np.ndarray # an n x m matrix where n = length of x and m = length of y.
    # gets all matches up to this current time as an ordered list of Tuples (path, totalCost)
    dtw.Matches
```

Methods of the sdtw class.
``` python
    # get the end points accumulated cost, the input argument is the match similarity path.
    sdtw.GetEndCost(path)
```