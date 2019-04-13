import numpy as np

def mean_2d_diff_size(a):
    """
    Takes 2d array with different sub-sizes and return correct mean
    
    Argument:
        a -- 2d numpy or not. E.g.: [[1,1,1],[2,2,2,2],[3,3,3,3,3]]
    Return:
        Mean of a, ignoring missing elements.
        
    Example:
        mean_2d_diff_size([[1,1,1],[2,2,2,2]) == [1.5, 1.5, 1.5, 2]
    """
    if type(a) == list:
        a = np.array(a)
    arr = np.ma.empty((a.shape[0],max([len(i) for i in a])))
    arr.mask = True
    for i, sub_a in enumerate(a):
        arr[i, :len(sub_a)] = sub_a
    return arr.mean(axis = 0)