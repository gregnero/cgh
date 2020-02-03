import numpy as np

def isCongruent(array1, array2):

    '''
    Description: Checks if two input arrays have the same dimensions
    
    Parameters: 
      array1 : (numpy array) input array 1
      array2 : (numpy array) input array 2
    
    returns: (int) returns dim if congruent, returns -1 if not
    '''

    N1 = np.shape(array1)
    N2 = np.shape(array2)

    if (N1 != N2):

        print("Input arrays are not congruent!")
        print("N1 = " + str(N1))
        print("N2 = " + str(N2))
        return -1
    
    else:
        
        N = N1[0]
        return N
