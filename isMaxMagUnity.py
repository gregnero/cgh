import numpy as np

def isMaxMagUnity(real_part, imag_part):

    '''
    Description: Checks if the maximum magnitude of the arguments is one

    Note: Consider this problem on the complex plane ... no vector magnitude larger than one
    Note: accuracy to account for computational error is included in the check

    Parameters: 
      real_part : (numpy array) the array of real parts of the complex object 
      imag_part : (numpy array) the array of imag parts of the complex object

    returns: returns nothing if passed, if not returns -1
    '''

    magnitude = np.sqrt(np.square(real_part) + np.square(imag_part))

    max_magnitude = np.max(magnitude)

    accuracy = 0.05
    threshold_upper = 1 + accuracy
    threshold_lower = 1 - accuracy

    if (max_magnitude >= threshold_upper or max_magnitude <= threshold_lower):

        print("The maximum magnitude of the args is not unity!")
        print("Maximum magnitude = " + str(max_magnitude))
        return -1
