import numpy as np

def isPhaseBehaved(real_part, imag_part):

    '''
    Description: Checks if the phase of the arguments falls in [-pi, pi] range

    Note: for more insight, read the np.arctan2 docs
    Note: makes sure you execute with >python3 ... >python handles very slightly differently?
    Note: accuracy to account for computational error is included in the check

    Parameters: 
      real_part : (numpy array) the array of real parts of the complex object 
      imag_part : (numpy array) the array of imag parts of the complex object

    returns: returns nothing if passed, if not returns -1
    '''

    phase = np.arctan2(imag_part, real_part)

    max_abs_phase = np.max(np.abs(phase))

    accuracy = 0.05
    threshold_upper = np.pi + accuracy
    threshold_lower = np.pi - accuracy

    if (max_abs_phase >= threshold_upper or max_abs_phase <= threshold_lower):

        print("Phase is misbehaved!")
        print("Max absoulte phase = " + str(max_abs_phase))
        return -1
