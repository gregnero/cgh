import numpy as np
import matplotlib.pyplot as plt

def idMax(array, verb, disp):

    '''
    Description: Identifies the maximum value and its location in a 2D array 

    Note: Input array should have an expected maximum
    
    Parameters: 
      array : (2D numpy array) a single-channel 2D numpy array
      verb  : (boolean) choose to print max value and location to screen
      disp  : (boolean) choose to display mask with cross at max location
    
    returns: list that contains max value and location
    '''
    
    #Get array shape
    M = np.shape(array)[0]
    N = np.shape(array)[1]


    #Get the peak value and x,y location
    detected_peak_value = np.max(array)
    detected_peak_x = np.unravel_index(np.argmax(array), array.shape)[1]
    detected_peak_y = np.unravel_index(np.argmax(array), array.shape)[0]
    detected_peak_location = [detected_peak_x, detected_peak_y]

    #Place info into storage
    info = []
    info.append(detected_peak_value)
    info.append(detected_peak_location)
    
    if (verb == True):

        print("===== idMax Verbose Output =====")
        print("Array Size: (" + str(M) + ", " + str(N) + ")")
        print("Max Value: " + str(detected_peak_value))
        print("Max Value Location: (" + str(detected_peak_x) + ", " + str(detected_peak_y) + ")")

    if (disp == True):

        mask = np.zeros((M,N))

        fig, (ax1, ax2) = plt.subplots(1,2)
        
        cross_half_width = int(M * 0.07)
        cross_half_height = int(N * 0.07)

        L1 = detected_peak_x - cross_half_width + 1
        L2 = detected_peak_x + cross_half_width

        L3 = detected_peak_y - cross_half_height + 1
        L4 = detected_peak_y + cross_half_height

        mask[int(detected_peak_y-1):int(detected_peak_y+1), int(L1):int(L2)] = 1
        mask[int(L3):int(L4), int(detected_peak_x-1):int(detected_peak_x+1)] = 1

        ax1.set_title("Input Array")
        ax1.imshow(array, cmap = 'gray')

        ax2.set_title("idMax Mask")
        ax2.imshow(mask, cmap = 'gray')

    return info
