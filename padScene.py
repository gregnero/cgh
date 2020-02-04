import numpy as np

def padScene(scene, x, y, available_pad):

    '''
    Description: 'Smart' pads a numpy array (containing an object in the scene at (x,y)) with zeros
    
    Note: Preserves relative object location in larger scene

    Parameters: 
      scene         : (numpy array) the scene that you want to pad
      x             : (int) the x-coordinate of the object in the scene
      y             : (int) the y-coordinate of the object in the scene
      available_pad : (int) total (linear) padding along one array dimension

    returns: list that contains:
      0 : the padded scene
      1 : the true x coordinate of object in new padded scene
      2 : the true y coordinate of object in new padded scene
    '''

    #Get dimension of input array 
    N = np.shape(scene)[0]

    #Get the left/right pad widths 

    #Get the left/right padding
    ratio_L = (x) / N
    ratio_R = 1 - ratio_L
    pad_L = int(ratio_L * available_pad)
    pad_R = int(ratio_R* available_pad)

    #Get the top/bottomg padding
    ratio_T = (y) / N
    ratio_B = 1 - ratio_T
    pad_T = int(ratio_T * available_pad)
    pad_B = int(ratio_B * available_pad)

    #Get true center of object in new scene (in case it's needed for stats/comparision...)
    scene_true_x = pad_L + x
    scene_true_y = pad_T + y

    #Construct the padded scene
    scene_padded = np.pad(scene, ((pad_T, pad_B),(pad_L,pad_R)), 'constant')

    return_list = []
    return_list.append(scene_padded)
    return_list.append(scene_true_x)
    return_list.append(scene_true_y)

    return return_list
