import numpy as np

def makE(x_c, y_c, N, p, gradient, orientation):
    
    '''
    Author: Gregory M. Nero

    Description: 'Draws' the letter 'E' in an NxN  numpy array centered at specified coordinates
    
    Note: This function dynamically scales the E based on a chosen N and petite metric
    Note: The larger petite metric, the smaller the E and vice versa
    Note: Requires some boundary awareness from the user

    Parameters: 
      x_c         : (int) x center coordinate
      y_c         : (int) y center coordinate
      N           : (int) size of square image
      p           : (int) petite metric 
      gradient    : (int) gradient option #TODO: add option for direction / orientation of gradient?
                  : 0 no gradient
                  : 1 horizontal gradient
                  : 2 vertical gradient
      orientation : (int) choose which way the gradient goes
                  : 0 smallest on left/top
                  : 1 smallest on right/bottom

    returns: NxN numpy array with a letter 'E' "drawn" centered at the desired coordinate
    '''
    
    #Instantiate a blank 'canvas' to draw the E on
    canvas = np.zeros((N,N))
    
    #Set bar petite metric
    petite = p
    
    #Scale the width of the E to the array size desired
    width = int( N / petite )
    
    #This part will be sort of confusing... set the E boundaries
    B_1_a = int(0 + np.abs((2.5 * width)-y_c))
    B_1_b = int(width + np.abs((2.5 * width)-y_c))
    B_2_a = int(0 + np.abs((1.5 * width)-x_c))
    B_2_b = int((3 * width) + np.abs((1.5 * width)-x_c))
    
    B_3_a = int(width + np.abs((2.5 * width)-y_c))
    B_3_b = int((5 * width) + np.abs((2.5 * width)-y_c))
    B_4_a = int(0 + np.abs((1.5 * width)-x_c))
    B_4_b = int(width + np.abs((1.5 * width)-x_c))
    
    B_5_a = int((4 * width) + np.abs((2.5 * width)-y_c))
    B_5_b = int((5 * width) + np.abs((2.5 * width)-y_c))
    B_6_a = int(width + np.abs((1.5 * width)-x_c))
    B_6_b = int((3 * width) + np.abs((1.5 * width)-x_c))
    
    B_7_a = int((2 * width) + np.abs((2.5 * width)-y_c))
    B_7_b = int((3 * width) + np.abs((2.5 * width)-y_c))
    B_8_a = int(width + np.abs((1.5 * width)-x_c))
    B_8_b = int((2 * width) + np.abs((1.5 * width)-x_c))
  
    #Draw the E
    canvas[B_1_a:B_1_b, B_2_a:B_2_b] = 1.0
    canvas[B_3_a:B_3_b, B_4_a:B_4_b] = 1.0
    canvas[B_5_a:B_5_b, B_6_a:B_6_b] = 1.0
    canvas[B_7_a:B_7_b, B_8_a:B_8_b] = 1.0

    #Get the E bounding box
    top = B_1_a
    bottom = B_3_b
    left = B_4_a
    right = B_6_b

    #Extract the E
    E = canvas[top:bottom, left:right]

    #Return with no gradient applied
    if (gradient == 0):

        return canvas
    
    #Get the dimensions of the E
    horizontal_range = int(right - left)
    vertical_range = int(bottom - top)

    #Get the orientation
    orientation = orientation

    #If the horizontal gradient is desired
    if (gradient == 1):

        if (orientation == 0):

            #Create a horizontal line
            horizontal_line = np.linspace(0,1,horizontal_range)

        elif (orientation == 1):

            #Create a horizontal line
            horizontal_line = np.linspace(1,0,horizontal_range)

        #Resize it
        horizontal_line = np.resize(horizontal_line, (1, horizontal_range))

        #Instantiate gradient storage
        horizontal_gradient = np.zeros((vertical_range, horizontal_range))

        #Populate the gradient
        for r in range(0,vertical_range):

            horizontal_gradient[r,:] = horizontal_line

        #Apply the gradient to the E
        E_horizontal_gradient = horizontal_gradient * canvas[top:bottom, left:right]

        #Update the canvas
        canvas[top:bottom, left:right] = E_horizontal_gradient

        return canvas

    #If the vertical gradient is desired
    if (gradient == 2):

        if (orientation == 0):

            #Create a vertical line
            vertical_line = np.linspace(0,1,vertical_range)

        elif (orientation == 1):

            #Create a vertical line
            vertical_line = np.linspace(1,0,vertical_range)

        #Instantiate gradient storage
        vertical_gradient = np.zeros((vertical_range, horizontal_range))

        #Populate the gradient
        for c in range(0,horizontal_range):

            vertical_gradient[:,c] = vertical_line

        #Apply the gradient to the E
        E_vertical_gradient = vertical_gradient * canvas[top:bottom, left:right]

        #Update the canvas
        canvas[top:bottom, left:right] = E_vertical_gradient

        return canvas
