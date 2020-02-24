import numpy as np
import matplotlib.pyplot as plt

def inspectCGH(CGH, SHOWCGH):

    '''
    Description: 'Inspects' the CGH by taking its fft and displaying its squared magintude
    
    Note: User has option to show the CGH along with its sqmag'd transform 

    Parameters: 
      CGH     : (real NxN numpy array) the CGH you wish to test
      SHOWCGH : (boolean) True/False if you want to show the CGH

    returns: no returned object, just plots desired objects
    '''
   
    #Get CGH dims
    N1 = np.shape(CGH)[0]
    N2 = np.shape(CGH)[1]
    
    #Input check
    if (N1 != N2):
        
        print("PLEASE PROVIDE A CGH WITH SQUARE DIMS") 
        return -1
   
    #If you want to show the CGH on the plot
    if (SHOWCGH == True):
        
    
        #Establish plot conditions
        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (10,5))

        #Take fft and get magnitude
        fft = np.fft.fftshift(np.fft.fft2(CGH))
        magnitude = np.abs(fft)

        #Calculate sqmag
        sqmag = np.square(magnitude)

        #Plot CGH
        ax1.imshow(CGH, cmap = 'gray')
        ax1.set_title("CGH")

        #Plot the sqmag of the CGH
        ax2.imshow(sqmag, cmap = 'gray')
        ax2.set_title("SQMAG { FFT2 {CGH} }")

        plt.show()

    #If you don't want to show the CGH on the same plot    
    if (SHOWCGH == False):
        
    
        #Establish plot conditions
        fig, (ax1) = plt.subplots(1,1, figsize = (10,5))

        #Take fft and get magnitude
        fft = np.fft.fftshift(np.fft.fft2(CGH))
        magnitude = np.abs(fft)

        #Calculate sqmag
        sqmag = np.square(magnitude)

        #plot the sqmag of the CGH
        ax1.imshow(sqmag, cmap = 'gray')
        ax1.set_title("SQMAG { FFT2 {CGH} }")

        plt.show()
