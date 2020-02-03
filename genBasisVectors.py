import numpy as np

def genBasisVectors(phase_sectors):

    '''
    Description: Generates unit basis vectors that divide complex plane into desired number of phase sectors
    

    Parameters: 
      phase_sectors : (int) the number of sectors you wish to divide the complex plane into
    
    returns: numpy array containing numpy array representations of the unit basis vectors 
    '''

    if (phase_sectors == 8):

        '''

        3  2  1
         \ | /
      4 -- . -- 0
         / | \ 
        5  6  7
        
        '''
        
        #0 rad
        b0 = np.array([1,0], dtype = np.float64)

        #pi/4 rad
        b1 = np.array([1/np.sqrt(2),1/np.sqrt(2)], dtype = np.float64)

        #pi/2 rad
        b2 = np.array([0,1], dtype = np.float64)

        #3pi/4 rad
        b3 = np.array([-1/np.sqrt(2),1/np.sqrt(2)], dtype = np.float64)

        #pi rad
        b4 = np.array([-1,0], dtype = np.float64)

        #5pi/4 rad
        b5 = np.array([-1/np.sqrt(2),-1/np.sqrt(2)], dtype = np.float64)

        #3pi/2 rad
        b6 = np.array([0,-1], dtype = np.float64)

        #7pi/4 rad
        b7 = np.array([1/np.sqrt(2),-1/np.sqrt(2)], dtype = np.float64)

        basis_vectors = np.array([b0, b1, b2, b3, b4, b5, b6, b7])

        return basis_vectors

    else:

        print("This desired phase sector number has yet to be implemented!")
        print("You requested a phase sector number of: " + str(phase_sectors))
        return -1
