import numpy as np
import cgh

def genCgh(complex_array, error_diffusion, T3, rounding_type):

    '''
    Author: Gregory M. Nero (gmn8357@rit.edu)

    Description: Constructs a binary computer-generated Fraunhofer hologram

    Parameters: 
      complex_array      : (numpy array) an array of absolue max normalized real/imaginary values
      error_diffusion    : (boolean) option to perform intersample error diffusion 
      T3                 : (boolean) option to make the slit width three instead of one
      rounding_type      : (int) the kind of rounding you want to do for value quantization
                         : 1 -> traditional rounding
                         : 2 -> always up
                         : 3 -> always down
    Returns              : (numpy array) CGH
    '''

    #Get the real and imaginary parts
    real_part = np.real(complex_array)
    imag_part = np.imag(complex_array)

    #Check input
    cgh.isMaxMagUnity(real_part, imag_part)
    cgh.isPhaseBehaved(real_part, imag_part)
    N = cgh.isCongruent(real_part, imag_part)

    #Generate the basis vectors (restricted to eight) 
    phase_sectors = 8 
    basis_vectors = cgh.genBasisVectors(phase_sectors)
