import numpy as np

def genSubAp(magnitude, phase, max_cell_magnitude, phase_sectors, T3):

    '''
    Description: Generates a cell-based subaperture to represent a magnitude/phase value
    
    Parameters: 
      magnitude            : (int) the magnitude that this cell should represent
      phase                : (int) the phase that this cell should represent
      max_cell_magnitude   : (int) the maximum quantized magnitude value allowed
      phase_sectors        : (int) the number of quantized complex plane phase value representations
      T3                   : (boolean) option to make the slit width three instead of one
    
    returns: a (max_cell_magnitude x phase_sectors) numpy array that is the cell representation of the magnitude/phase
    '''

    #create the blank cell
    cell = np.zeros((max_cell_magnitude,phase_sectors))

    #boundary check
    if (magnitude < 0 or magnitude > max_cell_magnitude):

        print("MAGNITUDE IS OUT OF BOUNDS")
        return -1

    if (phase < 0 or phase >= phase_sectors):

        print("PHASE IS OUT OF BOUNDS")
        return -1


    if (T3 == True):

        cell[0:magnitude, phase] = 1.0

        if (phase == 0): #if the slit is on the left side of the cell

            cell[0:magnitude, phase+1] = 1.0
            cell[0:magnitude, phase+int(phase_sectors-1)] = 1.0

        elif (phase == int(phase_sectors-1)): #if the slit is on the right side of the cell

            cell[0:magnitude, phase-1] = 1.0
            cell[0:magnitude, phase-int(phase_sectors-1)] = 1.0

        else: #if the slit is somewhere in the middle of the cell

            cell[0:magnitude, phase-1] = 1.0
            cell[0:magnitude, phase+1] = 1.0

    elif (T3 == False):

        cell[0:magnitude, phase] = 1.0

    #flip it for 'natural' orientation (this step does not seem to have any profound effect on the result)
    cell = np.flipud(cell)

    return cell
