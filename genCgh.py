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

    #Set max cell magnitude (restricted to eight)
    max_cell_magnitude = 8

    #Instantiate storage for CGH cells
    linear_array_of_cells = []

    for m in range(0,N):

        for n in range(0,N):

            #Get the real and imaginary components of the sample
            raw_sample_real_part = real_part[m,n]
            raw_sample_imag_part = imag_part[m,n]

            # --- ERROR DIFFUSION: CORRECTION --- #

            if (error_diffusion == True):

                #Assign zero error to the first iteration
                if (m == 0 and n == 0):
                    error_real = 0
                    error_imag = 0
                    error = np.array([error_real, error_imag])
                    error_magnitude = 0

                #Real part
                if (error_real >= 0):
                    raw_sample_real_part = raw_sample_real_part + np.abs(error_real)
                elif (error_real < 0):
                    raw_sample_real_part = raw_sample_real_part - np.abs(error_real)

                #Imaginary part
                if (error_imag >= 0):
                    raw_sample_imag_part = raw_sample_imag_part + np.abs(error_imag)
                elif (error_imag < 0):
                    raw_sample_imag_part = raw_sample_imag_part - np.abs(error_imag)

                # --- ERROR DIFFUSION DRIFT CORRECTION --- #

                raw_sample_mag = np.sqrt(np.square(raw_sample_real_part) + np.square(raw_sample_imag_part))

                if (raw_sample_mag > 1.0):

                    #Purpose: Bring diffused sample back into the unit circle if it was diffused outside.
                    #This doesn't happen very often, but it can

                    raw_sample_real_part = raw_sample_real_part / raw_sample_mag
                    raw_sample_imag_part = raw_sample_imag_part / raw_sample_mag

                # --- /ERROR DIFFUSION DRIFT CORRECTION --- #


                raw_sample = np.array([raw_sample_real_part, raw_sample_imag_part])

            # --- /ERROR DIFFUSION: CORRECTION --- #

            #If error diffusion is off, just use the raw sample
            elif (error_diffusion == False):

                raw_sample = np.array([raw_sample_real_part, raw_sample_imag_part])

            #Establish storage to keep track of projections
            projection_dictionary = {}

            #Perform projection of raw sample onto basis vectors
            for b in range(0, len(basis_vectors)):

                #TODO: consider unit normalizing all vectors before projection? -> phase error is all we care about...?

                basis = basis_vectors[b]
                projection = np.dot(raw_sample, basis)

                projection_dictionary[b] = projection

            #Sort the projections from largest to smallest
            projections_sorted = sorted(projection_dictionary.items(), key=lambda kv:kv[1], reverse=True)

            #Get the phase axis and value of the largest projection
            largest_projection_phase_axis = projections_sorted[0][0]
            largest_projection_value = projections_sorted[0][1]

            #Notify user of unexpected projection value (expected range [0:1])
            if (largest_projection_value > 1.0 or largest_projection_value < 0.0):

                print("UNEXPECTED PROJECTION OCCURED:", largest_projection_value)

            # --- ERROR DIFFUSION: GET ERROR --- #

            #TODO: consider getting and diffusing error based only on phase error? As in,
                  #compensate only for difference in angle between sample and basis.

            if (error_diffusion == True):

                #Get the real and imag parts of the basis onto which the sample was projected
                basis_real = basis_vectors[largest_projection_phase_axis][0]
                basis_imag = basis_vectors[largest_projection_phase_axis][1]

                #Scale the basis according to the projection
                scaled_basis_real = basis_real * largest_projection_value
                scaled_basis_imag = basis_imag * largest_projection_value
                
                #Get the error associated with that projection
                error_real = raw_sample_real_part - scaled_basis_real
                error_imag = raw_sample_imag_part - scaled_basis_imag
                error = np.array([error_real, error_imag])
                error_magnitude = np.sqrt(np.square(error_real) + np.square(error_imag))

            # --- /ERROR DIFFUSION: GET ERROR --- #

            #Scale the largest projection value by the max cell magnitude
            largest_projection_value_scaled = max_cell_magnitude * largest_projection_value

            #Round the projection and create the subaperture (cell)
            if (rounding_type == 1):

                rounded_projection_value = int(np.round(largest_projection_value_scaled))
                cell = cgh.genSubAp(rounded_projection_value, largest_projection_phase_axis, max_cell_magnitude, phase_sectors, T3)

            elif (rounding_type == 2):

                rounded_projection_value = int(np.ceil(largest_projection_value_scaled))
                cell = cgh.genSubAp(rounded_projection_value, largest_projection_phase_axis, max_cell_magnitude, phase_sectors, T3)

            elif (rounding_type == 3):

                rounded_projection_value = int(np.floor(largest_projection_value_scaled))
                cell = cgh.genSubAp(rounded_projection_value, largest_projection_phase_axis, max_cell_magnitude, phase_sectors, T3)

            #Append the cell for this [m,n] sample to list
            linear_array_of_cells.append(cell)

    #Instantiate row-based cell storage
    stacked_CGH_rows = []

    #Concatenate rows and append them to the row storage
    for r in range(0, np.square(N), N):

        stacked_CGH_rows.append(np.hstack(linear_array_of_cells[r:(r+N)]))

    #Concatenate all of the rows vertically to create the CGH
    CGH = np.vstack(stacked_CGH_rows)

    return CGH
