import numpy as np
import cgh
import matplotlib.pyplot as plt

def genCgh(complex_array, error_diffusion, T3, rounding_type, vis):

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
      vis                : (boolean) option to view ED in vector space with analytics per sample (not dev'd for non-ED case yet)
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
         
            if (vis == True and error_diffusion == True):

                #set figure params
                precision = 5
                fig, (ax, ax2) = plt.subplots(1,2, figsize = (10,5))
                ax.grid()
                titlestr1 = "Sample Visualization: " + "[" + str(m) + ", " + str(n) + "]"
                titlestr2 = "Sample Data: " + "[" + str(m) + ", " + str(n) + "]"
                ax.set_title(titlestr1)
                ax2.set_title(titlestr2)
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.set_aspect('equal')
                ax2.set_xlim(-1, 1)
                ax2.set_ylim(-1, 1)
                ax2.set_aspect('equal')
                ax2.xaxis.set_visible(False)
                ax2.yaxis.set_visible(False)

                #plot/info for raw sample
                ax.arrow(0, 0, raw_sample_real_part, raw_sample_imag_part, head_width = 0.02, color = 'k', length_includes_head = True) 
                textinfo1 = "Raw Sample: [" + str(np.round(raw_sample_real_part, precision)) + ", " + str(np.round(raw_sample_imag_part, precision)) + "]"
                ax2.text(-0.9, 0.6, textinfo1)

            # --- ERROR DIFFUSION: CORRECTION --- #

            if (error_diffusion == True):

                #Assign zero error to the first iteration
                if (m == 0 and n == 0):
                    error_real = 0
                    error_imag = 0
                    error = np.array([error_real, error_imag])
                    error_magnitude = 0

                if (vis == True and error_diffusion == True):

                    #info for error
                    textinfo5 = "Error from Previous: [" + str(np.round(error[0], precision)) + ", " + str(np.round(error[1], precision)) + "]"
                    ax2.text(-0.9, 0.8, textinfo5, color = 'r')

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

                #Purpose: Bring diffused sample back into the unit circle if it was diffused outside.
                #This doesn't happen very often, but it can

                raw_sample_mag = np.sqrt(np.square(raw_sample_real_part) + np.square(raw_sample_imag_part))

                if (raw_sample_mag > 1.0):

                    raw_sample_real_part = raw_sample_real_part / raw_sample_mag
                    raw_sample_imag_part = raw_sample_imag_part / raw_sample_mag

                # --- /ERROR DIFFUSION DRIFT CORRECTION --- #

                raw_sample = np.array([raw_sample_real_part, raw_sample_imag_part])

                if (vis == True and error_diffusion == True):

                    #plot/info for corrected sample
                    ax.arrow(0, 0, raw_sample[0], raw_sample[1], head_width = 0.02, color = 'g', length_includes_head = True) 
                    textinfo2 = "Corrected Sample: [" + str(np.round(raw_sample[0], precision)) + ", " + str(np.round(raw_sample[1], precision)) + "]"
                    ax2.text(-0.9, 0.4, textinfo2, color = 'g')

            # --- /ERROR DIFFUSION: CORRECTION --- #

            #If error diffusion is off, just use the raw sample
            elif (error_diffusion == False):

                raw_sample = np.array([raw_sample_real_part, raw_sample_imag_part])

            #Establish storage to keep track of projections
            projection_dictionary = {}

            #Perform projection of raw sample onto basis vectors
            for b in range(0, len(basis_vectors)):

                basis = basis_vectors[b] 

                if (vis == True and error_diffusion == True):

                    #plot the basis vectors
                    ax.arrow(0, 0, basis[0], basis[1], color = 'b', width = 0.01, head_width = 0.03, length_includes_head = True) 

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

            #Scale the largest projection value by the max cell magnitude
            largest_projection_value_scaled = max_cell_magnitude * largest_projection_value

            # --- ERROR DIFFUSION: GET ERROR --- #

            if (error_diffusion == True):

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

                #now, downsize that rounded sample so we can perform ED on the quantized sub-unit vector!
                #THIS is an important thing I wasn't understanding
                #this allows us to diffuse error within the unit circle WHILE TAKING THE ROUDING INTO ACCOUNT
                #AKA THE THING THAT IS CAUSING THE ERROR DUHHH
                downsized_rounded_projection_value = rounded_projection_value / max_cell_magnitude

                #Get the real and imag parts of the basis onto which the sample was projected
                basis_real = basis_vectors[largest_projection_phase_axis][0]
                basis_imag = basis_vectors[largest_projection_phase_axis][1]
                
                #Scale the basis according to the downsized projection
                scaled_basis_real = basis_real * downsized_rounded_projection_value
                scaled_basis_imag = basis_imag * downsized_rounded_projection_value
            
                if (vis == True and error_diffusion == True):

                    #plot/info for scaled basis
                    ax.arrow(0, 0, scaled_basis_real, scaled_basis_imag, color = 'indigo', width = 0.01, head_width = 0.03, length_includes_head = True) 
                    textinfo3 = "Projecting Onto: [" + str(np.round(scaled_basis_real, precision)) + ", " + str(np.round(scaled_basis_imag, precision)) + "]"
                    ax2.text(-0.9, 0.2, textinfo3, color = 'indigo')

                #Get the error associated with that projection
                error_real = raw_sample_real_part - scaled_basis_real
                error_imag = raw_sample_imag_part - scaled_basis_imag
                error = np.array([error_real, error_imag])
                error_magnitude = np.sqrt(np.square(error_real) + np.square(error_imag))

                if (vis == True and error_diffusion == True):

                    #info for error to be passed along
                    textinfo4 = "Error to be Passed: [" + str(np.round(error[0], precision)) + ", " + str(np.round(error[1], precision)) + "]"
                    ax2.text(-0.9, 0, textinfo4, color = 'r')

                    plt.show()

            # --- /ERROR DIFFUSION: GET ERROR --- #

            else: #if error diffusion is off, just perform the cell assignment

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
