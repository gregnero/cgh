import numpy as np
import cgh

def genCGH(complex_array, phase_sectors, max_cell_magnitude, rounding_type, verbose, return_type):

    '''
    Author: Gregory M. Nero

    Description: constructs a cell-based CGH from a complex array using intrasample error diffusion
    TODO: intersample ED

    Note: restricted error diffusion to the unit circle because of limited cell representation
    
    Parameters: 
      complex_array      : (numpy array) an array of absolue max normalized real/imaginary values
      phase_sectors      : (int) the number of equal sections to divide the complex plane into
      max_cell_magnitude : (int) the max cell size 
      rounding_type      : (int) the kind of rounding you want to do for value quantization
                         : 1 -> traditional rounding
                         : 2 -> always up
                         : 3 -> always down
      verbose            : (boolean) choose to display useful projection/cell information
      return_type        : (int) return a CGH with the desired number of projections
                         
    returns: (numpy array) the CGH
    '''

    #Get the real and imaginary parts
    real_part = np.real(complex_array)
    imag_part = np.imag(complex_array)

    #Input check
    cgh.isMaxMagUnity(real_part, imag_part)
    cgh.isPhaseBehaved(real_part, imag_part)
    N = cgh.isCongruent(real_part, imag_part)

    #Generate the basis vectors (also acts as gatekeeper for invalid phase sector requests)
    basis_vectors = cgh.genBasisVectors(phase_sectors)

    #Establish container to store verbose information
    info = []

    if (phase_sectors == 8):

        ''' return_type options
        : 1 -> one projection
        : 2 -> two projections
        : 3 -> three projections
        : 4 -> four projections (if possible)
        '''
        
        #Establish primary linear cell storage
        linear_array_of_cells_p1 = []
        linear_array_of_cells_p2 = []
        linear_array_of_cells_p3 = []
        linear_array_of_cells_p4 = [] #when possible
    
        for m in range(0,N):

            for n in range(0,N):

                info.append("------------ BEGIN SAMPLE ------------")
                info.append("Desired Number of Phase Sectors: " + str(phase_sectors))
                info.append("Desired Max Cell Magnitude: " + str(max_cell_magnitude))
                info.append("Desired Number of Projections: " + str(return_type))
                
                if (rounding_type == 1):

                    round_str = 'traditional'

                elif (rounding_type == 2):

                    round_str = 'up'
                
                elif (rounding_type == 3):

                    round_str = 'down'

                info.append("Desired Rounding Type: " + round_str)
                info.append("Index in Complex Array: " + "[" + str(m) + ", " + str(n) + "]") 
                info.append("~")

                #Get raw sample information for this point in the complex array
                raw_sample_real_part = real_part[m,n]
                raw_sample_imag_part = imag_part[m,n]
                raw_sample = np.array([raw_sample_real_part, raw_sample_imag_part])

                info.append("Raw Sample: [" + str(raw_sample_real_part) + ", " + str(raw_sample_imag_part) + "]")
                info.append("~")

                #Set up a dictionary to track basis/positive projection pairs
                positive_projection_dictionary = {}

                #Set up tracker to sum all the positive projections that occur
                positive_projection_sum = 0

                #Project the raw sample onto each of the basis vectors and scale/store the projection if it is positive
                for b in range(0, phase_sectors): #consider thresholding in this loop

                    basis = basis_vectors[b]
                    projection = np.dot(raw_sample, basis)

                    if (projection <= 0):

                        continue
                        
                    positive_projection_sum = positive_projection_sum + projection

                    scaled_positive_projection = max_cell_magnitude * projection

                    positive_projection_dictionary[b] = scaled_positive_projection

                #This is a list object that stores basis/projection pair tuples from largest to smallest projection
                positive_projections_sorted = sorted(positive_projection_dictionary.items(), key=lambda kv:kv[1], reverse=True)
                
                #Get the number of non negative projections that occured for this sample
                number_of_positive_projections = len(positive_projections_sorted)

                info.append("Total Number of Positive Projections: " + str(number_of_positive_projections))
                info.append("~")

                #Take care of the case where the raw sample is zero
                if (positive_projection_sum == 0):
   
                    blank_cell = cgh.genSubAp(0, 0, max_cell_magnitude, phase_sectors)
                    linear_array_of_cells_p1.append(blank_cell)
                    linear_array_of_cells_p2.append(blank_cell)
                    linear_array_of_cells_p3.append(blank_cell)
                    linear_array_of_cells_p4.append(blank_cell)
                    continue

                #Take care of the case where there are three positive projections
                elif (number_of_positive_projections == 3):
                
                    basis_magnitude_dictionary = {}

                    #The fourth projection will not exist if there are only three positive projections
                    p4 = np.zeros((max_cell_magnitude, phase_sectors))

                    for p in range(0, number_of_positive_projections):

                        current_basis_index = positive_projections_sorted[p][0]
                        current_basis_vector = basis_vectors[current_basis_index]
                        
                        #The first projection
                        if (p == 0):

                            #Calculate the projection
                            projection1 = np.dot(raw_sample, current_basis_vector)
                            
                            #Scale and round based on preference
                            if (rounding_type == 1):
                                scaled_magnitude1 = np.abs(np.round(projection1 * max_cell_magnitude))
                            elif (rounding_type == 2):
                                scaled_magnitude1 = np.abs(np.ceil(projection1 * max_cell_magnitude))
                            elif (rounding_type == 3):
                                scaled_magnitude1 = np.abs(np.floor(projection1 * max_cell_magnitude))
                            
                            #Threshold if out of bounds
                            if (scaled_magnitude1 > max_cell_magnitude):
                                scaled_magnitude1 = max_cell_magnitude

                            #Catch the first projection
                            p1 = cgh.genSubAp(int(scaled_magnitude1), current_basis_index, max_cell_magnitude, phase_sectors)
                            linear_array_of_cells_p1.append(p1)
                           
                            #Store the projection and its scaled magnitude in the dictionary
                            basis_magnitude_dictionary[current_basis_index] = scaled_magnitude1
                            
                            #Calculate the approximated basis vector
                            approximated_vector1 = projection1 * current_basis_vector
                            approximated_vector1_real = approximated_vector1[0]
                            approximated_vector1_imag = approximated_vector1[1]

                            #Bring it back inside the unit circle if it is out
                            approximated_vector1_mag = np.sqrt(np.square(approximated_vector1_real) + np.square(approximated_vector1_imag))

                            if (approximated_vector1_mag > 1.0):

                                approximated_vector1_real = approximated_vector1_real / approximated_vector1_mag
                                approximated_vector1_imag = approximated_vector1_imag / approximated_vector1_mag

                            #---------- BEGIN INTRASAMPLE ERROR CORRECTION 1 ----------#

                            delta_real1 = raw_sample_real_part - approximated_vector1_real
                            delta_imag1 = raw_sample_imag_part - approximated_vector1_imag

                            if (delta_real1 >= 0):
                                new_sample_real1 = raw_sample_real_part + np.abs(delta_real1)
                            elif (delta_real1 < 0):
                                new_sample_real1 = raw_sample_real_part - np.abs(delta_real1)
                            else:
                                print("UNEXCPECTED CASE ENCOUNTERED - 1")

                            if (delta_imag1 >= 0):
                                new_sample_imag1 = raw_sample_imag_part + np.abs(delta_imag1)
                            elif (delta_imag1 < 0):
                                new_sample_imag1 = raw_sample_imag_part - np.abs(delta_imag1)
                            else:
                                print("UNEXCPECTED CASE ENCOUNTERED - 1")

                            new_sample1_mag = np.sqrt(np.square(new_sample_real1) + np.square(new_sample_imag1))

                            if (new_sample1_mag > 1.0):
                                
                                new_sample_real1 = new_sample_real1 / new_sample1_mag
                                new_sample_imag1 = new_sample_imag1 / new_sample1_mag

                            new_sample1 = np.array([new_sample_real1, new_sample_imag1])


                            #---------- END INTRASAMPLE ERROR CORRECTION 1 ----------#

                            info.append("PROJECTION 1")
                            info.append("Will project this vector onto basis " + str(current_basis_index) + ": " + "[" + str(raw_sample_real_part) + ", " + str(raw_sample_imag_part) + "]")
                            info.append("Projection Value = " + str(projection1))
                            info.append("Scaled and Rounded Magnitude: " + str(scaled_magnitude1))
                            info.append("Approximate Vector: [" + str(approximated_vector1_real) + ", " + str(approximated_vector1_imag) + "]")
                            info.append("New Sample: [" + str(new_sample_real1) + ", " + str(new_sample_imag1) + "]")
                            info.append("~")

                        #The second projection
                        if (p == 1):

                            #Calculate the projection
                            projection2 = np.dot(new_sample1, current_basis_vector)

                            #Round based on preference
                            if (rounding_type == 1):
                                scaled_magnitude2 = np.abs(np.round(projection2 * max_cell_magnitude))
                            elif (rounding_type == 2):
                                scaled_magnitude2 = np.abs(np.ceil(projection2 * max_cell_magnitude))
                            elif (rounding_type == 3):
                                scaled_magnitude2 = np.abs(np.floor(projection2 * max_cell_magnitude))

                            #Threshold if out of bounds
                            if (scaled_magnitude2 > max_cell_magnitude):
                                scaled_magnitude2 = max_cell_magnitude

                            #Catch the second projection
                            p2 = cgh.genSubAp(int(scaled_magnitude2), current_basis_index, max_cell_magnitude, phase_sectors)
                            linear_array_of_cells_p2.append(p1+p2)

                            #Store the projection and its scaled magnitude in the dictionary
                            basis_magnitude_dictionary[current_basis_index] = scaled_magnitude2
                            
                            #Calculate the approximated basis vector
                            approximated_vector2 = projection2 * current_basis_vector
                            approximated_vector2_real = approximated_vector2[0]
                            approximated_vector2_imag = approximated_vector2[1]

                            #Bring it back inside the unit circle if it is out
                            approximated_vector2_mag = np.sqrt(np.square(approximated_vector2_real) + np.square(approximated_vector2_imag))

                            if (approximated_vector2_mag > 1.0):

                                approximated_vector2_real = approximated_vector2_real / approximated_vector2_mag
                                approximated_vector2_imag = approximated_vector2_imag / approximated_vector2_mag

                            #---------- BEGIN INTRASAMPLE ERROR CORRECTION 2 ----------#

                            delta_real2 = new_sample_real1 - approximated_vector2_real
                            delta_imag2 = new_sample_imag1 - approximated_vector2_imag

                            if (delta_real2 >= 0):
                                new_sample_real2 = new_sample_real1 + np.abs(delta_real2)
                            elif (delta_real2 < 0):
                                new_sample_real2 = new_sample_real1  - np.abs(delta_real2)
                            else:
                                print("UNEXCPECTED CASE ENCOUNTERED - 2")

                            if (delta_imag2 >= 0):
                                new_sample_imag2 = new_sample_imag1 + np.abs(delta_imag2)
                            elif (delta_imag2 < 0):
                                new_sample_imag2 = new_sample_imag1 - np.abs(delta_imag2)
                            else:
                                print("UNEXCPECTED CASE ENCOUNTERED - 2")

                            new_sample2_mag = np.sqrt(np.square(new_sample_real2) + np.square(new_sample_imag2))

                            if (new_sample2_mag > 1.0):

                                new_sample_real2 = new_sample_real2 / new_sample2_mag
                                new_sample_imag2 = new_sample_imag2 / new_sample2_mag

                            new_sample2 = np.array([new_sample_real2, new_sample_imag2]) 

                            #---------- END INTRASAMPLE ERROR CORRECTION 2 ----------#

                            info.append("PROJECTION 2")
                            info.append("Will project this vector onto basis " + str(current_basis_index) + ": " + "[" + str(new_sample_real1) + ", " + str(new_sample_imag1) + "]")
                            info.append("Projection Value = " + str(projection2))
                            info.append("Scaled and Rounded Magnitude: " + str(scaled_magnitude2))
                            info.append("Approximate Vector: [" + str(approximated_vector2_real) + ", " + str(approximated_vector2_imag) + "]")
                            info.append("New Sample: [" + str(new_sample_real2) + ", " + str(new_sample_imag2) + "]")
                            info.append("~")

                        #The third projection
                        if (p == 2):

                            #Calculate the projection
                            projection3 = np.dot(new_sample2, current_basis_vector)

                            #Round based on preference
                            if (rounding_type == 1):
                                scaled_magnitude3 = np.abs(np.round(projection3 * max_cell_magnitude))
                            elif (rounding_type == 2):
                                scaled_magnitude3 = np.abs(np.ceil(projection3 * max_cell_magnitude))
                            elif (rounding_type == 3):
                                scaled_magnitude3 = np.abs(np.floor(projection3 * max_cell_magnitude))

                            #Threshold if out of bounds
                            if (scaled_magnitude3 > max_cell_magnitude):
                                scaled_magnitude3 = max_cell_magnitude

                            #Catch the third projection
                            p3 = cgh.genSubAp(int(scaled_magnitude3), current_basis_index, max_cell_magnitude, phase_sectors)
                            linear_array_of_cells_p3.append(p1+p2+p3)

                            #Also fill the p4 array because we only want three projections represented
                            linear_array_of_cells_p4.append(p1+p2+p3)

                            #Store the projection and its scaled magnitude in the dictionary
                            basis_magnitude_dictionary[current_basis_index] = scaled_magnitude3
                            
                            #Calculate the approximated basis vector
                            approximated_vector3 = projection3 * current_basis_vector
                            approximated_vector3_real = approximated_vector3[0]
                            approximated_vector3_imag = approximated_vector3[1]

                            #Bring it back inside the unit circle if it is out
                            approximated_vector3_mag = np.sqrt(np.square(approximated_vector3_real) + np.square(approximated_vector3_imag))

                            if (approximated_vector3_mag > 1.0):

                                approximated_vector3_real = approximated_vector3_real / approximated_vector3_mag
                                approximated_vector3_imag = approximated_vector3_imag / approximated_vector3_mag

                            info.append("PROJECTION 3")
                            info.append("Will project this vector onto basis " + str(current_basis_index) + ": " + "[" + str(new_sample_real2) + ", " + str(new_sample_imag2) + "]")
                            info.append("Projection Value = " + str(projection3))
                            info.append("Scaled and Rounded Magnitude: " + str(scaled_magnitude3))
                            info.append("Approximate Vector: [" + str(approximated_vector3_real) + ", " + str(approximated_vector3_imag) + "]")
                            info.append("~")

                #Take care of the case where there are four positive projections
                elif (number_of_positive_projections == 4):
                
                    basis_magnitude_dictionary = {}

                    for p in range(0, number_of_positive_projections):

                        current_basis_index = positive_projections_sorted[p][0]
                        current_basis_vector = basis_vectors[current_basis_index]
                        
                        #The first projection
                        if (p == 0):

                            #Calculate the projection
                            projection1 = np.dot(raw_sample, current_basis_vector)

                            #Scale and round based on preference
                            if (rounding_type == 1):
                                scaled_magnitude1 = np.abs(np.round(projection1 * max_cell_magnitude))
                            elif (rounding_type == 2):
                                scaled_magnitude1 = np.abs(np.ceil(projection1 * max_cell_magnitude))
                            elif (rounding_type == 3):
                                scaled_magnitude1 = np.abs(np.floor(projection1 * max_cell_magnitude))

                            #Threshold if out of bounds
                            if (scaled_magnitude1 > max_cell_magnitude):
                                scaled_magnitude1 = max_cell_magnitude

                            #Catch the first projection
                            p1 = cgh.genSubAp(int(scaled_magnitude1), current_basis_index, max_cell_magnitude, phase_sectors)
                            linear_array_of_cells_p1.append(p1)

                            #Store the projection and its scaled magnitude in the dictionary
                            basis_magnitude_dictionary[current_basis_index] = scaled_magnitude1
                            
                            #Calculate the approximated basis vector
                            approximated_vector1 = projection1 * current_basis_vector
                            approximated_vector1_real = approximated_vector1[0]
                            approximated_vector1_imag = approximated_vector1[1]

                            #Bring it back inside the unit circle if it is out
                            approximated_vector1_mag = np.sqrt(np.square(approximated_vector1_real) + np.square(approximated_vector1_imag))

                            if (approximated_vector1_mag > 1.0):

                                approximated_vector1_real = approximated_vector1_real / approximated_vector1_mag
                                approximated_vector1_imag = approximated_vector1_imag / approximated_vector1_mag

                            #---------- BEGIN INTRASAMPLE ERROR CORRECTION 1 ----------#

                            delta_real1 = raw_sample_real_part - approximated_vector1_real
                            delta_imag1 = raw_sample_imag_part - approximated_vector1_imag

                            if (delta_real1 >= 0):
                                new_sample_real1 = raw_sample_real_part + np.abs(delta_real1)
                            elif (delta_real1 < 0):
                                new_sample_real1 = raw_sample_real_part - np.abs(delta_real1)
                            else:
                                print("UNEXCPECTED CASE ENCOUNTERED - 1")

                            if (delta_imag1 >= 0):
                                new_sample_imag1 = raw_sample_imag_part + np.abs(delta_imag1)
                            elif (delta_imag1 < 0):
                                new_sample_imag1 = raw_sample_imag_part - np.abs(delta_imag1)
                            else:
                                print("UNEXCPECTED CASE ENCOUNTERED - 1")

                            new_sample1_mag = np.sqrt(np.square(new_sample_real1) + np.square(new_sample_imag1))

                            if (new_sample1_mag > 1.0):
                                
                                new_sample_real1 = new_sample_real1 / new_sample1_mag
                                new_sample_imag1 = new_sample_imag1 / new_sample1_mag

                            new_sample1 = np.array([new_sample_real1, new_sample_imag1]) 

                            #---------- END INTRASAMPLE ERROR CORRECTION 1 ----------#

                            info.append("PROJECTION 1")
                            info.append("Will project this vector onto basis " + str(current_basis_index) + ": " + "[" + str(raw_sample_real_part) + ", " + str(raw_sample_imag_part) + "]")
                            info.append("Projection Value = " + str(projection1))
                            info.append("Scaled and Rounded Magnitude: " + str(scaled_magnitude1))
                            info.append("Approximate Vector: [" + str(approximated_vector1_real) + ", " + str(approximated_vector1_imag) + "]")
                            info.append("New Sample: [" + str(new_sample_real1) + ", " + str(new_sample_imag1) + "]")
                            info.append("~")
                             
                        #The second projection
                        if (p == 1):

                            #Calculate the projection
                            projection2 = np.dot(new_sample1, current_basis_vector)

                            #Round based on preference
                            if (rounding_type == 1):
                                scaled_magnitude2 = np.abs(np.round(projection2 * max_cell_magnitude))
                            elif (rounding_type == 2):
                                scaled_magnitude2 = np.abs(np.ceil(projection2 * max_cell_magnitude))
                            elif (rounding_type == 3):
                                scaled_magnitude2 = np.abs(np.floor(projection2 * max_cell_magnitude))

                            #Threshold if out of bounds
                            if (scaled_magnitude2 > max_cell_magnitude):
                                scaled_magnitude2 = max_cell_magnitude
                            
                            #Catch the second projection
                            p2 = cgh.genSubAp(int(scaled_magnitude2), current_basis_index, max_cell_magnitude, phase_sectors)
                            linear_array_of_cells_p2.append(p1+p2)

                            #Store the projection and its scaled magnitude in the dictionary
                            basis_magnitude_dictionary[current_basis_index] = scaled_magnitude2
                            
                            #Calculate the approximated basis vector
                            approximated_vector2 = projection2 * current_basis_vector
                            approximated_vector2_real = approximated_vector2[0]
                            approximated_vector2_imag = approximated_vector2[1]

                            #Bring it back inside the unit circle if it is out
                            approximated_vector2_mag = np.sqrt(np.square(approximated_vector2_real) + np.square(approximated_vector2_imag))

                            if (approximated_vector2_mag > 1.0):

                                approximated_vector2_real = approximated_vector2_real / approximated_vector2_mag
                                approximated_vector2_imag = approximated_vector2_imag / approximated_vector2_mag

                            #---------- BEGIN INTRASAMPLE ERROR CORRECTION 2 ----------#

                            delta_real2 = new_sample_real1 - approximated_vector2_real
                            delta_imag2 = new_sample_imag1 - approximated_vector2_imag

                            if (delta_real2 >= 0):
                                new_sample_real2 = new_sample_real1 + np.abs(delta_real2)
                            elif (delta_real2 < 0):
                                new_sample_real2 = new_sample_real1  - np.abs(delta_real2)
                            else:
                                print("UNEXCPECTED CASE ENCOUNTERED - 2")

                            if (delta_imag2 >= 0):
                                new_sample_imag2 = new_sample_imag1 + np.abs(delta_imag2)
                            elif (delta_imag2 < 0):
                                new_sample_imag2 = new_sample_imag1 - np.abs(delta_imag2)
                            else:
                                print("UNEXCPECTED CASE ENCOUNTERED - 2")

                            new_sample2_mag = np.sqrt(np.square(new_sample_real2) + np.square(new_sample_imag2))

                            if (new_sample2_mag > 1.0):

                                new_sample_real2 = new_sample_real2 / new_sample2_mag
                                new_sample_imag2 = new_sample_imag2 / new_sample2_mag

                            new_sample2 = np.array([new_sample_real2, new_sample_imag2]) 

                            #---------- END INTRASAMPLE ERROR CORRECTION 2 ----------#

                            info.append("PROJECTION 2")
                            info.append("Will project this vector onto basis " + str(current_basis_index) + ": " + "[" + str(new_sample_real1) + ", " + str(new_sample_imag1) + "]")
                            info.append("Projection Value = " + str(projection2))
                            info.append("Scaled and Rounded Magnitude: " + str(scaled_magnitude2))
                            info.append("Approximate Vector: [" + str(approximated_vector2_real) + ", " + str(approximated_vector2_imag) + "]")
                            info.append("New Sample: [" + str(new_sample_real2) + ", " + str(new_sample_imag2) + "]")
                            info.append("~")

                        #The third projection
                        if (p == 2):

                            #Calculate the projection
                            projection3 = np.dot(new_sample2, current_basis_vector)

                            #Round based on preference
                            if (rounding_type == 1):
                                scaled_magnitude3 = np.abs(np.round(projection3 * max_cell_magnitude))
                            elif (rounding_type == 2):
                                scaled_magnitude3 = np.abs(np.ceil(projection3 * max_cell_magnitude))
                            elif (rounding_type == 3):
                                scaled_magnitude3 = np.abs(np.floor(projection3 * max_cell_magnitude))

                            #Threshold if out of bounds
                            if (scaled_magnitude3 > max_cell_magnitude):
                                scaled_magnitude3 = max_cell_magnitude

                            #Catch the third projection
                            p3 = cgh.genSubAp(int(scaled_magnitude3), current_basis_index, max_cell_magnitude, phase_sectors)
                            linear_array_of_cells_p3.append(p1+p2+p3)

                            #Store the projection and its scaled magnitude in the dictionary
                            basis_magnitude_dictionary[current_basis_index] = scaled_magnitude3
                            
                            #Calculate the approximated basis vector
                            approximated_vector3 = projection3 * current_basis_vector
                            approximated_vector3_real = approximated_vector3[0]
                            approximated_vector3_imag = approximated_vector3[1]
                            
                            #Bring it back inside the unit circle if it is out
                            approximated_vector3_mag = np.sqrt(np.square(approximated_vector3_real) + np.square(approximated_vector3_imag))

                            if (approximated_vector3_mag > 1.0):

                                approximated_vector3_real = approximated_vector3_real / approximated_vector3_mag
                                approximated_vector3_imag = approximated_vector3_imag / approximated_vector3_mag

                            #---------- BEGIN INTRASAMPLE ERROR CORRECTION 3 ----------#

                            delta_real3 = new_sample_real2 - approximated_vector3_real
                            delta_imag3 = new_sample_imag2 - approximated_vector3_imag

                            if (delta_real3 >= 0):
                                new_sample_real3 = new_sample_real2 + np.abs(delta_real3)
                            elif (delta_real3 < 0):
                                new_sample_real3 = new_sample_real2  - np.abs(delta_real3)
                            else:
                                print("UNEXCPECTED CASE ENCOUNTERED - 3")

                            if (delta_imag3 >= 0):
                                new_sample_imag3 = new_sample_imag2 + np.abs(delta_imag3)
                            elif (delta_imag3 < 0):
                                new_sample_imag3 = new_sample_imag2 - np.abs(delta_imag3)
                            else:
                                print("UNEXCPECTED CASE ENCOUNTERED - 3")

                            new_sample3_mag = np.sqrt(np.square(new_sample_real3) + np.square(new_sample_imag3))

                            if (new_sample3_mag > 1.0):

                                new_sample_real3 = new_sample_real3 / new_sample3_mag
                                new_sample_imag3 = new_sample_imag3 / new_sample3_mag

                            new_sample3 = np.array([new_sample_real3, new_sample_imag3]) 

                            #---------- END INTRASAMPLE ERROR CORRECTION 3 ----------#

                            info.append("PROJECTION 3")
                            info.append("Will project this vector onto basis " + str(current_basis_index) + ": " + "[" + str(new_sample_real2) + ", " + str(new_sample_imag2) + "]")
                            info.append("Projection Value = " + str(projection3))
                            info.append("Scaled and Rounded Magnitude: " + str(scaled_magnitude3))
                            info.append("Approximate Vector: [" + str(approximated_vector3_real) + ", " + str(approximated_vector3_imag) + "]")
                            info.append("New Sample: [" + str(new_sample_real3) + ", " + str(new_sample_imag3) + "]")
                            info.append("~")

                        #The fourth projection
                        if (p == 3):

                            #Calculate the projection
                            projection4 = np.dot(new_sample3, current_basis_vector)

                            #Round based on preference
                            if (rounding_type == 1):
                                scaled_magnitude4 = np.abs(np.round(projection4 * max_cell_magnitude))
                            elif (rounding_type == 2):
                                scaled_magnitude4 = np.abs(np.ceil(projection4 * max_cell_magnitude))
                            elif (rounding_type == 3):
                                scaled_magnitude4 = np.abs(np.floor(projection4 * max_cell_magnitude))

                            #Threshold if out of bounds
                            if (scaled_magnitude4 > max_cell_magnitude):
                                scaled_magnitude4 = max_cell_magnitude

                            #Catch the fourth projection
                            p4 = cgh.genSubAp(int(scaled_magnitude4), current_basis_index, max_cell_magnitude, phase_sectors)
                            linear_array_of_cells_p4.append(p1+p2+p3+p4)

                            #Store the projection and its scaled magnitude in the dictionary
                            basis_magnitude_dictionary[current_basis_index] = scaled_magnitude4
                            
                            #Calculate the approximated basis vector
                            approximated_vector4 = projection4 * current_basis_vector
                            approximated_vector4_real = approximated_vector4[0]
                            approximated_vector4_imag = approximated_vector4[1]

                            approximated_vector4_mag = np.sqrt(np.square(approximated_vector4_real) + np.square(approximated_vector4_imag))

                            if (approximated_vector4_mag > 1.0):

                                approximated_vector4_real = approximated_vector4_real / approximated_vector4_mag
                                approximated_vector4_imag = approximated_vector4_imag / approximated_vector4_mag

                            info.append("PROJECTION 4")
                            info.append("Will project this vector onto basis " + str(current_basis_index) + ": " + "[" + str(new_sample_real3) + ", " + str(new_sample_imag3) + "]")
                            info.append("Projection Value = " + str(projection4))
                            info.append("Scaled and Rounded Magnitude: " + str(scaled_magnitude4))
                            info.append("Approximate Vector: [" + str(approximated_vector4_real) + ", " + str(approximated_vector4_imag) + "]")
                            info.append("~")

                else:

                    print("UH OH! THIS NUMBER OF POSITIVE PROJECTIONS WASN'T ACCOUNTED FOR!")
                    return -1


                
                info.append("Cell: ")

                if (return_type == 1):

                    info.append(str(p1))
                
                elif (return_type == 2):

                    info.append(p1 + p2)

                elif (return_type == 3):

                    info.append(p1 + p2 + p3)

                elif (return_type == 4):

                    info.append(p1 + p2 + p3 + p4)

                info.append("------------ END SAMPLE ------------")
                info.append("\n")

                if (verbose == True):

                    #To ensure that 'useful' cells are being displayed
                    cell_importance = 1

                    if (np.sum(p1) >= cell_importance):

                        print(*info, sep = '\n')

                info.clear()

        #Create the CGH from the cells
        stacked_CGH_rows_p1 = []
        stacked_CGH_rows_p2 = []
        stacked_CGH_rows_p3 = []
        stacked_CGH_rows_p4 = []

        for r in range(0, np.square(N), N):

            stacked_CGH_rows_p1.append(np.hstack(linear_array_of_cells_p1[r:(r+N)]))
            stacked_CGH_rows_p2.append(np.hstack(linear_array_of_cells_p2[r:(r+N)]))
            stacked_CGH_rows_p3.append(np.hstack(linear_array_of_cells_p3[r:(r+N)]))
            stacked_CGH_rows_p4.append(np.hstack(linear_array_of_cells_p4[r:(r+N)]))
        
        CGH_p1 = np.vstack(stacked_CGH_rows_p1)
        CGH_p2 = np.vstack(stacked_CGH_rows_p2)
        CGH_p3 = np.vstack(stacked_CGH_rows_p3)
        CGH_p4 = np.vstack(stacked_CGH_rows_p4)
        
        if (return_type == 1):

            return CGH_p1
        
        elif (return_type == 2):
            
            return CGH_p2

        elif (return_type == 3):

            return CGH_p3

        elif (return_type == 4):

            return CGH_p4

        else:

            print("PLEASE SPECIFY A VALID RETURN TYPE")
            return -1
