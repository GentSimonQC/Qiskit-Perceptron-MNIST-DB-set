import numpy as np
from math import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import imageio, elasticdeform

import DB_MNIST_checkpunctual



#==============================================================================================================
# I CREATE max_data VARIATIONS ON THE X INPUT MATRIX-MNIST-IMAGE FOR TESTING 'SAME'/'DIFFERENT' CLASSIFICATION
#==============================================================================================================

def stackflatmatrixes(max_data, original_im, sample_built_on_im, the_other_class_num, flatshape):
 


    FLATTEN_MERG_MATR = []
    #print('shape(original im)',np.shape(original_im))
    #print('shape(sample_built_on_im)',np.shape(sample_built_on_im))
    FLATTEN_MERG_MATR.append(original_im.flatten())
    FLATTEN_MERG_MATR = np.append(FLATTEN_MERG_MATR, sample_built_on_im, axis=0)

    
    #APPEND THE OTHER CLASS TO FLATTEN_MERG_MATR

    DATASET_3 = []
    LABELS_3 = []
    count = 0
    randstart = np.random.randint(300)
    randend = np.random.randint(500)
    for i in range(randstart, 2000+randend, 1):
        if DB_MNIST_checkpunctual.data_sets.train.labels[i] == the_other_class_num:
            DATASET_3.append(DB_MNIST_checkpunctual.REDUCEPIXELSTOTHEMAX(i))
            LABELS_3.append(DB_MNIST_checkpunctual.data_sets.train.labels[i])
            count += 1
        if count >= max_data:
            break
    print('shape(DATASET_3)',np.shape(DATASET_3),'np.shape(LABELS_3)',np.shape(LABELS_3))


    DATASET_3 = np.array(DATASET_3)
    FLATTEN_MERG_MATR = np.append(FLATTEN_MERG_MATR, DATASET_3.reshape(max_data,flatshape), axis=0)
    
    
    return FLATTEN_MERG_MATR





def Normalize(array, tuple_shape):

    X_deformed_flat = array.flatten()
        
    resim_max = np.max(X_deformed_flat)
    resim_min = np.min(X_deformed_flat)
    deltazero = -(resim_min-0)
    NORM_X_deformed_flat = []
    for idx in range(len(X_deformed_flat)):
        r = 0
        r = X_deformed_flat[idx] + deltazero
        r /= resim_max + deltazero
        NORM_X_deformed_flat.append(r)
    
    NORM_X_deformed_flat = np.array(NORM_X_deformed_flat)
    for idx, e in enumerate(NORM_X_deformed_flat):
        if abs(e) < 0.3:
            NORM_X_deformed_flat[idx] = 0
         
    #plt.imshow(NORM_X_deformed_flat.reshape(tuple_shape), cmap = plt.cm.binary)
    #plt.show()
    #print(NORM_X_deformed_flat)

    #print('shape(NORM_X_deformed_flat)',np.shape(NORM_X_deformed_flat))
    return NORM_X_deformed_flat



def example_main(what_number_label, tuple_shape, max_data, the_other_class_num):

    randstart = np.random.randint(300)
    randend = np.random.randint(500)
    for i in range(randstart, 2000+randend, 1):
        if DB_MNIST_checkpunctual.data_sets.train.labels[i] == what_number_label:
            break
    
    pick_a_random_input = i   # AMONGST THE data_sets.train.images (28x28) and labels STORED in THE DB
    
    im = DB_MNIST_checkpunctual.data_sets.train.images[pick_a_random_input]
    im = np.reshape(im, (28,28))
    #plt.imshow(im, cmap = plt.cm.binary)
    #plt.show()
    print('LABEL corr to index',pick_a_random_input,'of the DB is:',DB_MNIST_checkpunctual.data_sets.train.labels[pick_a_random_input])


    im_deformed = []
    for i in range(max_data-1):
        im_d_pre = elasticdeform.deform_random_grid(im, sigma=(0.25,1,-0.5,2,-0.5), points=5)
        im_deformed.append(Normalize(im_d_pre, (28,28)))
        #plt.imshow(im_deformed[-1].reshape(28,28), cmap = plt.cm.binary)
        #plt.show()
    
    X = []
    X.append(DB_MNIST_checkpunctual.REDUCEPIXELSTOTHEMAX__(im))
    for e in im_deformed:
        red = DB_MNIST_checkpunctual.REDUCEPIXELSTOTHEMAX__(e)
        #norm = Normalize(red, tuple_shape)
        X.append(red)
    
    # FLATTEN X
    X = np.reshape(X, (max_data, tuple_shape[0]*tuple_shape[1]))

    #===========================================================================
    # CHECK THE SAMPLE BUILT
    #===========================================================================
    #for e in X:
    #    plt.imshow(e.reshape(tuple_shape), cmap = plt.cm.binary)
    #    plt.show()

    # CHECK THE ORIGINAL REDUCEPIZELTOTHEMAX
    #just_to_check = DB_MNIST_checkpunctual.REDUCEPIXELSTOTHEMAX__(im) 
    #plt.imshow(just_to_check, cmap = plt.cm.binary)
    #plt.show()
    #plt.imshow(X[0].reshape(tuple_shape), cmap = plt.cm.binary)
    #plt.show()

    FLATTEN_MERG_MATR = stackflatmatrixes(max_data, X[0], X[1:], the_other_class_num, tuple_shape[0]*tuple_shape[1])
    print('shape FLATTEN_MERG_MATR',np.shape(FLATTEN_MERG_MATR))
    


    return FLATTEN_MERG_MATR


