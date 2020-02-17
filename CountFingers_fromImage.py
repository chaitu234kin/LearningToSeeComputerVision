

import numpy as np
def breakIntoGrids(im, s = 9):
    '''
    Break overall image into overlapping grids of size s x s, s must be odd.
    '''
    grids = []

    h = s//2 #half grid size minus one.
    for i in range(h, im.shape[0]-h):
        for j in range(h, im.shape[1]-h):
            grids.append(im[i-h:i+h+1,j-h:j+h+1].ravel())

    return np.vstack(grids)

def reshapeIntoImage(vector, im_shape, s = 9):
    '''
    Reshape vector back into image. 
    '''
    h = s//2 #half grid size minus one. 
    image = np.zeros(im_shape)
    image[h:-h, h:-h] = vector.reshape(im_shape[0]-2*h, im_shape[1]-2*h)

    return image

def count_fingers(im):
    '''
    Example submission for coding challenge. 
    
    Args: im (nxm) unsigned 8-bit grayscale image 
    Returns: One of three integers: 1, 2, 3
    
    '''

    ## ------ Input Pipeline Develped in this Module ----- ##
    #You may use the finger pixel detection pipeline we developed in this module:
    #You may also replace this code with your own pipeline if you prefer
    im = im > 75 #Threshold image
    X = breakIntoGrids(im, s = 9) #Break into 9x9 grids

    #Use rule we learned with decision tree
    treeRule1 = lambda X: np.logical_and(np.logical_and(X[:, 40] == 1, X[:,0] == 0), X[:, 53] == 0)
    yhat = treeRule1(X)
    number_of_finger_pixels = (yhat==True).sum()
    #Reshape prediction ino image:
    yhat_reshaped = reshapeIntoImage(yhat, im.shape)
    ## ----- Your Code Here ---- ##
    if (number_of_finger_pixels > 156):
        return 3
    elif (number_of_finger_pixels > 112):
        return 2
    else:
        return 1
