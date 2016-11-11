"""A module for various utilities and helper functions for images"""

import numpy as np
import scipy.ndimage as snd
import inpaint

#
# This function will return a resized image
#   obtained by resampling by a factor "factor"
#   the original "image"
def resize_image(image, factor):
    #
    myMask = (np.isnan(image))
    myMaskedImg = np.ma.array(image, mask=myMask)
    NANMask =  myMaskedImg.filled(np.NaN)
    if factor <= 1:
        resized_image = NANMask
    else:
        myBadArrays, my_num_BadArrays = snd.label(myMask)
        my_data_slices = snd.find_objects(myBadArrays)
        filled = inpaint.replace_nans(NANMask, 5, 0.5, 2, 'idw')
        zoom_filled = snd.zoom(filled, factor, order=3)
        zoom_mask = snd.zoom(myMask, factor, order=0)
        myZoomFilled = np.ma.array(zoom_filled, mask=zoom_mask)
        resized_image = myZoomFilled.filled(np.NaN)
    #
    return resized_image

#
# This function will return the median of the
#     points in one image outside a radius "radius" (in pixels)
def outmedian(data, radius):
    cen = np.shape(data)
    y,x = np.mgrid[0:cen[0]-1:cen[0]*1j,0:cen[1]-1:cen[1]*1j]
    wm = np.where((y-cen[0]/2.)*(y-cen[0]/2.)+(x-cen[1]/2.)*(x-cen[1]/2.) >= radius*radius)
    return np.nanmedian(data[wm])

#
# returns the sign of the sum of all array elements
def getsign(data):
    mytot = np.nansum(data)
    dfac = +1
    if mytot < 0.:
        dfac = -1
    return dfac

#
# Returns xc,yc position of the centroid of an array
def get_centroid(d):
    #
    #d = data[y1:y2+1,x1:x2+1]
    #y,x = np.mgrid[y1:y2:2*dd*1j,x1:x2:2*dd*1j]
    ll = np.shape(d)
    y,x = np.mgrid[0:ll[0]-1:ll[0]*1j,0:ll[1]-1:ll[1]*1j]
    dtot = np.nansum(d*d)
    #print("x={0}, y={1}, d={2}".format(np.shape(x),np.shape(y),np.shape(d)))
    xc = np.nansum((x*d*d))/dtot
    yc = np.nansum((y*d*d))/dtot
    return xc, yc

#
# function that extracts the subimages 
def get_subimage(par):
    data = par[0]
    x = par[1][0]
    y = par[1][1]
    subimsiz = par[2]
    rsfac = par[3]
    submed = par[4]
    #
    x1 = x[0]
    x2 = x[1]
    #
    ab_res = resize_image(data, rsfac)
    #
    outsize = rsfac * subimsiz
    subims = np.zeros((2,outsize,outsize))
    ab_res = ab_res - np.nanmedian(ab_res)
    # cycle the two positions
    for i in range(2):
        y1 = y[i][0]
        y2 = y[i][1]
        # get sign
        dfac = getsign(ab_res[y1:y2+1,x1:x2+1])
        # get centroid
        xc, yc = get_centroid(dfac*ab_res[y1:y2,x1:x2])
        xc = xc+x1
        yc = yc+y1
        #
        ixc = int(round(xc))
        iyc = int(round(yc))
        #ab_res[mynanmasks[0]] = np.nan
        #ab_res[mynanmasks[1]] = np.nan
        subims[i,:,:] = dfac*ab_res[iyc-outsize/2:iyc+outsize/2,ixc-outsize/2:ixc+outsize/2]
        if submed:
            radius = 2./3.*(float(outsize)/2.)
            subims[i,:,:] = subims[i,:,:] - outmedian(subims[i,:,:], radius)
    return subims
