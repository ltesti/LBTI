"""A module for various utilities and helper functions for images"""

import numpy as np
import scipy.ndimage as snd
import inpaint

#
# This function will return a resized image
#   obtained by resampling by a factor "factor"
#   the original "image"
def resize_image(image, factor):
    myMask = (np.isnan(image))
    myMaskedImg = np.ma.array(image, mask=myMask)
    NANMask =  myMaskedImg.filled(np.NaN)
    myBadArrays, my_num_BadArrays = snd.label(myMask)
    my_data_slices = snd.find_objects(myBadArrays)
    filled = inpaint.replace_nans(NANMask, 5, 0.5, 2, 'idw')
    zoom_filled = snd.zoom(filled, factor, order=3)
                # remove recentering option
                #if recenter:
                #    zoom_filled = self.__do_recenter(zoom_filled, resize * (xc - float(ixc)), resize * (yc - float(iyc)))
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