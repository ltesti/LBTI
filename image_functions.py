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
        resized_image = myZoomFilled
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

def block_sign_centroid(par):
    
    data = par[0]
    x = par[1][0]
    y = par[1][1]
    #
    x1 = x[0]
    x2 = x[1]
    #
    #print ("par: median={0}, x={1}, y={2}".format(np.nanmedian(data),x, y))
    data = data - np.nanmedian(data)
    values=[]
    for i in range(2):
        y1 = y[i][0]
        y2 = y[i][1]
        # get sign
        dfac = getsign(data[y1:y2+1,x1:x2+1])
        # get centroid
        xc, yc = get_centroid(dfac*data[y1:y2,x1:x2])
        xc = xc+x1
        yc = yc+y1
        values.append([dfac,xc,yc])
    #
    #print ("values: {0}".format(values))
    return values

#
# function that extracts the subimages 
def get_subimage_blkshift(par):
    data = par[0]
    sign = par[1]
    center = par[2]
    shift = par[3]
    subimsiz = par[4]
    rsfac = par[5]
    submed = par[6]
    do_shift = par[7]

    #
    ab_res = resize_image(data, rsfac)
    #
    outsize = rsfac * subimsiz
    subims = np.zeros((2,outsize,outsize))
    ab_res = ab_res - np.nanmedian(ab_res)
    for i in range(2):
        if do_shift:
            ab_shift = snd.interpolation.shift(ab_res,(rsfac*shift[i][1],rsfac*shift[i][0]))
            subims[i,:,:] = sign[i]*ab_shift[center[i][1]-outsize/2:center[i][1]+outsize/2,center[i][0]-outsize/2:center[i][0]+outsize/2]
        else:
            subims[i,:,:] = sign[i]*ab_res[center[i][1]-outsize/2:center[i][1]+outsize/2,center[i][0]-outsize/2:center[i][0]+outsize/2]
        if submed:
            radius = 2./3.*(float(outsize)/2.)
            subims[i,:,:] = subims[i,:,:] - outmedian(subims[i,:,:], radius)
    return subims

#
# Ancillary function to reformat the parameter set and to compute the
# median vales for the center and the shift of the block
def get_pars_ext(shiftcen,pars):
    mya = np.zeros(np.shape(shiftcen))
    for i in range(np.shape(shiftcen)[0]):
        mya[i,0,1] = pars[0][3]*shiftcen[i][0][1]
        mya[i,0,2] = pars[0][3]*shiftcen[i][0][2]
        mya[i,1,1] = pars[0][3]*shiftcen[i][1][1]
        mya[i,1,2] = pars[0][3]*shiftcen[i][1][2]
    med = ( [np.median(mya[:,0,1]),np.median(mya[:,0,2])], \
            [np.median(mya[:,1,1]),np.median(mya[:,1,2])] )
    center = ( [int(round(med[0][0])),int(round(med[0][1]))], \
               [int(round(med[1][0])),int(round(med[1][1]))] )
    shift = ( [-(med[0][0]-center[0][0]), -(med[0][1]-center[0][1])], \
              [-(med[1][0]-center[1][0]), -(med[1][1]-center[1][1])] )
    pars_ext=[]
    for i in range(len(pars)):
        sign = (shiftcen[i][0][0],shiftcen[i][1][0])
        par = (pars[i][0],sign,center,shift,pars[i][2],pars[i][3],pars[i][4])
        pars_ext.append(par)
    return pars_ext

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

#
#  image rotation
def rotima(par):
        return snd.interpolation.rotate(par[0],par[1],reshape=False)

def anular_stats(ima,r1,r2):
    xy = np.shape(ima)
    yc = float(xy[0])/2.
    xc = float(xy[1])/2.
    y,x = np.mgrid[0:xy[0]-1:xy[0]*1j,0:xy[1]-1:xy[1]*1j]
    radius = np.sqrt( (x-xc)**2. + (y-yc)**2. )
    n_in = np.where((radius >=r1) & (radius <= r2))
    mymean = np.mean(ima[n_in])
    mystd = np.std(ima[n_in])
    return mymean,mystd
