from __future__ import division, print_function

__author__ = 'ltesti - 3 Nov 2016'

import numpy as np
import scipy.signal as ssig
import scipy.ndimage as snd

from multiprocessing import Pool

#from astroquery.irsa import Irsa
#import astropy.units as u
#import astropy.coordinates as coord
import astropy.io.fits as aiof

import matplotlib.pyplot as plt

#import aplpy

import time
import os

# logging
import logging

# local package functions
import inpaint 

    #log = logging.getLogger()
    #log.setLevel(LOG_LEVEL)
    #logfile = logging.FileHandler(filename=LOG_FILENAME, mode='w')
    #logfile.setFormatter(logging.Formatter(LOG_FORMAT))
    #log.addHandler(logfile)

    #logging.info("")
    #logging.info("****************************************************")
    #logging.info("*****************      PyVFit      *****************")
    #logging.info("****************************************************")

#
# From stackoverflow comment, how to make a method pickleable
# http://stackoverflow.com/questions/1816958/cant-pickle-type-instancemethod-when-using-pythons-multiprocessing-pool-ma/7309686#7309686
from copy_reg import pickle
from types import MethodType

def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

pickle(MethodType, _pickle_method, _unpickle_method)

class StarDataset(object):

    def __init__(self, datadir, fname, startframes, outname, nfrpos=200, 
                 frame_size=400, fill_nan=True, resize=None):
        self.datadir = datadir
        self.fname = fname
        self.startframes = startframes
        self.outname = outname
        self.nfrpos = nfrpos
        self.frame_size = frame_size
        self.fill_nan = fill_nan
        self.abcycles = []
        self.resize = resize
        #
        self.log_level = 20
        self.log_filename = 'local.log'
        self.log_filemode = 'w'
        self.log = self.init_logs()
        #
        # output frame size (may need resampling)
        if self.resize == None:
            self.out_frame_size = self.frame_size
        else:
            self.out_frame_size = self.frame_size*self.resize
        #
        ts = time.time()
        logging.info("Starting to set up the AB cycles")
        for startframe in self.startframes:
            tss = time.time()
            self.abcycles.append(ABCycle(self.datadir, self.fname, startframe, fill_nan = self.fill_nan))
            tss = time.time() - tss
            logging.info("  Initialized block starting at {0}, time {1}s".format(startframe,tss))
        tset = time.time() - ts
        logging.info("--> Setup of {0} AB-Cycles complete, time {1}s".format(len(self.startframes),tset))

    def init_logs(self):
        """
        Initialize the logging system.

        Parameters
        ----------
        log_levels: integer
            Level of logging. 10: logging.DEBUG, 20: logging.INFO, 30: logging.WARNING, 40: logging.ERROR, 50: logging.CRITICAL

        """
        log_format = '%(asctime)-15s %(name)-25s %(levelname)-8s %(message)s'

        log = logging.getLogger()
        log.setLevel(self.log_level)
        logfile = logging.FileHandler(filename=self.log_filename, mode=self.log_filemode)
        logfile.setFormatter(logging.Formatter(log_format))
        log.addHandler(logfile)

        logging.info("")
        logging.info("*****************************************************")
        logging.info("***********      LBTI Data Reduction      ***********")
        logging.info("*****************************************************")

        return log
            
    def do_framescube(self, multi=False):
        ts = time.time()
        logging.info("Starting the extraction of subcubes")
        self.framescube = np.zeros((len(self.startframes)*2*self.nfrpos,self.out_frame_size,self.out_frame_size))
        for i in range(len(self.startframes)):
            tss = time.time()
            if multi:
                self.abcycles[i].get_framescube_multiproc(frame_size=self.frame_size, resize=self.resize)
            else:
                self.abcycles[i].get_framescube(frame_size=self.frame_size, resize=self.resize)
            self.framescube[i*(2*self.nfrpos):i*(2*self.nfrpos)+2*self.nfrpos,:,:] = self.abcycles[i].framescube
            tss = time.time() - tss
            logging.info("  subcube extracted for block starting at {0}, time {1}s".format(self.startframes[i],tss))
        tscu = time.time() - ts
        logging.info("--> Extraction of {0} subcubes complete, time {1}s".format(len(self.startframes),tscu))
           

class ABCycle(object):
    
    def __init__(self, datadir, fname, startframe, fill_nan = False, nfrpos = 200, width = 600, 
                 height = 1024, xcen = 615, ylow = 340, dy = 425, plscale = 10.707):
        self.datadir = datadir
        self.fill_nan = fill_nan
        self.fname = fname
        self.startframe = startframe
        self.nfrpos = nfrpos
        self.width = width
        self.height = height
        self.xcen = xcen
        self.ylow = ylow
        self.dy = dy
        self.plscale = plscale
        self.filenames = self.__get_filenames()
        self.subcube, self.nanmasks, self.parangs = self.__fillcube()
        self.have_framescube = False
        
    # This method creates an array with all the file names
    def __get_filenames(self):
        #
        mynames = []
        for i in range(2*self.nfrpos):
            sfrnum = str(self.startframe+i).zfill(5)
            mynames.append(self.datadir+'/'+self.fname+'_'+sfrnum+'.fits')
        return mynames
    
    # This method reads all the files, performs the a-b nod subtraction
    # then returns a cube that contains the full set of a-b
    # sub-images of width self.width, height self.height and center self.xcen
    def __fillcube(self, simplesub = True):
        mycube = np.zeros((self.nfrpos, self.height, self.width))
        mynanmasks = []
        myparangs = np.zeros((self.nfrpos,2))
        for i in range(self.nfrpos):
            f1 = self.filenames[i]
            if simplesub:
                f2 = self.filenames[i+self.nfrpos]
            else:
                f2 = self.filenames[2*self.nfrpos-(i+1)]
            y1, y2, x1, x2 = self.__imgsec()
            i1 = myImage(f1, y1, y2, x1, x2, fill_nan = self.fill_nan)
            i2 = myImage(f2, y1, y2, x1, x2, fill_nan = self.fill_nan)
            mycube[i,:,:] = i1.data - i2.data
            mynanmasks.append(i1.nan_idx())
            mynanmasks.append(i2.nan_idx())
            myparangs[i,0] = i1.parang
            myparangs[i,1] = i2.parang
        #
        return mycube, mynanmasks, myparangs
    
    # 
    # This method is used to return a median filtered image of one
    #   one of the subcube planes. It is obsolete (but working).
    def median_filter(self, plane, kernel_size):
        return ssig.medfilt(self.subcube[plane,:,:],kernel_size=5)
    
    def __imgsec(self):
        #
        dx = int(self.width/2.)
        x1 = self.xcen-dx
        x2 = self.xcen+dx
        y1 = 0
        y2 = self.height
        return y1, y2, x1, x2
    
    #
    # These set of methods __detsign(), __get_centroid(), __get_subimages()
    #   are used by get_framescube() to center the positive and negative images
    #   of the star in a A-B nod image and extract subframes centered on the star.
    def __getsign(self,data):
        mytot = np.nansum(data)
        dfac = +1
        if mytot < 0.:
            dfac = -1
        return dfac
        
    def __get_centroid(self, data, y1, y2, x1, x2, dd):
        #
        d = data[y1:y2+1,x1:x2+1]
        y,x = np.mgrid[y1:y2:2*dd*1j,x1:x2:2*dd*1j]
        dtot = np.nansum(d*d)
        #print("x={0}, y={1}, d={2}".format(np.shape(x),np.shape(y),np.shape(d)))
        xc = np.nansum((x*d*d))/dtot
        yc = np.nansum((y*d*d))/dtot
        return xc, yc

    def __outmedian(self, data, radius):
        cen = np.shape(data)
        y,x = np.mgrid[0:cen[0]-1:cen[0]*1j,0:cen[1]-1:cen[1]*1j]
        wm = np.where((y-cen[0]/2.)*(y-cen[0]/2.)+(x-cen[1]/2.)*(x-cen[1]/2.) >= radius*radius)
        return np.nanmedian(data[wm])

    #
    # This procedure recenter the image to a float dx, dx
    #   note that this shift is assumed to be small.
    #   The image is resampled using a kernel interpolation scheme.
    def __do_recenter(self, image, dx, dy, kernel_size=3):
        y,x = np.mgrid[0:np.shape(image)[0]-1:np.shape(image)[0]*1j,
                       0:np.shape(image)[1]-1:np.shape(image)[1]*1j]
        x = x + dx
        y = y + dy
        dk = int(kernel_size/2.)
        dk = kernel_size
        xn = np.where(x < dk)
        xp = np.where(x > np.shape(image)[1]-1-dk)
        yn = np.where(y < dk)
        yp = np.where(y > np.shape(image)[0]-1-dk)
        x[xn] = dk
        x[xp] = np.shape(image)[1]-1-dk
        y[yn] = dk
        y[yp] = np.shape(image)[0]-1-dk
        return inpaint.sincinterp(image, x,  y, kernel_size=kernel_size )

    
    #def __get_subimages(self, plane = 0, subimsiz = 400, dd = 100, submed = True, 
    #                    resize = None):
    def __get_subimages(self, par):
        plane = par[0]
        subimsiz = par[1]
        dd = par[2]
        submed = par[3]
        resize = par[4]
        recenter = par[5]
        #
        # define subsection
        if resize == None:
            outsize = subimsiz
        else:
            outsize = resize * subimsiz
        subims = np.zeros((2,subimsiz,subimsiz))
        #outims = np.zeros((2,outsize,outsize))
        mydylist = [0, self.dy]
        #
        data = self.subcube[plane,:,:] 
        if submed:
            data = data - np.nanmedian(self.subcube[plane,:,:])
        #
        for i in range(len(mydylist)):
            mydy = mydylist[i]
            x1 = self.width/2.-dd
            x2 = self.width/2.+dd-1
            y1 = self.ylow+mydy-dd
            y2 = self.ylow+mydy+dd-1
            # 
            dfac = self.__getsign(data[y1:y2+1,x1:x2+1])
            #print("dy = {0} dfac = {1}".format(mydy,dfac))
            xc, yc = self.__get_centroid(dfac*data, y1, y2, x1, x2, dd)
            #print("   xc = {0} yc = {1}".format(xc,yc))
            if self.fill_nan:
                data[self.nanmasks[2*plane]] = np.nan
                data[self.nanmasks[2*plane+1]] = np.nan
            # extract data
            ixc = int(round(xc))
            iyc = int(round(yc))
            subims[i,:,:] = dfac*data[iyc-subimsiz/2:iyc+subimsiz/2,ixc-subimsiz/2:ixc+subimsiz/2]
            #
            # subtract median values from the edges of the map
            if submed:
                radius = 2./3.*(float(subimsiz)/2.)
                subims[i,:,:] = subims[i,:,:] - self.__outmedian(subims[i,:,:], radius)
            #
            # resample the images, recenter and interpolate.
            if resize == None:
                self.framescube[plane*2+i] = subims[i,:,:]
            else:
                # This procedure is described in
                # http://astrolitterbox.blogspot.de/2012/03/healing-holes-in-arrays-in-python.html
                myMask = (np.isnan(subims[i,:,:]))
                myMaskedImg = np.ma.array(subims[i,:,:], mask=myMask)
                NANMask =  myMaskedImg.filled(np.NaN)
                myBadArrays, my_num_BadArrays = snd.label(myMask)
                my_data_slices = snd.find_objects(myBadArrays)
                filled = inpaint.replace_nans(NANMask, 5, 0.5, 2, 'idw')
                zoom_filled = snd.zoom(filled, resize, order=3)
                if recenter:
                    zoom_filled = self.__do_recenter(zoom_filled, resize * (xc - float(ixc)), resize * (yc - float(iyc)))
                zoom_mask = snd.zoom(myMask, resize, order=0)
                myZoomFilled = np.ma.array(zoom_filled, mask=zoom_mask)
                self.framescube[plane*2+i] = myZoomFilled.filled(np.NaN)
            #
        #
    
    #
    # Non parallel version (executes each image sequentially)
    def get_framescube(self, frame_size=400, resize=None, recenter=False):
        #
        dd = 100
        submed = True
        if resize == None:
            self.framescube = np.zeros((self.nfrpos*2, frame_size, frame_size))
        else:
            self.framescube = np.zeros((self.nfrpos*2, resize*frame_size, resize*frame_size))
        self.have_framescube = True
        #
        for i in range(self.nfrpos):
            par = ( i, frame_size, dd, submed, resize, recenter)
            self.__get_subimages(par)

    #
    # Attempt at parallelization: 
    def get_framescube_multiproc(self, frame_size=400, resize=None, recenter=False):
        #
        dd = 100
        submed = True
        if resize == None:
            self.framescube = np.zeros((self.nfrpos*2, frame_size, frame_size))
        else:
            self.framescube = np.zeros((self.nfrpos*2, resize*frame_size, resize*frame_size))
        self.have_framescube = True
        #
        pool = Pool(processes=self.nfrpos)
        allpars=[]
        for i in range(self.nfrpos):
            allpars.append(( i, frame_size, dd, submed, resize))
        pool.map(self.__get_subimages, allpars)
        #results = [ \
        #     pool.apply_async(self.__get_subimages, ( i, frame_size, dd, submed, resize)) \
        #     for i in range(self.nfrpos)]
        pool.close()
        pool.join()
    
    
class myImage(object):
    
    def __init__(self, filename, y1, y2, x1, x2, fill_nan = False, maxvalid = 65000.):
        #
        self.fname = filename
        
        self.maxvalid = maxvalid
        self.y1 = y1
        self.y2 = y2
        self.x1 = x1
        self.x2 = x2
        self.data, self.parang = self.__readdata()
        #
        # Find star - does not work before nod subtraction
        #self.xstar, self.ystar = self.get_centroid()
        #
        self.fill_nan = fill_nan
        if self.fill_nan:
            self.__fill_nan()
            
    def __readdata(self):
        #
        h = aiof.open(self.fname)
        d = (h[0].data[self.y1:self.y2,self.x1:self.x2]).astype(np.float64)
        head = h[0].header
        parang = head['LBT_PARA']
        h.close()
        return d, parang
    
    def __fill_nan(self):
        idxn = np.where(self.data >= self.maxvalid)
        self.data[idxn] = np.nan 

    def nan_idx(self):
        idxn = np.where(self.data >= self.maxvalid)
        return idxn 

