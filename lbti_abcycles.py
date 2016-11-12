from __future__ import division, print_function

__author__ = 'ltesti - 11 Nov 2016'

import numpy as np
import scipy.signal as ssig
import scipy.ndimage as snd

from multiprocessing import Pool
#import dill

import astropy.io.fits as aiof

import matplotlib.pyplot as plt

#import aplpy

import time
import os

# logging
import logging

# local package functions
import inpaint 
import image_functions as imfu


# # trying another one from:
# # http://stackoverflow.com/questions/8804830/python-multiprocessing-pickling-error/24673524#24673524
# def run_dill_encoded(payload):
#     fun, args = dill.loads(payload)
#     return fun(*args)

# def apply_async(pool, fun, args):
#     payload = dill.dumps((fun, args))
#     return pool.apply_async(run_dill_encoded, (payload,))

# def map(pool, fun, args):
#     payload = dill.dumps((fun, args))
#     return pool.map(run_dill_encoded, (payload,))
           

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
        self.has_framescube = False
        self.has_subcube = False

    ############################################################################
    #    Section that reads and fills the cube
    ############################################################################
        
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

    ############################################################################
    #    Section that extracts the A-B frames
    ############################################################################
    
    #
    # Extract and resample nod images: 
    def get_framescube(self, frame_size=400, resize=None, recenter=False, multi=False, nproc=10):
        #
        dd = 150
        submed = True
        #
        # define the frame and resampling factor
        if resize == None:
            rsfac = 1.
        else:
            rsfac = resize
        self.framescube = np.zeros((self.nfrpos*2, rsfac*frame_size, rsfac*frame_size))
        self.has_framescube = True

        # (x,y) location of stars in A and B
        x = [rsfac*(self.width/2.-dd),rsfac*(self.width/2.+dd)-1]
        y = [[rsfac*(self.ylow-dd),rsfac*(self.ylow+dd)-1],\
            [rsfac*(self.ylow+self.dy-dd),rsfac*(self.ylow+self.dy+dd)-1]]
        
        pars=[]
        for plane in range(self.nfrpos):
            par = (self.subcube[plane,:,:],(x,y),frame_size,rsfac,submed)
            pars.append(par)

        if multi:
            self.__run_multiproc_framescube(pars, nproc)
        else:
            self.__run_framescube(pars)

    #
    # Extract and resample nod images: 
    def get_framescube_blkshift(self, frame_size=400, resize=None, recenter=False, multi=False, nproc=10):
        #
        dd = 100
        submed = True
        #
        # define the frame and resampling factor
        if resize == None:
            rsfac = 1.
        else:
            rsfac = resize
        self.framescube = np.zeros((self.nfrpos*2, rsfac*frame_size, rsfac*frame_size))
        self.has_framescube = True

        # (x,y) location of stars in A and B
        x = [(self.width/2.-dd),(self.width/2.+dd)-1]
        y = [[(self.ylow-dd),(self.ylow+dd)-1],\
            [(self.ylow+self.dy-dd),(self.ylow+self.dy+dd)-1]]
        
        pars_cen=[]
        for plane in range(self.nfrpos):
            #par = (self.subcube[plane,:,:],(x,y),frame_size,rsfac,submed)
            par = (self.subcube[plane,:,:],(x,y),frame_size,rsfac,submed)
            pars_cen.append(par)

        if multi:
            self.__run_multiproc_framescube_blkshift(pars_cen, nproc)
        else:
            self.__run_framescube_blkshift(pars_cen)

    #
    # No multiprocessing version
    def __run_framescube(self, pars):
        for plane in range(self.nfrpos):
            self.framescube[2*plane:2*plane+2,:,:] = imfu.get_subimage(pars[plane])

    #
    # multiprocessing version
    def __run_multiproc_framescube(self, pars, nproc):
        pool = Pool(processes=nproc)

        results = pool.map(imfu.get_subimage, pars)

        for plane in range(self.nfrpos):
            self.framescube[2*plane:2*plane+2,:,:] = results[plane]  

        pool.close()
        pool.join()      

    #
    # No multiprocessing version
    def __run_framescube_blkshift(self, pars):
        shiftcen=[]
        for plane in range(self.nfrpos):
            shiftcen.append(imfu.block_sign_centroid(pars[plane]))

        pars_ext = imfu.get_pars_ext(shiftcen,pars)
        
        # pars_ext=[]
        # for plane in range(self.nfrpos):
        #     sign = (shiftcen[0][0],shiftcen[1][0])
        #     center = ((int(round(shiftcen[0][1])),int(round(shiftcen[0][2]))), (int(round(shiftcen[1][1])),int(round(shiftcen[1][2]))))
        #     shift = ((-(shiftcen[0][1]-center[0][0]),-(shiftcen[0][0]-center[0][1])),\
        #              (-(shiftcen[1][1]-center[1][0]),-(shiftcen[1][0]-center[1][1])))
        #     par = (pars[plane][0],(shiftcen[i][0]),pars[plane][2],pars[plane][3],pars[plane][4])
        #     pars_ext.append(par)

        for plane in range(self.nfrpos):
            self.framescube[2*plane:2*plane+2,:,:] = imfu.get_subimage_blkshift(pars_ext[plane])

    #
    # multiprocessing version
    def __run_multiproc_framescube_blkshift(self, pars, nproc):
        pool = Pool(processes=nproc)

        shiftcen = pool.map(imfu.block_sign_centroid, pars)

        pars_ext = imfu.get_pars_ext(shiftcen,pars)

        results = pool.map(imfu.get_subimage_blkshift, pars_ext)

        for plane in range(self.nfrpos):
            self.framescube[2*plane:2*plane+2,:,:] = results[plane]  

        pool.close()
        pool.join()      

    ############################################################################

    #
    # subtract median image and create subcube
    def do_subcube(self,medianimage):
        self.subcube = self.framescube - medianimage
        self.has_subcube = True

    ############################################################################
    #    Section that rotates the subtracted cube
    ############################################################################

    #
    # No multiprocessing version
    def __run_rotatecube(self, pars):
        for plane in range(self.nfrpos):
            self.abrotsubcube[2*plane,:,:] = imfu.rotima(pars[2*plane]) 
            self.abrotsubcube[2*plane+1,:,:] = imfu.rotima(pars[2*plane+1]) 

    #
    # multiprocessing version
    def __run_multiproc_rotateccube(self, pars, nproc):
        pool = Pool(processes=nproc)

        results = pool.map(imfu.rotima, pars)

        for plane in range(self.nfrpos):
            self.abrotsubcube[2*plane,:,:] = results[2*plane]  
            self.abrotsubcube[2*plane+1,:,:] = results[2*plane+1]  

        pool.close()
        pool.join()      

    #
    # rotate subcubes
    def do_rotate_subcube(self, multi=False, nproc=10):
        xy = np.shape(self.framescube)
        self.abrotsubcube = np.zeros((xy[0], xy[1], xy[2]))

        pars=[]
        for plane in range(self.nfrpos):
            par = (self.subcube[2*plane],-self.parangs[plane,0])#,reshape=False)
            pars.append(par)
            par = (self.subcube[2*plane+1],-self.parangs[plane,1])#,reshape=False)
            pars.append(par)

        if multi:
            self.__run_multiproc_rotateccube(pars, nproc)
        else:
            self.__run_rotatecube(pars)

    ############################################################################

    
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

