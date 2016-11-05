__author__ = 'ltesti - 3 Nov 2016'

from __future__ import division, print_function

import numpy as np
import scipy.signal as ssig

#from astroquery.irsa import Irsa
#import astropy.units as u
#import astropy.coordinates as coord
import astropy.io.fits as aiof

import matplotlib.pyplot as plt

#import aplpy

import os

class StarDataset(object):
    
    def __init__(self, datadir, fname, startframes, outname, nfrpos=200, frame_size=400, fill_nan=True):
        self.datadir = datadir
        self.fname = fname
        self.startframes = startframes
        self.outname = outname
        self.nfrpos = nfrpos
        self.frame_size = frame_size
        self.fill_nan = fill_nan
        self.abcycles = []
        for startframe in self.startframes:
            self.abcycles.append(ABCycle(self.datadir, self.fname, startframe, fill_nan = self.fill_nan))
            
    def do_framescube(self):
        self.framescube = np.zeros((len(self.startframes)*2*self.nfrpos,self.frame_size,self.frame_size))
        for i in range(len(self.startframes)):
            self.abcycles[i].get_framescube(frame_size=self.frame_size)
            self.framescube[i*(2*self.nfrpos):i*(2*self.nfrpos)+2*self.nfrpos,:,:] = self.abcycles[i].framescube
            

class ABCycle(object):
    
    def __init__(self, datadir, fname, startframe, fill_nan = False, nfrpos = 200, width = 600, height = 1024, xcen = 615, ylow = 340, dy = 425, plscale = 10.707):
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
    
    def __get_subimages(self, plane = 0, subimsiz = 400, dd = 100, submed = True):
        #
        # define subsection
        subims = np.zeros((2,subimsiz,subimsiz))
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
            if submed:
                radius = 2./3.*(float(subimsiz)/2.)
                subims[i,:,:] = subims[i,:,:] - self.__outmedian(subims[i,:,:], radius)
        #
        return subims[0,:,:],subims[1,:,:]
    
    def get_framescube(self, frame_size=400):
        #
        self.framescube = np.zeros((self.nfrpos*2, frame_size, frame_size))
        for i in range(self.nfrpos):
            self.framescube[i*2], self.framescube[i*2+1] = self.__get_subimages(plane = i, subimsiz = frame_size)
    
    
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

