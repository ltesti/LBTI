from __future__ import division, print_function

__author__ = 'ltesti - 3 Nov 2016'

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
from lbti_abcycles import ABCycle


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


class StarDataset(object):

    def __init__(self, datadir, fname, startframes, outname, nfrpos=200, 
                 frame_size=400, fill_nan=True, resize=None,
                 xcen = 615, ylow = 340, dy = 425, plscale = 10.707):
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
        self.has_medianimage = False
        self.has_framescube = False
        self.has_subcube = False
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
            self.abcycles.append(ABCycle(self.datadir, self.fname, startframe, \
                                         fill_nan = self.fill_nan, nfrpos=self.nfrpos, width = xcen*2., \
                                         xcen = xcen, ylow = ylow, dy = dy, plscale = plscale))
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
            
    def do_framescube(self, multi=False, recenter=False, nproc=10):
        ts = time.time()
        logging.info("Starting the extraction of subcubes")
        self.framescube = np.zeros((len(self.startframes)*2*self.nfrpos,self.out_frame_size,self.out_frame_size))
        for i in range(len(self.startframes)):
            tss = time.time()
            #self.abcycles[i].get_framescube(frame_size=self.frame_size, resize=self.resize, recenter=recenter,
            #                                multi=multi, nproc=nproc)
            self.abcycles[i].get_framescube_blkshift(frame_size=self.frame_size, resize=self.resize, recenter=recenter,
                                            multi=multi, nproc=nproc)
            # if multi:
            #     self.abcycles[i].get_framescube_multiproc(frame_size=self.frame_size, resize=self.resize, recenter=recenter, nproc=nproc)
            # else:
            #     self.abcycles[i].get_framescube(frame_size=self.frame_size, resize=self.resize, recenter=recenter)
            self.framescube[i*(2*self.nfrpos):i*(2*self.nfrpos)+2*self.nfrpos,:,:] = self.abcycles[i].framescube
            tss = time.time() - tss
            logging.info("  subcube extracted for block starting at {0}, time {1}s".format(self.startframes[i],tss))
        tscu = time.time() - ts
        logging.info("--> Extraction of {0} subcubes complete,d time {1}s".format(len(self.startframes),tscu))
        self.has_framescube=True

    def do_subcube(self):
        ts = time.time()
        logging.info("Starting median subtraction")
        if self.has_framescube:
            self.medianimage = np.median(self.framescube,axis=0)
            self.has_medianimage = True
            for i in range(len(self.startframes)):
                self.abcycles[i].do_subcube(self.medianimage)
            self.has_subcube = True
        else:
            print("Nothing done: please run do_framescube() method first!")
        tsub = time.time() - ts
        logging.info("--> Subtraction of {0} subcubes completed, time {1}s".format(len(self.startframes),tsub))

    def do_derotate_cube(self, multi=False, nproc=10, subcube=True):
        ts = time.time()
        logging.info("Starting subcube rotation")
        if subcube:
            if self.has_subcube:
                self.rotsubcube = np.zeros((len(self.startframes)*2*self.nfrpos,self.out_frame_size,self.out_frame_size))
                for i in range(len(self.startframes)):
                    trr = time.time()
                    self.abcycles[i].do_rotate_subcube(multi=multi, nproc=25)
                    self.rotsubcube[i*(2*self.nfrpos):i*(2*self.nfrpos)+2*self.nfrpos,:,:] = self.abcycles[i].abrotsubcube
                    trr = time.time() - trr
                    logging.info("  subcube rotated for block starting at {0}, time {1}s".format(self.startframes[i],trr))
            else:
                print("Nothing done: please run do_subcube() method first!")
        else:
            print("Not yet implemented!")
        trot = time.time() - ts
        logging.info("--> Rotation of {0} subcubes complete, time {1}s".format(len(self.startframes),trot))

    def do_write_allsubframes(self,framesdir='./'):
        ts = time.time()
        logging.info("Starting write subframes")
        nstart = 0
        for i in range(len(self.startframes)):
            twr = time.time()
            nstart = nstart + self.abcycles[i].writeframes(nstart,framesdir,self.outname)
            twr = time.time() - twr
            logging.info("  frames written for block starting at {0}, time {1}s".format(self.startframes[i],twr))
        tsub = time.time() - ts
        logging.info("--> Writing subframes of {0} subcubes completed, time {1}s".format(len(self.startframes),tsub))

           
