#!/usr/bin/env python

"""

maptest.py: Compute phase maps from streams of images.

Algorithm:

1. compute (or get) period of image sequence and number of repeats

2. fold data

3. take mean across all sequences (reduction to one sequence or cycle)


The result is plotted (matplotlib).



Includes test routine (call with '-t' flag) to generate a noisy image with

superimposed sinusoids in blocks with different phases.


5/12/2010 Paul B. Manis

UNC Chapel Hill

pmanis@med.unc.edu



"""



import sys, os

import numpy

import numpy as np

import scipy.signal
import scipy.io as sio

import scipy.ndimage 

from skimage.morphology import reconstruction

import pyqtgraph as pg #added to deal with plottng issues TFR 11/13/15
from skimage.feature import peak_local_max


import pickle

import matplotlib

import matplotlib.mlab as mlab

import matplotlib.pyplot as mpl

import pylab

from PyQt5 import QtGui

from skimage import feature

import tifffile as tf



import pyqtgraph.configfile as configfile

from pyqtgraph.metaarray import MetaArray



from optparse import OptionParser



app = pg.Qt.QtGui.QApplication([])



D = []

d = []

measuredPeriod = 6.444

binsize = 4

gfilt = 0



freqlist = np.logspace(0, 4, num=17, base=2.0)

fl = [3000*x for x in freqlist]

print 'fl:', fl



basepath = 'micromanager'

homedir = '/Volumes/TROPPDATA/data/'

basepath = os.path.join(homedir, basepath)



fn = '../camera_updown_20161017_7_MMStack_Pos0.ome.tif'



class testAnalysis():

    def __init__(self):

        global d

        global measuredPeriod

        global gfilt

        global binsize

        self.times = []

        self.upfile = []

        self.downfile = []

        self.avgimg = []

        self.imageData = []

        self.subtracted = []

        self.divided = []

        self.phasex = []

        self.phasey = []

        self.nPhases = 1

        self.nCycles = 1

        self.period = 10.0 # sec

        self.framerate = 20. # Hz

        self.binsize = 1

        self.zbinsize = 1



    def parse_and_go(self, argsin = None):

        global period

        global binsize

        global options

        global basepath

        parser=OptionParser() # command line options

        ##### parses all of the options inputted at the command line TFR 11/13/2015

        parser.add_option("-u", "--upfile", dest="upfile", metavar='FILE',

                          help="load the up-file")

        parser.add_option("-d", "--downfile", dest="downfile", metavar='FILE',

                          help="load the down-file")

        parser.add_option("-D", "--directory", dest="directory", metavar='FILE',

                          help="Use directory for data")

        parser.add_option("-t", "--test", dest="test", action='store_true',

                          help="Test mode to check calculations", default=False)

        parser.add_option("-p", '--period', dest = "period", default=4.25, type="float",

                          help = "Stimulus cycle period")

        parser.add_option("-c", '--cycles', dest = "cycles", default=0, type="int",

                          help = "# cycles to analyze")

        parser.add_option("-g", '--gfilter', dest = "gfilt", default=0, type="float",

                          help = "gaussian filter width")

        parser.add_option("-f", '--fdict', dest = "fdict", default=0, type="int",

                          help = "Use dictionary entry")

        parser.add_option("-T", "--tiff", dest="tifffile", default=fn, type="str",

                          help="load a tiff file")

        parser.add_option("-b", '--binning', dest = "binsize", default=self.binsize, type="int",

                          help = "bin reduction x,y")

        parser.add_option("-z", '--zbinning', dest = "zbinsize", default=self.zbinsize, type="int",

                          help = "bin reduction z")

        



        if argsin is not None:

            (options, args) = parser.parse_args(argsin)

        else:

            (options, args) = parser.parse_args()

      

        if options.tifffile is not None:

            self.tifffile = options.tifffile

        if options.binsize is not None:

            self.binsize = options.binsize

        if options.zbinsize is not None:

            self.zbinsize = options.zbinsize

        # divided=np.zeros((4,100,512,512),float)

        if options.tifffile is not None:

            n2 = self.tifffile + '_MMStack_Pos0.ome.tif'
            # matfile = sio.loadmat(os.path.join(basepath,self.tifffile,'updown_data.mat'))
            self.read_tiff_stack(filename=os.path.join(basepath, self.tifffile, n2))

            im = self.imageData
            pg.image(im, title='raw image')
            pg.image(np.std(im,axis=0),title='standard deviation')
            pg.plot(im[:,50,60],title='50,60')
            pg.plot(im[:,60,50],title='60,50')
            print 'image shape: ',np.shape(im)
            # self.binReps()
            # self.avg_over_trials()
            # self.binToStim()
            # self.Image_Background()

            # self.Image_Divided()

            

        pylab.show() 

    



        return



    def read_tiff_stack(self, filename=None):

        """

        Read a diff stack into the data space.

        Using preset parameters, populates the frames, repetitions and frame times
        """

          

        print filename

        if filename is None:

            raise ValueError('No file specitied')

        print 'Reading tifffile: %s' % filename

        self.imageData = tf.imread(filename)

        self.nFrames = self.imageData.shape[0]

        print 'image shape: ', self.imageData.shape[:]

        self.nrepetitions = np.floor(self.nFrames/(self.period * self.framerate))

        print 'n_reps:', self.nrepetitions

        print 'period:', self.period

        print 'framerate', self.framerate

        print 'nFrames:', self.nFrames

        self.stimtime = (1, 6)  #stimulus times in seconds

        return
    
    def binReps(self):
        
        sh = np.shape(self.imageData)
        framesperiod = int(sh[0]/self.nrepetitions)
        print 'Frames per period:', framesperiod
        reps = np.zeros([self.nrepetitions,framesperiod,sh[1],sh[2]],float)
        for i in range(int(self.nrepetitions)):
            start = i*framesperiod
            finish = (i+1)*framesperiod
            print 'start, finish: ', (i, start,finish)
            reps[i] = self.imageData[start:finish,::]
            # pg.image(reps[i],title=str(i))
            background = np.mean(np.mean(reps[i,0:18,:,:],axis=1),axis=0)
            reps[i] = (reps[i]-background)/background
        print 'size of reps: ', np.shape(reps)
        self.AvgReps = np.mean(reps[1:],axis=0)
        # mpl.figure(1)
        # for i in range(10):
        #     mpl.subplot(2,5,i+1)
        #     secstart=i*20
        #     secstop=(i+1)*20-1
            # mpl.imshow(np.mean(self.AvgReps[secstart:secstart,:,:],axis=0))
        pg.image(self.AvgReps,title='averaged across reps')
        pg.image(np.std(self.AvgReps,axis=0),title='std across reps')
        pg.image(np.max(self.AvgReps[10:20],axis=0),title='stim1')
        pg.image(np.max(self.AvgReps[60:70],axis=0),title='stim 2')
        # background = np.mean(self.AvgReps[0:18])
        # self.divided = (self.AvgReps-background)/background
        # pg.image(self.divided,title='averages across reps and divided by first second')
        
        return

    def binToStim(self):
        sh=np.shape(self.imageData)
        print 'shape: ', sh
        stimlen=len(self.stimtime)
        stim = np.zeros([stimlen,self.framerate+1,sh[1],sh[2]],float)
        for i in range(stimlen):
            framestart = self.stimtime[i]*self.framerate-1
            framestop = (self.stimtime[i] + 1)* self.framerate
            print 'framestart, framestop: ', (framestart,framestop)
            stim[i] = self.imageData[framestart:framestop]
            bkstop= int(framestart - 2)
            bkstart = int(framestart - 5)
            self.Image_Background(bkstart,bkstop)
            self.Image_Divided(stim[i])
            #get rid of the outliers
            stim[i] = self.removeOutliers(stim[i])
        mpl.figure(1)
        mpl.subplot(3,2,1)
        mpl.imshow(np.mean(stim[0],axis=0))
        mpl.title('stim1 mean')
        mpl.subplot(3,2,3)
        mpl.imshow(np.sum(stim[0],axis=0))
        mpl.title('stim1 sum')
        mpl.subplot(3,2,5)
        mpl.imshow(np.amax(stim[0],axis=0))
        mpl.title('stim1 max')
        mpl.subplot(3,2,2)
        mpl.imshow(np.mean(stim[1],axis=0))
        mpl.title('stim2 mean')
        mpl.subplot(3,2,4)
        mpl.imshow(np.sum(stim[1],axis=0))
        mpl.title('stim2 sum')
        mpl.subplot(3,2,6)
        mpl.imshow(np.amax(stim[1],axis=0))
        mpl.title('stim2 max')
        mpl.figure(2)
        mpl.subplot(2,1,1)
        coordinates0 = peak_local_max(np.amax(stim[0],axis=0), min_distance=20)
        mpl.imshow(np.amax(stim[0],axis=0),origin='lower')
        mpl.colorbar()
        mpl.title('stim1 max')
        mpl.hold('on')
        print 'coordinates0', coordinates0
        mpl.plot(coordinates0[0],coordinates0[1],'ko')

        mpl.subplot(2,1,2)
        coordinates1 = peak_local_max(np.amax(stim[1],axis=0), min_distance=20)
        mpl.imshow(np.amax(stim[1],axis=0),origin='lower')
        mpl.colorbar()
        mpl.title('stim2 max')
        mpl.hold('on')
        print 'coordinates1', coordinates1
        mpl.plot(coordinates1[0],coordinates1[1],'ko')

            
        return

    def removeOutliers(self,stim):
        stimavg = np.mean(stim,axis=0)
        stimstd = np.std(stim,axis=0)
        stim[np.where(stim>stimavg+2*stimstd)]=0
        stim[np.where(stim<stimavg-2*stimstd)]=0
        return stim

    def Image_Background(self,start,stop):
        print 'start, stop: ', (start,stop)
        self.background=[]

        # background = self.imageData[self.times<1]

        # pg.image(np.mean(background[1:],axis=0), title='average background')



        self.background = np.mean(self.imageData[start:stop],axis=0)

        

        # pg.image(self.background, title='background')# self.background = self.imageData[1]

        # pg.image(self.background,title='background image')

        return



    def Image_Divided(self,im):
        self.divided=[]
        self.divided = (im-self.background)/self.background

        # self.divided=scipy.ndimage.gaussian_filter(self.divided, sigma=[0,1,1], order=0,mode='reflect')

        # pg.image(self.divided,title='divide image')

        #self.divided = np.mean(divided[self.times>=1],axis=0)

        #pg.image(subtracted, title='subtracted')

        # pg.image(self.divided,title='divided')    

        return





    def avg_over_trials(self):

        self.shaped = []

        single = int(self.period*self.framerate)

        self.shaped = np.reshape(self.imageData,[self.nrepetitions,single,self.imageData.shape[1],self.imageData.shape[2]])

        self.imageData = np.mean(self.shaped,axis=0)

        # pg.image(self.imageData, title='folded and averaged')



        

        return



    def bin_ndarray(self, ndarray, new_shape, operation='sum'):

        """

        Bins an ndarray in all axes based on the target shape, by summing or

            averaging.

        Number of output dimensions must match number of input dimensions.

        Example

        -------

        >>> m = np.arange(0,100,1).reshape((10,10))

        >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')

        >>> print(n)

        [[ 22  30  38  46  54]

         [102 110 118 126 134]

         [182 190 198 206 214]

         [262 270 278 286 294]

         [342 350 358 366 374]]

        

        found at:

        https://gist.github.com/derricw/95eab740e1b08b78c03f

        aka derricw/rebin_ndarray.py

        """

        if not operation.lower() in ['sum', 'mean', 'average', 'avg']:

            raise ValueError("Operation {} not supported.".format(operation))

        if ndarray.ndim != len(new_shape):

            raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,

                                                               new_shape))

        compression_pairs = [(d, c//d) for d, c in zip(new_shape,

                                                       ndarray.shape)]

        flattened = [l for p in compression_pairs for l in p]

        ndarray = ndarray.reshape(flattened)

        for i in range(len(new_shape)):

            if operation.lower() == "sum":

                ndarray = ndarray.sum(-1*(i+1))

            elif operation.lower() in ["mean", "average", "avg"]:

                ndarray = ndarray.mean(-1*(i+1))

        return ndarray  

if __name__ == "__main__":

    ta=testAnalysis()  # create instance (for debugging)

    ta.parse_and_go(sys.argv[1:])
    app.exec_()
    mpl.show()


    

    