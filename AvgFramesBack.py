#!/usr/bin/env python
"""
maptest.py: Compute phase maps from streams of images.
Algorithm:
1. compute (or get) period of image sequence and number of repeats
2. fold data
3. take mean across all sequences (reduction to one sequence or cycle)
4. compute phase  as a function of (x,y) within the map. We use an FFT
   for this, as it seems to be faster than anything else. 
The result is both plotted (matplotlib) and written to disk.

Includes test routine (call with '-t' flag) to generate a noisy image with
superimposed sinusoids in blocks with different phases.

To do:
1. sum phase maps (allow two inputs) to get absolute delay
2. (done)
3. data reduction in image to nxn blocks in (x,y)

5/12/2010 Paul B. Manis
UNC Chapel Hill
pmanis@med.unc.edu

"""

import sys, os
import numpy
import numpy as np
import scipy.signal
import scipy.ndimage
from skimage.morphology import reconstruction
import pyqtgraph as pg #added to deal with plottng issues TFR 11/13/15

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

# Keys are file #. Data are file number, stimulus type, number of reps, wavelength, attn, date, frequency, comment
DB = {0: ('000','SineAM_Stim_Camera',1, 610, 30.0, '14Jun16', 16.0, 'thinned skull')} 
DB[1] = ('001','SineAM_Stim_Camera', 1, 610, 15.0, '14Jun16', 32.0, 'thinned skull') 
DB[2] = ('002','SineAM_Stim_Camera', 1, 610, 15.0, '14Jun16', 32.0, 'thinned skull') 
DB[3] = ('003','SineAM_Stim_Camera', 1, 610, 15.0, '14Jun16', 32.0, 'thinned skull')  
DB[5] = ('005','Noise_Stimulation_Camera', 1, 610, 15.0, '14Jun16', 32.0, 'thinned skull')  
DB[7] = ('007','Noise_Stimulation_Camera', 1, 610, 15.0, '14Jun16', 32.0, 'thinned skull')  
DB[8] = ('008','Noise_Stimulation_Camera', 1, 610, 15.0, '14Jun16', 32.0, 'thinned skull')  
DB[9] = ('009','Noise_Stimulation_Camera', 1, 610, 15.0, '14Jun16', 32.0, 'thinned skull')  
DB[6] = ('006','Noise_Stimulation_Camera', 1, 610, 15.0, '14Jun16', 32.0, 'thinned skull')  

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
        self.period = 8.0 # sec
        self.framerate = 30. # Hz
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
        # done_deal=np.zeros((4,256,256),float)
        if options.binsize is not None:
            self.binsize = options.binsize
        if options.zbinsize is not None:
            self.zbinsize = options.zbinsize
        if argsin is not None:
            (options, args) = parser.parse_args(argsin)
        else:
            (options, args) = parser.parse_args()
        print 'DB keys', DB.keys()
        if options.tifffile is not None:
            self.tifffile = options.tifffile
        # divided=np.zeros((4,100,512,512),float)
        if options.tifffile is not None:
            n2 = self.tifffile + '_MMStack_Pos0.ome.tif'
            self.read_tiff_stack(filename=os.path.join(basepath, self.tifffile, n2))
            self.avg_over_trials()
        pylab.show() 


        return
        ######## stim1 is 0 to 30 and stim 2 is 120 to 150
        # if options.reps is not None:
        #     for nn in range(options.reps):
        #         self.load_file(nn)
        #         if nn == 0: #check the shape of imagedata and alter divided if necessary
        #             imshape = np.shape(self.imageData)
        #             divided=np.zeros((4,imshape[0],imshape[1],imshape[2]),float)

        #         # self.Image_Background()
        #         self.Image_Divided()
        #         # print 'divided', np.shape(self.divided)
        #         # self.divided= self.imageData
        #         divided[nn] = self.divided

        #     self.AvgFrames=np.mean(divided, axis=0)
        #     stim1=self.AvgFrames[0:19]
        #     stim2=self.AvgFrames[20:39]
        #     stim3=self.AvgFrames[40:59]
        #     stim4=self.AvgFrames[60:79]
        #     stim5=self.AvgFrames[80:99]
        #     pg.image(np.max(stim1,axis=0),title='Stimulus 1')
        #     pg.image(np.max(stim2,axis=0),title='Stimulus 2')
        #     pg.image(np.max(stim3,axis=0),title='Stimulus 3')
        #     pg.image(np.max(stim4,axis=0),title='Stimulus 4')
        #     pg.image(np.max(stim5,axis=0),title='Stimulus 5')
                
        #     pg.image(np.max(self.AvgFrames,axis=0),title='Max across all stimuli')      

                 
        return

    def read_tiff_stack(self, filename=None):
        """
        Read a diff stack into the data space.
        Using preset parameters, populates the frames, repetitions and frame times
        Rebins the image stack according to the parameters provided.
        Performe the analysis of the maps, and then plots using matplotlib
        
        Parameters
        ----------
        filename : str (default None)
        
        Returns
        -------
        Nothing
        """
        
        print filename
        if filename is None:
            raise ValueError('No file specitied')
        print 'Reading tifffile: %s' % filename
        imData = tf.imread(filename)
        # self.imageData=scipy.ndimage.gaussian_filter(self.imageData, sigma=[0,2,2], order=0,mode='reflect',truncate=4.0)
        # self.imageData=imData[0:6000]
        self.imageData=imData

        self.nFrames = self.imageData.shape[0]
        print 'image shape: ', self.imageData.shape[:]
        self.nrepetitions = np.floor(self.nFrames/(self.period * self.framerate))
        print 'n_reps:', self.nrepetitions
        print 'period:', self.period
        print 'framerate', self.framerate
        print 'nFrames:', self.nFrames
        # self.imageData = self.imageData[:self.nrepetitions*self.period*self.framerate]
        # self.adjust_image_data()
        self.avgimg = np.mean(self.imageData[5:], axis=0) # get mean image for reference later: average across all time
        # pg.image(self.imageData,title='raw image')
        # pg.image(self.avgimg,title = 'raw average image')
        self.Image_Background(imData)
        self.Image_Divided()
        if self.binsize > 1 or self.zbinsize > 1:
            nredx = int(sh[1]/self.binsize)
            nredy = int(sh[2]/self.binsize)
            nredz = int(self.imageData.shape[0]/self.zbinsize)
            self.imageData = self.bin_ndarray(self.imageData, new_shape=(nredz, nredx, nredy), operation='mean')
            if nredz > 1:
                beforeFrames = self.nFrames
                self.nFrames = self.imageData.shape[0]
                self.framerate = self.nFrames/(self.nrepetitions*self.period)
                self.times = np.arange(0, self.nFrames/self.framerate, 1.0/self.framerate)
        print 'Image Rebinned: '

        # print 'Read tiff file, original Image Info: '
        # self.print_image_info()
        # self.rebin_image()
        # self.clean_windowerrors(amount = self.skip)
        # self.analysis_fourier_map(target=1, mode=0)
        # self.plot_maps(mode=1, gfilter=self.gfilter)
        # mpl.show()
        return

    def Image_Background(self,im):
        self.background=[]
        # background = self.imageData[self.times<1]
        # pg.image(np.mean(background[1:],axis=0), title='average background ')

        self.background = np.mean(im[6100:6120],axis=0)
        # self.background = self.imageData[1]
        # pg.image(self.background,title='background image')
        return

    def Image_Divided(self):
        self.divided = (self.imageData-self.background)/self.background
        # self.divided=scipy.ndimage.gaussian_filter(self.divided, sigma=[0,1,1], order=0,mode='reflect')
        self.imageData = self.divided
        pg.image(self.divided[1:],title='divide image')
        #self.divided = np.mean(divided[self.times>=1],axis=0)
        #pg.image(subtracted, title='subtracted')
        # pg.image(self.divided,title='divided')    
        return


    def avg_over_trials(self):
        self.shaped = []
        single = int(self.period*self.framerate)
        self.shaped = np.reshape(self.divided,[self.nrepetitions,single,self.imageData.shape[1],self.imageData.shape[2]])
        self.shapedavg = np.mean(self.shaped[1:],axis=0)
        pg.image(self.shapedavg)

        self.stddev = np.std(self.shaped[1:],axis=0)

        pg.image(np.log(self.stddev))
        image77 = np.copy(self.shapedavg)
        image77[np.where(np.log(self.stddev)<-5)] = 0
        # pg.image(image77, 'image77')
        # pg.image(np.sum(image77[1:16],axis=0),title='stim 1 max locus')
        # pg.image(np.sum(image77[121:136],axis=0),title='stim 2 max locus')

        image1 = np.sum(image77[1:21],axis=0)
        seed1 = np.copy(image1)
        seed1[1:-1, 1:-1] = image1.min()
        rec1 = reconstruction(seed1,image1,method='dilation')

        image2 = np.sum(image77[121:141],axis=0)
        seed2 = np.copy(image2)
        seed2[1:-1, 1:-1] = image2.min()
        rec2 = reconstruction(seed2,image2,method='dilation')

        # print 'shape of shapedavg, ', np.shape(self.shapedavg)
        self.filtshapedavg=scipy.ndimage.gaussian_filter(self.shapedavg, sigma=[0,3,3], order=0,mode='reflect',truncate=4.0)
        stim1 = self.shapedavg[1:16]
        stim2 = self.shapedavg[121:136]
        mpl.figure(1)
        mpl.imshow(np.amax(self.shapedavg,axis=0),cmap=matplotlib.cm.gray)
        top1 = np.mean(np.sum(stim1,axis=0))+3*np.std(np.sum(stim1,axis=0))
        top2 = np.mean(np.sum(stim2,axis=0))+3*np.std(np.sum(stim2,axis=0))
        print 'top: ',(top1,top2)
        loci1 = np.where(np.sum(stim1,axis=0)>top1)
        loci2 = np.where(np.sum(stim2,axis=0)>top2)
        # print 'loci1 dim:', np.shape(loci1)
        mpl.hold('on')
        mpl.plot(loci1[1],loci1[0],'co')
        mpl.plot(loci2[1],loci2[0],'rx')
        # pg.image(self.shapedavg, title = 'average over repetitions')
        mpl.figure(2)
        mpl.subplot(2,3,1)
        mpl.imshow(np.amax(self.shapedavg,axis=0),cmap=matplotlib.cm.gray, interpolation='nearest')
        mpl.colorbar()
        mpl.title('Average image')
        mpl.subplot(2,3,2)
        # mpl.imshow(np.amax(self.filtshapedavg,axis=0),cmap=matplotlib.cm.gray,interpolation=None)
        # mpl.subplot(2,3,4)
        mpl.imshow(np.amax(stim1,axis=0),cmap=matplotlib.cm.Blues,interpolation=None)
        mpl.title('Stim 1 (integral over time)')
        # maxamp = np.amax(stim1)
        # shstim1.set_clim = (0.0, maxamp)
        mpl.colorbar()

        mpl.subplot(2,3,3)
        mpl.imshow(np.amax(stim2,axis=0),cmap=matplotlib.cm.Reds,interpolation=None)
        mpl.title('Stim 2 (integral over time')
        mpl.colorbar()

        mpl.subplot(2,3,4)
        # mpl.hold('on')
        # mpl.title('Stim1, Stim2 overlaid')
        mpl.imshow(np.sum(stim1,axis=0),cmap=matplotlib.cm.Blues,interpolation=None)
        mpl.colorbar()
        mpl.subplot(2,3,5)
        
        mpl.imshow(np.sum(stim2,axis=0),cmap=matplotlib.cm.Reds,interpolation=None,alpha=0.5)
        mpl.colorbar()
        
        # image1 = np.sum(stim1,axis=0)
        # seed1 = np.copy(image1)
        # seed1[1:-1, 1:-1] = image1.min()
        # rec1 = reconstruction(seed1,image1,method='dilation')

        # # mpl.subplot(2,3,6)
        # # mpl.imshow(image1-rec1,cmap=matplotlib.cm.Blues)

        # image2 = np.sum(stim2,axis=0)
        # seed2 = np.copy(image2)
        # seed2[1:-1, 1:-1] = image2.min()
        # rec2 = reconstruction(seed2,image2,method='dilation')

        # # mpl.subplot(2,3,3)
        # # mpl.imshow(image2-rec2,cmap=matplotlib.cm.Reds)
        # # mpl.figure(2)
        # # mpl.hold('on')

        # mpl.imshow(np.amax(self.shapedavg,axis=0),cmap=matplotlib.cm.gray, interpolation='nearest')
        # mpl.subplot(2,3,5)
        # mpl.imshow(image2-rec2,cmap=matplotlib.cm.Reds)
        # mpl.colorbar()
        # mpl.subplot(2,3,6)
        # mpl.imshow(image1-rec1,cmap=matplotlib.cm.Blues)
        # mpl.colorbar()
        # mpl.imshow(np.sum(stim1,axis=0),cmap=matplotlib.cm.Blues,interpolation=None)
        # mpl.imshow(np.sum(stim2,axis=0),cmap=matplotlib.cm.Reds,interpolation=None)
        # mpl.hold('off')
        # mpl.figure(3)
        # loci3 = np.where(image1>.8)
        # loci4 = np.where(image2>.8)
        # mpl.imshow(np.amax(self.shapedavg,axis=0),cmap=matplotlib.cm.gray)
        # mpl.hold('on')
        # mpl.plot(loci1[1],loci1[0],'ko')
        # mpl.plot(loci2[1],loci2[0],'rx')
        # mpl.title('std dev correction')
        
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