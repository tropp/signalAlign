#!/usr/bin/env python
"""
map_fft_phase2.py: Compute phase maps from streams of images.

Algorithm:
1. compute (or get) period of image sequence and number of repeats
2. bin data x,y to provide a kind of averaging.
3. median filter to remove low-frequency baseline fluctuations.
4. Compute fft 
4. compute phase  as a function of (x,y) within the map. We use an FFT
   for this, as it seems to be faster than anything else. 
The result is both plotted (matplotlib) and written to disk.

Includes test routine (call with '-t' flag) to generate a noisy image with
superimposed sinusoids in blocks with different phases.

Includes computation of simple deltaF/F for fluorescence measurements as well.

To do:
1. sum phase maps (allow two inputs) to get absolute delay

Flags:

Usage: map_fft_phase2.py [options]

Options:
  -h, --help            show this help message and exit
  -t, --test            Test mode to check calculations
  --dF/F                analysis mode set to fft (default) vs. dF/F
  -u FILE, --upfile=FILE
                        load the up-file
  -d FILE, --downfile=FILE
                        load the down-file
  -D FILE, --directory=FILE
                        Use directory for data
  -T TIFFFILE, --tiff=TIFFFILE
                        load a tiff file
  -p PERIOD, --period=PERIOD
                        Stimulus cycle period
  -c CYCLES, --cycles=CYCLES
                        # cycles to analyze
  -b BINSIZE, --binning=BINSIZE
                        bin reduction x,y
  -z ZBINSIZE, --zbinning=ZBINSIZE
                        bin reduction z
  -g GFILT, --gfilter=GFILT
                        gaussian filter width
  -f FDICT, --fdict=FDICT
                        Use dictionary entry
  -P FREQPERIOD, --freqperiod=FREQPERIOD
                        Set Frequency period (seconds)
  -F FREQINFO, --frequencies=FREQINFO
                        Set Frequency settings as string. Example: '[4,32,8]'
                        start=4, end=32, nfreqs = 8
  --threshold=THRESHOLD
                        dFF threshold for map

5/12/2010 Paul B. Manis
updated 10/22/2016 pbm

UNC Chapel Hill
pmanis@med.unc.edu

"""
from __future__ import print_function  # use python 3 print statement format

import sys, os
import numpy as np
import scipy.signal
import scipy.ndimage as ndimage
from skimage import restoration
import skimage.feature as skif
import scipy.io

import pickle
import matplotlib
import matplotlib.pyplot as mpl
from matplotlib import cm # color map
import tifffile as tf
import pyqtgraph as pg
from PyQt4 import QtGui
#from pyqtgraph.metaarray import MetaArray

from optparse import OptionParser

import matplotlib.colors as mplcolor
app = pg.Qt.QtGui.QApplication([])



global win, roi
basepath = 'micromanager'
#homedir = '/media/ropphouse/TROPPDATA/data'
homedir = '/Volumes/TROPPDRIVE'
# homedir = '/Volumes/TROPPDATA/data'
# homedir = '/Users/tessajonneropp/Documents/ABR_data'
# basepath = homedir
basepath = os.path.join(homedir, basepath)

class FFTImageAnalysis():
    """
    Provide routines for analyzing image stacks by extracting phase and amplitude 
    across the image plane from the time data series
    """
    def __init__(self, layout, winsize, d=[], measurePeriod=8.0, binsize=1, gfilter=0):
        self.layout = layout
        self.winsize = winsize
        self.d = d
        self.mode = False # mode = False -> FFT analysis, mode = True -> dF/F analysis
        self.period = measurePeriod # sec
        self.freqperiod = 4.
        self.framerate = 20. # Hz
        self.nPhases = 6  # for test stimulus
        self.nrepetitions = 50 # number of repeats of the presentation
        self.imageSize = [128, 128] # image size x : y, used for testing
        self.bkgdIntensity = 1200. # mean intensity of test signal
        self.threshold = 0.25
        self.frequencies = None # '[4,32,8]'
        self.freqinfo = None
        self.gfilter = gfilter
        self.binsize = binsize
        self.zbinsize = 1
        self.times = [] # time points for average waveform
        self.timebase = [] # time points for ALL of the input images (whole sequence)
        self.upfile = []
        self.downfile = []
        self.avgimg = []
        self.imageData = []
        self.phasex = []
        self.phasey = []
        self.rejectionEnabled = False # do not (or do) reject trials based on SD of data
        self.removeFirstSequence = True
        self.DF = []
        self.stdimg = []
        self.dir = 'up'
        self.vars = {}
        self.background = []
        self.mxplts =[]
        self.subback = []
        self.rawimage = []
        
    def parse_commands(self, argsin=None):
        """
        Parse the command line options
        
        Parameters
        ----------
        argsin : arguments passed on the command line
        
        Returns
        -------
        Nothing
        """
        parser=OptionParser() # command line options
        parser.add_option("-t", "--test", dest="test", action='store_true',
                          help="Test mode to check calculations", default=False)
        parser.add_option("--dF/F", dest="mode", action='store_true',
                          help="analysis mode set to fft (default) vs. dF/F", default=False)
        parser.add_option("-u", "--upfile", dest="upfile", metavar='FILE',
                          help="load the up-file")
        parser.add_option("-d", "--downfile", dest="downfile", metavar='FILE',
                          help="load the down-file")
        parser.add_option("-D", "--directory", dest="directory", metavar='FILE',
                          help="Use directory for data")
        parser.add_option("-T", "--tiff", dest="tifffile", default=None, type="str",
                          help="load a tiff file")
        parser.add_option("-p", '--period', dest = "period", default=self.period, type="float",
                          help = "Stimulus cycle period")
        parser.add_option("-c", '--cycles', dest = "cycles", default=self.nrepetitions, type="int",
                          help = "# cycles to analyze")
        parser.add_option("-b", '--binning', dest = "binsize", default=self.binsize, type="int",
                          help = "bin reduction x,y")
        parser.add_option("-z", '--zbinning', dest = "zbinsize", default=self.zbinsize, type="int",
                          help = "bin reduction z")
        parser.add_option("-g", '--gfilter', dest = "gfilt", default=self.gfilter, type="float",
                          help = "gaussian filter width")
        parser.add_option("-f", '--fdict', dest = "fdict", default=None, type="int",
                          help = "Use dictionary entry")
        parser.add_option("-P", '--freqperiod', dest = "freqperiod", default=1.0, type="float",
                          help = "Set Frequency period (seconds)")
        parser.add_option("-F", '--frequencies', dest = "freqinfo", default='[4,32,8]', type="str",
                          help = "Set Frequency settings as string. Example: '[4,32,8]' start=4, end=32, nfreqs = 8")
        parser.add_option('--threshold', dest = "threshold", default=self.threshold, type="float",
                          help = "dFF threshold for map")
        if argsin is not None:
            (options, args) = parser.parse_args(argsin)
        else:
            (options, args) = parser.parse_args()

        if options.mode is not None:
            self.mode = options.mode
        if options.period is not None:
            self.period = options.period
        if options.freqperiod is not None:
            self.freqperiod = options.freqperiod
        if options.freqinfo is not None:
            self.freqinfo = options.freqinfo
        if options.cycles is not None:
            self.nrepetitions = options.cycles
        if options.binsize is not None:
            self.binsize = options.binsize
        if options.zbinsize is not None:
            self.zbinsize = options.zbinsize
        if options.gfilt is not None:
            self.gfilter = options.gfilt
        if options.tifffile is not None:
            self.tifffile = options.tifffile
        if options.threshold is not None:
            self.threshold = options.threshold
       
        print ('Freqperiod: ', self.freqperiod    )
        print ('Mode: ', self.mode)
        
        
        if options.tifffile is not None:
            n2 = self.tifffile + '_MMStack_Pos0.ome.tif'
            self.read_tiff_stack(filename=os.path.join(basepath, self.tifffile, n2))
        

    def read_tiff_stack(self, filename=None):
        global win, roi
        """
        Read a tiff stack into the data space.
        Using preset parameters, populates the frames, repetitions and frame times
        Rebins the image stack according to the parameters provided.
        Performe the analysis of the maps, and then plots using matplotlib
        
        If a matlab (.mat) file is associated with the stack, it will be read
        to obtain metadata as well, overriding default or passed parameters.
        Parameters
        ----------
        filename : str (default None)
        
        Returns
        -------
        Nothing
        """
        
        print ('Reading tifffile: %s' % filename)
        if filename is None:
            raise ValueError('No file specitied')
        self.read_matfile(filename)
        if self.vars is not None:
            self.framerate = float(self.vars['cameraFreq'])
            self.period = float(self.vars['sweeptimesec'])
            self.nrepetitions = int(self.vars['nsweeps'])
        
        self.imageData = tf.imread(filename)
        self.rawimage = self.imageData
        

    def ROIdata(self):
        global win, roi, img, rawdata
        pg.mkQApp()

        win = pg.GraphicsLayoutWidget()
        win.setWindowTitle('image with contours and ROI')
        win.show()
        # A plot area (ViewBox + axes) for displaying the image
        p1 = win.addPlot()

        # Item for displaying image data
        img = pg.ImageItem()
        p1.addItem(img)
        print('size of img',np.shape(self.imageData))
        # Custom ROI for selecting an image region
        roi = pg.ROI([-8, 14], [6, 5])
        roi.addScaleHandle([0.5, 1], [0.5, 0.5])
        roi.addScaleHandle([0, 0.5], [0.5, 0.5])
        p1.addItem(roi)
        roi.setZValue(10)  # make sure ROI is drawn above image
        pg.setConfigOptions(imageAxisOrder='row-major')
        rawdata=np.mean(self.rawimage[1:],axis=0)
        img.setImage(rawdata)
        print('got to this line')
        



    def imgAnal(self):
        global win, roi
        self.nFrames = self.imageData.shape[0]

        self.avgimg = np.mean(self.imageData, axis=0) # get mean image for reference later: average across all time
        self.times = np.arange(0, self.nFrames/self.framerate, 1.0/self.framerate)

        self.normal_image()
        self.rebin_image()

    def normal_image(self):
        background = []
        background = np.mean(self.imageData[1:int(self.framerate-1)],axis=0)
        self.subback = (self.imageData-background)/background
        self.imageData = self.subback
        pg.image(self.subback[1:],title='background subtracted')
        # pg.image(background,title='background')
        # pg.image(self.subback,title='imageData with background subtraction')
        # pg.image(background,title='background')
    
    def read_matfile(self, filename):
        """
        Read a matlab file associated with the data file - assumes that the
        'PARS' data structure contains some specific data we need for analysis.
        
        Parameters
        ----------
        filename : str (default None)
        
        Returns
        -------
        Nothing
        """
        p, f = os.path.split(filename)
        matname = os.path.join(p, 'updown_data.mat')
        if os.path.isfile(matname):
            fmat = scipy.io.loadmat(matname)
            f = fmat['PARS']['freqs'][0][0]
            self.frequencies = np.array([fx[0] for fx in f])
            varinfo = fmat['PARS'][0,0].dtype.names
            self.vars = {}
            for v in varinfo:
                self.vars[v] = fmat['PARS'][0,0][v][0][0]
            print('   Matfile Parameters:\n', self.vars)
            #            print self.frequenciesx
        else:
            print ('*** No matfile found ***\n')


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
        print(' in here-- binning')
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

    def image_analysis(self,xvars,yvars):
        pg.image(self.imageData)
        print('nreps',self.nrepetitions)
        imgmanip=self.imageData[:,xvars[0]:xvars[1],yvars[0]:yvars[1]]
        sh=np.shape(imgmanip)
        stim1=np.zeros([self.nrepetitions-1,sh[1],sh[2]],float)
        stim2=np.zeros([self.nrepetitions-1,sh[1],sh[2]],float)
        stim3=np.zeros([self.nrepetitions-1,sh[1],sh[2]],float)
        for i in range(self.nrepetitions):
            tempimg1=imgmanip[44:104]
            tempbk1=imgmanip[0:29]
            tempimg2=imgmanip[194:255]
            tempbk2=imgmanip[150:179]
            tempimg3=imgmanip[344:404]
            tempbk3=imgmanip[300:329]
            imgmanip= imgmanip[449:]
            if i>0:
                divimg1=np.mean(tempimg1,axis=0)/np.mean(tempbk1,axis=0)
                divimg2=np.mean(tempimg2,axis=0)/np.mean(tempbk2,axis=0)
                divimg3=np.mean(tempimg3,axis=0)/np.mean(tempbk3,axis=0)
                stim1[i-1]=divimg1
                stim2[i-1]=divimg2
                stim3[i-1]=divimg3
        print('stim1 dim: ',np.shape(stim1))
        pg.image(np.mean(stim1,axis=0),title='stim1')
        pg.image(np.mean(stim2,axis=0),title='stim2')
        pg.image(np.mean(stim3,axis=0),title='stim3')
    # def image_analysis(self, xvars, yvars):

    #     # self.imageData[:,xvars[0]:xvars[1],yvars[0]:yvars[1]] = 0
    #     tempimg= self.imageData[:,xvars[0]:xvars[1],yvars[0]:yvars[1]]
    #     self.imageData=[]
    #     self.imageData = tempimg
    #     print('in here')
    #     self.avgimg = np.mean(self.imageData[1:], axis=0) # get mean image for reference later: average across all time
    #     pg.image(self.avgimg, title='imageData')
    #     self.nFrames = self.imageData.shape[0]
    #     self.times = np.arange(0, self.nFrames/self.framerate, 1.0/self.framerate)
    #     self.maximg = np.max(self.imageData[1:], axis=0)
    #     pg.image(self.maximg, title='max img')
    #     # firstepoch = self.imageData[1:40]
    #     # secondepoch = self.imageData[41:80]
    #     # pg.image(np.max(firstepoch,axis=0),title='first epoch')
    #     # pg.image(np.max(secondepoch,axis=0),title='second epoch')
    #     self.normal_image()
    #     self.rebin_image()
    #     self.analysis_dFF_map()

    def rebin_image(self):
        """
        Rebin the image data in self.imageData
        Image data are binned as the mean across the volume specified
        
        Parameters
        ----------
        None
        
        Returns
        -------
        Nothing
        """
        # bin the image down to smaller size by combining groups of bins
        print('Rebinning image')
        sh = self.imageData.shape
        if self.binsize > 1 or self.zbinsize > 1:
            nredx = int(sh[1]/self.binsize)
            nredy = int(sh[2]/self.binsize)
            nredz = int(sh[0]/self.zbinsize)
            print('self.framerate (before binning:',self.framerate)
            self.imageData = self.bin_ndarray(self.imageData, new_shape=(nredz, nredx, nredy), operation='mean')
            if nredz > 1:
                self.nFrames = self.imageData.shape[0]
                self.framerate = self.framerate/self.zbinsize
                print('self.framerate (after binning:', self.framerate)
                self.times = np.arange(0, self.nFrames/self.framerate, 1.0/self.framerate)
        print('   Image Rebinned')
        self.print_image_info()
        print('shape of self.times after rebinning',self.times.shape)

    def print_image_info(self):
        """
        Print the image information - frames, rate, max time, shape
        
        Parameters
        ----------
        None
        
        Returns
        -------
        Nothing
        """
        maxt = np.max(self.times)
        print ("    Duration of Image Stack: %9.3f s (%8.3f min) period = %8.3f s" % (maxt, maxt/60.0, self.period))
        print ('    Image shape: ', self.imageData.shape)
        print ('    nFrames: %d   framerate: %9.3f\n' % (self.nFrames, self.framerate))
        

        
            
    def analysis_dFF_map(self):
        """
        Perform analysis of image stacks using dF/F measures and 
        the frequency list.
        
        Basic idea: each time window is associated with a response
        Assumed to be non-overlapping. Measure base (first point in time window)
        to peak of response in the
        window for each pixel in the image
        Assigns frequency to each time window, and generates a color map based on 
        the maxima responses at different locations.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        Nothing
        """
        
        self.stdimg = np.std(self.imageData[1:401], axis= 0) # and standard deviation
        pg.image(self.stdimg,title='standard deviation')
        #reduce self.imageData to size
        endpoint = int(self.framerate*(self.nrepetitions-1)*self.period+self.framerate)
        startpoint = int(self.framerate*self.period)
        print('startpoint,endpoint: ',startpoint,endpoint)
        print('number of frames: ',self.imageData.shape[0])



        # self.imageData =self.subback[self.framerate:endpoint,:,:]
        # self.imageData = self.imageData[startpoint:,:,:]
        # print('startpoint:',startpoint)
        # print('self.framerate:', self.framerate)
        # print('self.period:',self.period)
        # print('self.nrepetitions:', self.nrepetitions)
        # print('self.imageData:', np.shape(self.imageData))
        
        # sig = np.reshape(self.imageData, (self.nrepetitions-1, int(self.framerate*self.period), self.imageData.shape[1], self.imageData.shape[2]), order='C')
        # sigsq = sig**2
        # # sig = sig-np.min(sig)
        # self.nFrames= sig.shape[1] #update the numberof frames.
        # self.times = np.arange(0, self.nFrames/self.framerate, 1.0/self.framerate)
        # print('shape of self.times',self.times.shape)
#        mpl.figure(100)
#        mpl.imshow(np.mean(np.mean(sig,axis=0),axis=0))
        # pg.image(np.mean(sig,axis=0),title='signal averaged across trials')
        # meansig=np.mean(sig,axis=0)
        # # baseline=np.mean(meansig[self.times>8,:,:],axis=0)
        # # signormbaseline = meansig/baseline
        # # pg.image(signormbaseline,title='signal normalized to baseline')
        # imagepg1 = ndimage.rotate(np.max(meansig[int(self.framerate):int(2*self.framerate)],axis=0), 180, reshape = 'False')
        # imagepg2 = ndimage.rotate(np.max(meansig[int(6*self.framerate):int(7*self.framerate)],axis=0), 180, reshape = 'False')
        

        # imagempl1 = ndimage.rotate(np.max(meansig[int(self.framerate):int(2*self.framerate)],axis=0),-90, reshape = 'False')
        # imagempl2 = ndimage.rotate(np.max(meansig[int(6*self.framerate):int(7*self.framerate)],axis=0),-90, reshape = 'False')
        # pg.image(self.rawimage,title='rawimage')
        # pg.image(imagepg1,title='mean over first stimulus')
        # pg.image(imagepg2,title='mean over second stimulus')
        # imagepg = ndimage.rotate(meansig,180,reshape = 'False')
        # imagepgraw = ndimage.rotate(self.imageData,180,reshape = 'False')
        # pg.image(imagepg, title='averaged over stim presentations')
        # pg.image(imagepgraw,title='divided')
        # mpl.figure(1)
        # mpl.contour(imagempl1,levels=[0,.001,.002,.003,.004,.005,.01])
        # # mpl.contour(imagempl1,levels=[-0.025,-0.02,-0.015,-0.01,-0.005,-0.004,-0.003,-0.002,-0.001,0])
        # mpl.colorbar()

        # mpl.figure(2)
        # # mpl.contour(imagempl2,levels=[-0.025,-0.02,-0.015,-0.01,-0.005,-0.004,-0.003,-0.002,-0.001,0])
        # mpl.contour(imagempl2,levels=[0,.001,.002,.003,.004,.005,.01])
        # mpl.colorbar()
        # mpl.show()
        print ('   DF/F analysis finished.\n')

    def lineplot_maxes(self,localmax):
        ROIntx = np.arange(188,197)
        ROInty = np.arange(206,214)
        n = len(localmax)
        for i in range(n):
            maxes = localmax[i]
            for j in range(len(maxes)):
                if len(maxes) == 0:
                    continue
                lm=maxes[j]
                redlm = lm[(lm[0] in ROIntx)&(lm[1] in ROInty)]
                print ('i redlm', i, len(redlm))
                
               

    def plot_fft(self):
        """
        Provide plotting of the FFT from the middle of the image in the current axes
        
        Parameters
        ----------
        None
        
        Returns
        -------
        Nothing
        """
        self.ipx = int(self.imageData.shape[1]/2.)
        self.ipy = int(self.imageData.shape[2]/2.)
        nearf = np.absolute(self.DF[0:(self.freqs.shape[0]/2)-1,self.ipx-2:self.ipx+2,self.ipy-2:self.ipy+2])
        mpl.plot(self.freqs[0:(self.freqs.shape[0]/2)-1], np.mean(np.mean(nearf,axis=1), axis=1),
                'ko-', markersize=2.5)
        mpl.plot(self.freqs[self.freq_point], np.mean(np.mean(nearf,axis=1), axis=1)[self.freq_point], 'ro', markersize=5)
        nearf = np.absolute(self.DF[0:(self.freqs.shape[0]/2)-1,-6:-1,-6:-1])
        mpl.plot(self.freqs[0:(self.freqs.shape[0]/2)-1], np.mean(np.mean(nearf,axis=1), axis=1),
                'c-', markersize=2.5)
        mpl.title('FFT center of image and corner')

    def measure_amplitude(self, ipx, ipy):
        """
        Measure the amplitude of the signal as a function of time around the location specified.
        Note that this routine has no catch for out-of-bounds parameters
        
        Parameters
        ----------
        ipx : int (no default)
            the x position for the measurement
        ipy : int (no default)
            the y position for the measurement
        
        Returns
        -------
        The average tieme course over repitions for the region specified
        """
        intensity = np.mean(np.mean(self.imageData[:, ipx-1:ipx+1, ipy-1:ipy+1], axis=2), axis=1)
        # fold over repetitions
        remapped = intensity.reshape( self.nrepetitions, intensity.shape[0]/self.nrepetitions).mean(axis=0)
        return remapped
        
    def plot_averaged_amplitude(self):
        """
        Plot the averaged amplitude of the signal across several locations in the frame
        Note: This is the signal amplitude locked to the period of the sequence (cycle),
        so it should show responses. 
        Parameters
        ----------
        None
        
        Returns
        -------
        tuple
            y offset (int)
            a list of the x positions relative to the center of the image
        """
        tremap = np.arange(0, self.period, 1./self.framerate)
        ipy_offset = -2
        # find center of image
        self.ipx = int(self.imageData.shape[1]/2.)
        self.ipy = int(self.imageData.shape[2]/2.)
        posx = range(1, self.imageData.shape[1]-1, 2)
        colormap = mpl.cm.gist_ncar
        colx = [colormap(i) for i in np.linspace(0, 0.9, (len(posx)))]
        for i, px in enumerate(posx):
            remapped1 = self.measure_amplitude(self.ipy+ipy_offset, px)
            mpl.plot(tremap, remapped1/self.meanimagevalue, 'o-', markerfacecolor=colx[i], markersize=2.0)
        mpl.title('Averaged Amplitude Center')
        return (ipy_offset, posx, colx)
    
    def plot_amplitude_map(self, ampmap, maxamp, label, filter=0):
        """
        Provide a plot of the amplitude map
        
        Parameters
        ----------
        ampmap : two-dimensional numpy array (no default)
        maxamp : float (no default)
            scaling factor for the image coloring limits
        label : text label for the top of the plot
        filter : gaussian filter size if needed
        
        Returns
        -------
        Nothing
        """
        mpl.title(label)
        # scipy.ndimage.gaussian_filter(self.amplitudeImage1, 2, order=0, output=self.amplitudeImage1, mode='reflect')
        imga1 = mpl.imshow(ampmap, cmap=matplotlib.cm.gray) 
        mpl.colorbar()
        imga1.set_clim = (0.0, maxamp)
        
    def plot_phase_map(self, phasemap, label, filter=0):
        """
        Provide a plot of the phase map
        
        Parameters
        ----------
        phasemap : two-dimensional numpy array (no default)
        label : text label for the top of the plot
        filter : gaussian filter size if needed
        
        Returns
        -------
        Nothing
        """
        anglemap = self.mkcmap()
        mpl.title(label)
#        imgp1 = mpl.imshow(scipy.ndimage.gaussian_filter(map, filter, order=0, mode='reflect'), cmap=anglemap)#matplotlib.cm.hsv)
        imgp1 = mpl.imshow(phasemap, cmap=anglemap)#matplotlib.cm.hsv)
        imgp1.set_clim=(-np.pi/2.0, np.pi/2.0)
        mpl.colorbar()

# plot data
    def plot_maps(self, mode=0, target=1, gfilter=0):
        """
        Plot the maps from the analysis
        
        Currently, truncated to just plot in one window.
        Parameters
        ----------
        mode: int (default: 0)
            for plotting - determines what is plotted and how
        
        target: int (default: 1)
            chooses target dataset to plot (up, down)
        
        gfilter: int (default: 0)
            Gaussian filter to apply to data plot
        
        Returns
        -------
        Nothing
        """
        mpl.figure(1)
        mpl.imshow(self.avgimg, cmap=matplotlib.cm.gray, interpolation=None) # scipy.ndimage.gaussian_filter(ampmap, filter, order=0, mode='reflect'), cmap=matplotlib.cm.gray)
        mpl.colorbar()
        mpl.title('Average image')
        print ('target, mode: ', target, mode)
        max1 = np.amax(self.amplitudeImage1)
        if target > 1:
            max1 = np.amax([max1, np.amax(self.amplitudeImage2)])
        max1 = 10.0*int(max1/10.0)
        mpl.figure(2)
        mpl.subplot(2,2,4)
        ipy0, posl, coll = self.plot_averaged_amplitude()

        mpl.subplot(2,2,1)
        self.plot_amplitude_map(self.amplitudeImage1, max1, 'Amplitude Map1', filter=gfilter)
        mpl.subplot(2,2,3)
        self.plot_phase_map(self.phaseImage1, 'Phase Map1', filter=gfilter)
        for i, px in enumerate(posl):
            mpl.plot(px, self.ipy+ipy0, 'o-', markersize=5.0, markerfacecolor = coll[i], markeredgecolor='w')
        if target > 1:
            mpl.subplot(2,2,4)
            self.plot_phase_map(self.phaseImage1, 'Phase Map1', filter=gfilter)
        mpl.subplot(2,2,2)
        self.plot_fft()
        
        mpl.figure(3)
        mpl.title('Phase across center horizontally')
        # extract middle line
        sh = self.phaseImage1.shape
        iy0 = int(sh[1]/2)
        mpl.plot(self.phaseImage1[iy0, :], 'ko-')
        return
        
        if mode == 0:
            mpl.subplot(2,3,3)
            for i in range(0, self.nPhases):
                mpl.plot(ta.n_times, self.DF[:,5,5].view(ndarray))
                #mpl.plot(self.n_times, D[:,i*55+20, 60])
                mpl.hold('on')
            mpl.title('Waveforms')

            mpl.subplot(2,3,6)
            for i in range(0, self.nPhases):
                mpl.plot(ta.n_times, self.DF[:,5,5].view(ndarray))
                #mpl.plot(self.DF[:,i*55+20, 60])
                mpl.hold('on')
            mpl.title('FFTs')

        if mode == 1 and target > 1:
                
            mpl.subplot(2,3,2)
            mpl.title('Amplitude Map2')
            #scipy.ndimage.gaussian_filter(self.amplitudeImage2, 2, order=0, output=self.amplitudeImage2, mode='reflect')
            imga2 = mpl.imshow(scipy.ndimage.gaussian_filter(self.amplitudeImage2, gfilter, order=0, mode='reflect'))
            imga2.set_clim = (0.0, max1)
            mpl.colorbar()
            mpl.subplot(2,3,5)
            imgp2 = mpl.imshow(scipy.ndimage.gaussian_filter(self.phaseImage2, gfilter, order=0, mode='reflect'), cmap=matplotlib.cm.hsv)
            mpl.colorbar()
            imgp2.set_clim=(-np.pi/2.0, np.pi/2.0)
            mpl.title('Phase Map2')
            # doubled phase map
            mpl.subplot(2,3,6)
            #scipy.ndimage.gaussian_filter(self.phaseImage2, 2, order=0, output=self.phaseImage2, mode='reflect')
            np1 = scipy.ndimage.gaussian_filter(self.phaseImage1, gfilter, order=0, mode='reflect')
            np2 = scipy.ndimage.gaussian_filter(self.phaseImage2, gfilter, order=0, mode='reflect')
            dphase = np1 + np2
            #dphase = self.phaseImage1 - self.phaseImage2
           
            #scipy.ndimage.gaussian_filter(dphase, 2, order=0, output=dphase, mode='reflect')
            imgpdouble = mpl.imshow(dphase, cmap=matplotlib.cm.hsv)
            mpl.title('2x Phi map')
            mpl.colorbar()
            imgpdouble.set_clim=(-np.pi, np.pi)

        if mode == 2 or mode == 1:
            if self.phasex == []:
                self.phasex = np.random.randint(0, high=self.DF.shape[1], size=self.DF.shape[1])
                self.phasey = np.random.randint(0, high=self.DF.shape[2], size=self.DF.shape[2])

            mpl.subplot(2,3,3)
            sh = self.DF.shape
            spr = sh[2]/self.nPhases
            for i in range(0, self.nPhases):
                Dm = self.avgimg[i*spr,i*spr] # diagonal run
                mpl.plot(self.n_times, 100.0*(self.DF[:,self.phasex[i], self.phasey[i]]/Dm))
                mpl.hold('on')
            mpl.title('Waveforms')

        if mode == 2:
            mpl.subplot(2,3,6)
            sh = self.DF.shape
            x0 = int(sh[1]/2)
            y0 = int(sh[2]/2)
            for i in range(0, self.nPhases):
                mpl.plot(self.DF[1:,x0,y0])
                mpl.hold('on')
            mpl.title('FFTs')

    def mkcmap(self): 
        white = '#ffffff'
        black = '#000000'
        red = '#ff0000'
        blue = '#0000ff'
        magenta = '#ff00ff'
        cyan = '#00ffff'
        anglemap = mplcolor.LinearSegmentedColormap.from_list(
            'anglemap', [black, magenta, red, white, cyan, blue, black], N=256, gamma=1)
        return anglemap

        

if __name__ == "__main__":
    global win, roi, img, rawdata
    layout = None
    winsize = []
    ta = FFTImageAnalysis(layout=layout, winsize=winsize)  
    ta.parse_commands(sys.argv[1:])
    ta.ROIdata()
    app.exec_()
    
    assignedROI=roi.getArraySlice(rawdata,img,axes=(0,1),returnSlice=False)
    print('assignedROI', assignedROI)
    vals=assignedROI[0]
    xvals = vals[0]
    yvals = vals[1]
    print('assignedROIx',xvals)
    print('assignedROIy',yvals)

    ta.parse_commands(sys.argv[1:])
    ta.image_analysis(xvals,yvals)
    app.exec_()
