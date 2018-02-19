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
import scipy.ndimage
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
from pyqtgraph.metaarray import MetaArray
from optparse import OptionParser
import matplotlib.colors as mplcolor

#import seaborn as sns





# List of old experiments:



# frequency list for runs 15 May, 24 May and 2 June 2010, until #60 in 2-June

fl1=[1, 1.414, 2.0, 2.828, 4.0, 5.656, 8.0, 11.318, 16.0, 22.627, 32.0, 45.254]

# frequency list for runs 2-June 2010, starting at #60 (heavier coverage of higher frequencies) and text note

fl2 = [4.0, 4.756, 5.656, 6.727, 8.0, 9.5, 11.3, 13.45, 16.0, 19.02, 22.62, 26.91, 31.99, 38.09, 45.25, 53.8]

# dictionary of data sets

# Keys are first file #. Data are file name (up, down), wavelength, attn, period, date, frequency list, comment

# 15 May 10:  used amber LED (noisy) for 610 illumination

DB = {10: ('010', '011', 610, 15.0, 6.444, '15May10', fl1, 'thinned skull')} # lots of hf oscillations in image; phase map ???

DB[14] = ('014', '015', 610, 15.0, 6.444, '15May10', fl1, 'dura, focus near surface') # hmmm

DB[18] = ('018', '019', 610, 15.0, 6.444, '15May10', fl1, 'dura, deeper focus')

DB[22] = ('022', '023', 610, 8.0, 6.444, '15May10', fl1, 'dura, deeper focus')

DB[24] = ('024', '025', 610, 29.0, 6.444, '15May10', fl1, 'dura, deeper focus')

DB[26] = ('026', '027', 560, 29.0, 6.444, '15May10', fl1, 'dura, deeper focus') # light fluctuations; some phase shifts though

DB[28] = ('028', '029', 560, 26.0, 6.444, '15May10', fl1, 'dura, focus near surface') # too many large fluctuations - can't trust



# 24 May 10: used white LED and put 610 filter in front of camera (quieter illumination)

# potential damage to cortex (bleeder)

DB[32] = ('032', '033', 610, 5.0, 6.412, '24May10', fl1, 'dura, just below surface') # amplitude maps look similar; phase maps look good

DB[34] = ('034', '035', 610, 30.0, 6.412, '24May10', fl1, 'dura, just below surface') #linear drift, but corrections ok; phase gradient

DB[36] = ('037', '036', 610, 120.0, 6.412, '24May10', fl1, 'dura, just below surface') # no input to speaker; phase map somewhat flat

DB[39] = ('039', '041', 610, 20.0, 6.482, '24May10', fl1, 'dura, just below surface') # illim steady; no clear resonse in phase map



# 02 June 10: used white LED and green LED

DB[42] = ('042', '043', 610, 5.0, 6.452, '02Jun10', fl1, 'thinned skull') # not too bad

DB[44] = ('045', '044', 560, 5.0, 6.452, '02Jun10', fl1, 'thinned skull') # not bad; led is stable

DB[48] = ('049', '048', 560, 5.0, 6.412, '02Jun10', fl1, 'thinned skull') # up has large drift - NG

DB[50] = ('050', '051', 610, 20.0, 6.412, '02Jun10', fl1, 'thinned skull') # both have drift, but not many larg fluctuatiosn - map spotty

DB[52] = ('052', '033', 610, 35.0, 6.422, '02Jun10', fl1, 'thinned skull, focussed slightly deeper') # many large light flucturtions

DB[54] = ('054', '055', 610, 120.0, 6.412, '02Jun10', fl1, 'thinned skull') # no stim control

DB[56] = ('056', '057', 560, 120.0, 6.412, '02Jun10', fl1, 'thinned skull') # no stim control

# changed frequency map for next runs on 02 June 2010

DB[60] = ('061', '060', 560, 10.0, 4.276, '02Jun10', fl2, 'thinned skull') # large drift on direction

DB[62] = ('062', '063', 610, 10.0, 4.276, '02Jun10', fl2, 'thinned skull') # drift and noise

DB[64] = ('064', '065', 610, 30.0, 4.274, '02Jun10', fl2, 'thinned skull') # possibly OK - clean illumination

DB[66] = ('066', '067', 610, 15.0, 4.274, '02Jun10', fl2, 'thinned skull') # Might be good!!!!



# 09 June 10: QuantEM512SC for imaging 

DB[68] = ('068', '069', 610, 15.0, 4.228, '09Jun10', fl2, 'thinned skull') # focus near surface Might be good!!!! diagonal phase gradient

DB[70] = ('070', '071', 610, 15.0, 4.228, '09Jun10', fl2, 'thinned skull') # focus deeper Might be good!!!! Diagonal phase gradient - but horizontal stripe too

DB[73] = ('073', '074', 610, 15.0, 4.228, '09Jun10', fl2, 'thinned skull') # same as 68/70, but no stimulation Might be good!!!!

DB[75] = ('075', '076', 560, 15.0, 4.224, '09Jun10', fl2, 'thinned skull') # Green light, 120 msec integration time phase map with structure

DB[77] = ('077', '078', 560, 25.0, 4.248, '09Jun10', fl2, 'thinned skull') # Green light, 151 msec integration time - a little noisy ?

DB[79] = ('079', '081', 610, 15.0, 4.204, '09Jun10', fl2, 'thinned skull') # 610, 30 fps, 30 msec integration time way too big to handle

DB[82] = ('082', '085', 610, 15.0, 4.204, '09Jun10', fl2, 'thinned skull') # 610, 30 fps, 30 msec integration time broken down, gradient, but horizontal stripe

DB[83] = ('083', '086', 610, 15.0, 4.204, '09Jun10', fl2, 'thinned skull') # 610, 30 fps, 30 msec integration time -- Diagonal gradient 

DB[84] = ('084', '087', 610, 15.0, 4.204, '09Jun10', fl2, 'thinned skull') # 610, 30 fps, 30 msec integration time -- Diagonal gradient





fn = '../camera_updown_20161017_7_MMStack_Pos0.ome.tif'

freqlist = np.logspace(3, 4.7, 12, base=10)

# homedir = os.getenv('HOME')

# workingpath = '/Volumes/Pegasus/ManisLab_Data3/IntrinsicImaging/video_'

# basepath = os.path.join(homedir, workingpath)

# basepath = '/Volumes/Pegasus/ManisLab_Data3/IntrinsicImaging/'

# basepath = '/Volumes/Pegasus/ManisLab_Data3/Ropp_Tessa/AcquisitionData/'

#basepath = '.'

if sys.platform == 'win32':
    basepath = 'AcquisitionData'
    homedir = 'c:\Users\experimenters'
    basepath = os.path.join(homedir, basepath)

basepath = 'micromanager'

# homedir = '/Volumes/TROPPDATA/data'

homedir = '/Volumes/TROPPDRIVE/'
# 
# homedir = '/Users/tessa-jonneropp/Desktop/data'

basepath = os.path.join(homedir, basepath)



class FFTImageAnalysis():

    """

    Provide routines for analyzing image stacks by extracting phase and amplitude 

    across the image plane from the time data series

    """

    def __init__(self, layout, winsize, d=[], measurePeriod=4.0, binsize=1, gfilter=0):

        self.layout = layout

        self.winsize = winsize

        self.d = d

        self.mode = False # mode = False -> FFT analysis, mode = True -> dF/F analysis

        self.period = measurePeriod # sec

        self.freqperiod = 1.0

        self.framerate = 30. # Hz

        self.nPhases = 6  # for test stimulus

        self.nrepetitions = 50 # number of repeats of the presentation

        self.imageSize = [128, 128] # image size x : y, used for testing

        self.bkgdIntensity = 1200. # mean intensity of test signal

        self.signal_size = 5e-3*self.bkgdIntensity

        self.bkgdNoise = 1e-3*self.bkgdIntensity # std of gaussian noise in background

        self.threshold = 0.5

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

        

        columns = 2

        if self.layout is not None:

            dx = (self.winsize[1] - 50)/columns  # 50 pix for command bar on left; 2 columns

            for i in range(0, columns):

                self.layout.setColumnMinimumWidth(i+1,dx)

        

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

        

        if self.freqinfo is not None:

            f0, octave, nfreq = eval(self.freqinfo)

            self.frequencies = self.compute_freqs(f0, octave, nfreq)



        if options.upfile is not None:

            self.dir = 'up'



        if options.downfile is not None:

            self.dir = 'down'



        if options.directory is not None:

            self.directory = options.directory



        # if options.upfile is not None:

        #     self.upfile = options.upfile

        #     target = 1

        #

        # if options.downfile is not None:

        #     self.downfile = options.downfile

        #     target = 2

            

        if options.test is True:

            self.make_test_data()

            return

        

        if options.tifffile is not None:

            n2 = self.tifffile + '_MMStack_Pos0.ome.tif'

            self.read_tiff_stack(filename=os.path.join(basepath, self.tifffile, n2))

            return



        if options.fdict is not None:

            if options.fdict in DB.keys(): # populate options 

                options.upfile = DB[options.fdict][0]

                options.downfile = DB[options.fdict][1]

                options.period = DB[options.fdict][4]

                self.period = options.period

            else:

               print ("File %d NOT in DBase\n" % options.fdict)

               return

               

            self.read_meta_stack(os.path.join(basepath, 'video_' + options.upfile + '.ma'))





    def read_meta_stack(self, filename):

        self.imageData = []

        print ("Loading (as metaarray) data from %s" % filename)

        try:

            im = MetaArray(file = filename) # ,  subset=(slice(0,2), slice(64,128), slice(64,128)))

        except:

            print ('   Error loading upfile: %s' % filename)

            return

        print( '... Data loaded')

        self.timebase = im.axisValues('Time').astype('float32')

        self.framerate = 1./np.mean(np.diff(self.timebase))

        print('  Framerate: {:6.1f}/s   Period: {:6.1f}s'.format(self.framerate, self.period))

        self.imageData = im.view(np.ndarray).astype('float64')

        if self.imageData.shape[1] == 267:

            self.imageData = self.imageData[:,0:256,:]

        print ('   Data shape: ', self.imageData.shape)

        #insert line to truncate file

        # self.imageData = self.imageData[1:4001]

        self.nFrames = self.imageData.shape[0]

        self.nrepetitions = int(self.nFrames/(self.period * self.framerate))

        self.adjust_image_data()

        self.avgimg = np.mean(self.imageData, axis=0) # get mean image for reference later: average across all time

        print ('   Read file, original Image Info: ')

        self.print_image_info()

        self.rebin_image()

#        self.clean_windowerrors()

        if not self.mode:

            self.analysis_fourier_map(target=1, mode=0)

            self.plot_maps(mode=1, gfilter=self.gfilter)

        else:

            self.analysis_dFF_map()

        mpl.show()

        

    def analyze(self, options, target):

        """

        Perform analysis

        This is the old routine, works with acq4 metaarray files

        Not supported

        """

        target = 0

        upf = None

        dwnf = None

        if options.upfile is not None:

            upf = basepath + options.upfile + '.ma'

        if options.downfile is not None:

            dwnf = basepath + options.downfile + '.ma'



        for filename in (upf, dwnf):

            # if options.upfile is not None and options.downfile is not None:

            if filename is None:

               break

            im=[]

            self.imageData = []

            print ("Loading data from %s" % filename)

            try:

                im = MetaArray(file = filename,  subset=(slice(0,2), slice(64,128), slice(64,128)))

            except:

                print('   Error loading upfile: %s' % filename)

                return

            print('   Data loaded')

            target = target + 1

            self.times = im.axisValues('Time').astype('float32')

            self.imageData = im.view(np.ndarray).astype('float32')

            im=[]

            self.analysis_fourier_map(period=self.period, target=target,  bins=binsize,)

        if target > 0:

            self.plot_maps(mode = 1, target = target, gfilter = self.gfilter)



    def make_test_data(self):

        """

        Create a test data set to verify the operation of the algorithm.

        Can also be used to test sensitivty in the presence of noise

        

        Parameters

        ----------

        None

        

        Return

        ------

        Nothing

        """

        

        print ("Creating Test Sample:")

        print ('   Period, rate, reps, phases: ', self.period, self.framerate, self.nrepetitions, self.nPhases)

        nframes = int(self.period * self.framerate * self.nrepetitions)

        print ('   nframes: ', nframes)

        if self.bkgdNoise > 0.:

            d = np.random.normal(size=(nframes,self.imageSize[0],self.imageSize[1]),

                                loc=self.bkgdIntensity, scale=self.bkgdNoise).astype('float32')

        else:

            d = self.bkgdIntensity*np.ones((nframes,self.imageSize[0],self.imageSize[1])).astype('float32')

            

        ds = d.shape

        print ('   data shape: ', ds)

        dx = int(ds[2]/4)

        xc = int(ds[2]/2)

        xo = [xc-dx, xc+dx]

        ywidth = int(ds[2]/(self.nPhases+2))

        framedelay = 4



        if not self.mode:

            self.phasex = []

            self.phasey = []

            for i in range(0,self.nPhases):

                dy = int((i+1)*ds[2]/(self.nPhases+2)) # each phase is assigned to a region

                self.resp = np.zeros((nframes,))

                self.resp = np.cos(

                         np.linspace(0, 2.0*np.pi*nframes/(self.period*self.framerate), nframes-framedelay)+i*np.pi/8 - np.pi/2.0)

                self.resp = np.concatenate((np.zeros(framedelay), self.resp))

                d[:, xo[0]:xo[1], dy:dy+ywidth ] += self.resp[:, np.newaxis, np.newaxis]

                self.phasey.append( (2+(dy+int(ds[2]/self.nPhases))/2))

                self.phasex.append((6+int(ds[1]/2)/2)) # make the signal equivalent of digitized one (baseline 3000, signal at 1e-4 of baseline)

        else:

            self.nPhases = 4

            self.spotsize = 16

            nrpts = 20

            nsites = 4

            one_rep = int(self.period*self.framerate)

            isi = int(self.period*self.framerate/self.nPhases)

            print('period, isi: ', self.period, isi)

            r = np.arange(0, nrpts, 1.)

            alpha = 4.

            A  = r/alpha *np.exp(-(r-alpha)/alpha)  # scaled alpha function

            self.spot= self.gauss_spot(self.spotsize, 3.)  # the 2d spot

            sigsize = np.random.normal(size=self.nPhases, loc=self.signal_size, scale=self.signal_size*2)

            sigsize = [np.abs(s) for s in sigsize] # restrict to positive amplitudes

            print ('sigsize: ', sigsize)

            for j in range(self.nrepetitions):

                for i in range(self.nPhases):

                    self.resp = np.zeros((nrpts, self.spot.shape[0], self.spot.shape[1]))

                    for k in range(nrpts):

                        self.resp[k,:,:] += sigsize[i]*A[k] * self.spot  # make response an alpha time course of gaussian spot

                    start = j*one_rep + i*isi + framedelay

                    stop = start + nrpts

                    dy = int((i+1)*ds[2]/(self.nPhases+2)) # location for phase

                    #dy = dy + 2*z

#                    print ('start, stop: ', start, stop)

                    for z in range(nsites):

                        #self.resp = np.concatenate((np.zeros(framedelay), self.resp))

                        xp = xo[0] + i*10 - 10*z

                        yp = dy - i*10 + 10*z

                        d[start:stop, xp:xp+self.spotsize, yp:yp+self.spotsize ] += self.resp

        self.imageData = d  # reduce to a 16-bit map to match camera data type

        self.nFrames = self.imageData.shape[0]

        self.times = np.arange(0, nframes/self.framerate, 1.0/self.framerate)

        print( "   Test Image Created")

        # imv = pg.ImageView()

        # imv.show()

        # imv.setImage(self.imageData)



        if self.layout is not None:

            self.layout.addWidget(imv, 0, 0)



            avgImage = np.mean(self.imageData, axis=0)

            ima = pg.ImageView()

            ima.setImage(avgImage)

            self.layout.addWidget(ima, 0, 1)

        self.adjust_image_data()

        self.avgimg = np.mean(self.imageData, axis=0) # get mean image for reference later: average across all time

        print ('   Test file, original Image Info: ')

        self.print_image_info()

        self.rebin_image()

        #self.clean_windowerrors()

        # pg.image(self.imageData)

        # pg.show()

        # mpl.figure(1)

        # mpl.show()

        if not self.mode:  # FFT analysis

                self.analysis_fourier_map(target=1, mode=0)

                self.plot_maps(mode=2, gfilter=self.gfilter)

        else:

            self.analysis_dFF_map()

        mpl.show()



    def gauss_spot(self, xy, sigma, center=None):

        """

        make a gaussian spot, unitary amplitude, centered, sigma = s, size x y

        """

        x = np.arange(0, xy, 1.)

        y = x[:,np.newaxis]

    

        if center is None:

            x0 = y0 = xy // 2

        else:

            x0 = center[0]

            y0 = center[1]

    

        return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / sigma**2)

        

        

    def read_tiff_stack(self, filename=None):

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

        self.imageData = tf.imread(filename)

        self.nFrames = self.imageData.shape[0]

        self.nrepetitions = int(self.nFrames/(self.period * self.framerate))

        self.adjust_image_data()

        self.avgimg = np.mean(self.imageData, axis=0) # get mean image for reference later: average across all time

        print ('   Read tiff file, original Image Info: ')

        self.print_image_info()

        self.rebin_image()

        self.clean_windowerrors()

        if not self.mode:  # FFT analysis

            self.analysis_fourier_map(target=1, mode=0)

            self.plot_maps(mode=1, gfilter=self.gfilter)

        else:  # just deltaF/F analysis

            self.analysis_dFF_map()

        mpl.show()



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
            print('frequencies: ', self.frequencies)

        else:

            print ('*** No matfile found ***\n')





    def compute_freqs(self, basef, octavespacing, N):

        return([basef*2**(k*octavespacing) for k in range(N)])

            

    def plot_image_sequence(self):

        """

        Plot the current image sequence using pyqtgraph

        

        Parameters

        ----------

        None

        

        Returns

        -------

        Nothing

        """

        imv = pg.ImageView()

        imv.show()

        imv.setImage(self.imageData)

        self.layout.addWidget(imv, 0, 0)



        avgImage = np.mean(self.imageData, axis=0)

        ima = pg.ImageView()

        ima.setImage(avgImage)

        self.layout.addWidget(ima, 1, 0)



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
        print('flattened shape: ',np.shape(flattened))
        ndarray = ndarray.reshape(flattened)

        for i in range(len(new_shape)):

            if operation.lower() == "sum":

                ndarray = ndarray.sum(-1*(i+1))

            elif operation.lower() in ["mean", "average", "avg"]:

                ndarray = ndarray.mean(-1*(i+1))

        return ndarray



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

            nredz = int(self.imageData.shape[0]/self.zbinsize)
            print('nredx,nredy,nredz: ',[nredx,nredy,nredz])

            self.imageData = self.bin_ndarray(self.imageData, new_shape=(nredz, nredx, nredy), operation='mean')

            if nredz > 1:

                beforeFrames = self.nFrames

                self.nFrames = self.imageData.shape[0]

                self.framerate = self.nFrames/(self.nrepetitions*self.period)

                self.times = np.arange(0, self.nFrames/self.framerate, 1.0/self.framerate)

        print('   Image Rebinned')

        self.print_image_info()



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

        



    def adjust_image_data(self):

        """

        Adjust the image data shape to be an integral multiple of cycles

        Parameters

        ----------

        None

        

        Returns

        -------

        Nothing

        """

        print('Adjusting image data: ')

        if self.removeFirstSequence:  # used to remove the first trial from the sequence

            frames_per_rep = self.nFrames/self.nrepetitions

            self.imageData = self.imageData[frames_per_rep:, :, :]

            self.nFrames = self.imageData.shape[0]

            self.nrepetitions = int(self.nFrames/(self.period * self.framerate))

        self.times = np.arange(0, self.nFrames/self.framerate, 1.0/self.framerate)

        

        # first squeeze the image to 3d if it is 4d

        maxt = np.max(self.times) # find last image time

        sh = self.imageData.shape

        if len(sh) == 4:

           self.imageData = self.imageData.squeeze()

           sh = self.imageData.shape

        dt = np.mean(np.diff(self.times)) # get the mean dt

        n_Periods = int((maxt+dt)/self.period) # how many full periods in the image set - include the first?

        if self.nrepetitions > 0 and self.nrepetitions < n_Periods:

            n_Periods = self.nrepetitions

        n_PtsPerCycle = int(np.floor(self.period/dt)); # estimate image points in a stimulus cycle

        ndt = self.period/n_PtsPerCycle

        self.imageData = self.imageData[range(0, n_Periods*n_PtsPerCycle),:,:] # reduce to only what we need

        print ('   Adjusted image info')

        print ("   # Periods: %d  Pts/cycle: %d Cycle dt %8.4fs (%8.3fHz) Cycle: %7.4fs" %(n_Periods, n_PtsPerCycle, ndt, 1.0/ndt, self.period))

        self.print_image_info()



    def clean_windowerrors(self, amount=1.0-15/2e5, nskip=3):

        """

        Attempt to clean up an error due to camera window timing

        every 3rd frame seems to have the wrong intensity

        

        Parameters

        ----------

        amount : float (default provided)

            amount of correction to apply to the raw signal for every nth (nskip) frame

        

        nskip : int (default: 3)

            tframe skip parameter for application of the correction

        

        Returns

        -------

        Nothing

        """

        pts = range(2, self.imageData.shape[0], nskip)

        self.imageData[pts,:,:] = self.imageData[pts, :, :] * amount

        

    def analysis_fourier_map(self, target=1, mode=0):

        """

        Perform analysis of image stacks using Kalatsky-Stryker method.

        

        Parameters

        ----------

        target : int (default: 1)

            set to 1 for 'up' sequence, 0 for 'down' sequence

        mode : int (default: 0)

            plot mode for data (passed to plot_maps)

        up : int (default: 1)

        

        Returns

        -------

        Nothing

        """

        

        print('Starting fourier analysis:')

        self.print_image_info()

        # get the average image and the average of the whole image over time

        avgimg = np.mean(self.imageData, axis=0) # get mean image for reference later: average across all time

        self.meanimagevalue = np.mean(np.mean(avgimg, axis=1), axis=0)

        self.stdimg = np.std(self.imageData, axis= 0) # and standard deviation



        width = int(self.period*self.framerate*2)

        print( "   Detrending:")

        print( '      Median filter width: ', width)

        # footprint = np.ones((width, 1, 1))

        # self.imageData  = self.imageData - scipy.ndimage.median_filter(self.imageData, footprint=footprint)

        print( "      Done detrending")



        self.n_times = self.timebase



        # calculate FFT and get amplitude and phase

        self.DF = np.fft.fft(self.imageData, axis = 0)
        self.freqs = np.fft.fftfreq(self.DF.shape[0], d=1./self.framerate)

        # self.freqs = np.fft.fftfreq(self.DF.shape[0], d=1./self.framerate)

        print ('   df shape: ', self.DF.shape)

        print ('   1/framerate: ', 1./self.framerate)

        self.freq_point = np.argmin(np.abs(self.freqs - 1./self.period))
        print ('period:', self.period)
        print ('frequency: ', 1./self.period)
        print ('freq_point: ', self.freq_point)
        print ('frequency value: ',self.freqs[self.freq_point])
        steps = np.arange(1,6,dtype=np.float)
        steps = (steps)+1.
        self.assigned_freqs=2.*np.pi*1./1.6*steps
        print ('assigned freqs', self.assigned_freqs)

        #j = j + 2  # just looking at FFT leakage...`

        print ('   closest index/freq, period: ', self.freq_point, self.freqs[self.freq_point], 1./self.period)

        self.print_image_info()

        ampimg = np.absolute(self.DF[self.freq_point,:,:])

        phaseimg = np.angle(self.DF[self.freq_point,:,:])

        
        # ampimg = np.absolute(self.DF[self.freq_point,:,:])


        # phaseimg = np.angle(self.DF[self.freq_point,:,:])

        if target == 1:

            f = open('img_phase1.dat', 'w')

            pickle.dump(phaseimg, f)

            f.close()

            f = open('img_amplitude1.dat', 'w')

            pickle.dump(ampimg, f)

            f.close()

            self.amplitudeImage1 = ampimg

            self.phaseImage1 = phaseimg

        if target == 2:

            f = open('img_phase2.dat', 'w')

            pickle.dump(phaseimg, f)

            f.close()

            f = open('img_amplitude2.dat', 'w')

            pickle.dump(ampimg, f)

            f.close()

            self.amplitudeImage2 = ampimg

            self.phaseImage2 = phaseimg

        print ("   FFT calculated, data saved.\n")

        # save most recent calculation to disk

        

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

        

        print ('Starting dF/F analysis:')

        self.print_image_info()

        # smoothwin = int(self.imageData.shape[1]/8.)

        # get the average image and the average of the whole image over time

        avgimg = np.mean(self.imageData, axis=0) # get mean image for reference later: average across all time

        

        mpl.figure(99)

        mpl.imshow(avgimg, vmin=0, vmax=np.max(np.max(avgimg, axis=0), axis=0))

        # self.meanimagevalue = np.mean(np.mean(avgimg, axis=1), axis=0)

        # self.stdimg = np.std(self.imageData, axis= 0) # and standard deviation

        imgdatasm = scipy.ndimage.filters.gaussian_filter(self.imageData,[0,2,2],order=0,output=None,mode='reflect',cval=0.0,truncate=4.0)
        # field correction: smooth the average image, subtract it from the imagedata, then add back the mean value
        avgimgsm = scipy.ndimage.filters.gaussian_filter(avgimg, 2, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)

        # avgimgsm = scipy.ndimage.filters.gaussian_filter(avgimg, smoothwin, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)

        #self.imageData = (self.imageData-avgimgsm)+ self.meanimagevalue

        mpl.figure(98)
        mpl.imshow(avgimgsm,vmin=0, vmax=np.max(np.max(avgimgsm, axis=0), axis=0))
        mpl.figure(97)
        mpl.imshow(np.mean(imgdatasm,axis=0))
        self.n_times = self.timebase

        periodsize = int(self.period*self.framerate)
        print('periodsize: ',periodsize)

        # windowsize = int(self.freqperiod*self.framerate)  # window size for every response

        # r = range(0, self.imageData.shape[0], windowsize)

        sig = np.reshape(self.imageData, (self.nrepetitions, periodsize, 

                self.imageData.shape[1], self.imageData.shape[2]), order='C')

        delresp=np.zeros([19,256,256])
        repback = np.mean(sig[:,1:41,:,:],axis=1)
        resp = np.mean(sig[:,53:90,:,:],axis=1)
        for counter in range(19):
            delresp[counter,:,:]=(resp[counter,:,:]-repback[counter,:,:])/repback[counter,:,:]
        quot=np.mean(delresp,axis=0)
        print ('shape of quot: ', np.shape(quot))
        # quot=(resp-repback)/repback
        # quot[quot>0]=0
        # quot=-1000*quot

        mpl.figure(7)
        mpl.imshow(quot,cmap=mpl.cm.gist_rainbow)
        mpl.colorbar()

        quotsm = scipy.ndimage.filters.gaussian_filter(quot, 2, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
        mpl.figure(8)
        mpl.imshow(quotsm,cmap=mpl.cm.gist_rainbow)
        mpl.colorbar()
        
        # bl = np.mean(sig[:, range(0, sig.shape[1], windowsize), :, :], axis=0)

        # bl = scipy.ndimage.filters.gaussian_filter(bl, smoothwin, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)



        # print ('   windowsize: ', windowsize)

        # print ('   periodsize: ', periodsize)
        # mc = matplotlib.cm

        # only use sequential maps here

        # clist = [mc.Reds, mc.YlOrBr, mc.Oranges, mc.Greens, mc.GnBu, mc.Blues, mc.RdPu, mc.Purples,mc.Reds,mc.Greens,mc.Blues,mc.Reds,mc.Reds,mc.Reds,mc.Reds]
        # clist2 = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'black','red','purple','green','blue','red','red','red','red']

        cs = {}

        # sigd = np.zeros((bl.shape[0], sig.shape[2], sig.shape[3]))
# 
        # localmax = {}

        # sigmax = 0.
# 
        # kernel = np.ones((5, 5))

        # psf = kernel / np.sum(kernel)

        # compute dF/F, and get maximum over all frequencies

        print ('   sig shape: ', sig.shape)

        # print ('   bl shape: ', bl.shape)

        # smax = np.zeros(bl.shape[0])

        # for i in range(bl.shape[0]):

        #     sigd[i] = (np.mean(np.max(sig[:,range(i*windowsize, i*windowsize+windowsize),:,:], axis=0), axis=0) - bl[i,:,:])/bl[i,:,:]

        #     sigd[i] = sigd[i]**2.0

            # smooth

            #sigd[i] = scipy.ndimage.filters.gaussian_filter(sigd[i], 1., order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)

            # deconvolve

           # sigd[i] = restoration.richardson_lucy(sigd[i], psf, 5)

#             sm = sigd[i].max().max()

#             if sm > sigmax:

#                 sigmax = sm

#             smax[i] = sm

#             print( '   i, sm: ', i, sm)

#         # now process for display

#         print ('   sigd shape: ', sigd.shape)

#         wdat = np.mean(sig, axis=0)

#         wds = wdat.shape

#         print('wdat shape: ', wds)

# #        print (range(int(wds[1]/2.), int(3.*wds[1]/4.)), range(int(wds[2]/2.), int(3.*wds[2]/4.)))

#         print( 'reduced shape: ', wdat[:,range(int(wds[1]/2.),int(3.*wds[1]/4.)),:][:,:,range(int(wds[2]/2.), int(3.*wds[2]/4.))].shape)

#         wp = wdat[:,range(int(wds[1]/2.),int(3.*wds[1]/4.)),:][:,:,range(int(wds[2]/2.), int(3.*wds[2]/4.))]

#         wp = np.mean(np.mean(wdat, axis=1), axis=1)

#         mpl.figure(1)

#         mpl.plot(np.linspace(0., len(wp)*1./self.framerate, num=len(wp)), wp)



#         mpl.figure(2)

#         for i in range(sigd.shape[0]):

#             sigd[i][sigd[i] < self.threshold*sigmax] = 0.

#             # find center of mass of areas above threshold

#             # mass = sigd[i].copy()

#             # mass[sigd[i] > 0.] = 1.

#             # structuring_element = [[0,1,0],[1,1,1],[0,1,0]]

#             # segmentation, segments = scipy.ndimage.label(mass, structuring_element)

#             # coords = scipy.ndimage.center_of_mass(sigd[i], segmentation, range(1,segments+1))

#             # xcoords = np.array([x[1] for x in coords])

#             # ycoords = np.array([x[0] for x in coords])

#             # cs[i] = (xcoords, ycoords)



#             # Calculating local maxima

#             lm = skif.peak_local_max(sigd[i], min_distance=2, threshold_rel=0.25, exclude_border=False, 

#                 indices=True, num_peaks=10, footprint=None, labels=None)

#             localmax[i] = [(m[0], m[1], sigd[i][(m[0], m[1])]) for m in lm]

#             # print ('i, local max: ',i, localmax)

#             mpl.subplot(5,5,i+1)
#             print ('shape of sigd: ',[np.shape(sigd),i])

#             imga1 = mpl.imshow(sigd[i], cmap=clist[i], vmin=0, origin='lower')

#             if len(localmax[i]) > 0:

#                 max_fr = np.max([m[2] for m in localmax[i]])

#             else:

#                 continue

#             scattersize = 30.

#             for k, lm in enumerate(localmax[i]):

#                 mpl.scatter(lm[1], lm[0], marker='o', c=clist2[i], edgecolors='k',

#                     s=scattersize*lm[2]/max_fr, linewidths=0.125, alpha=0.5)

#             mpl.subplot(6,5,i+15+1)

#             wr = range(i*windowsize, i*windowsize+windowsize)

#             # print ('   wr: len, min max: ', len(wr), min(wr), max(wr))

#             wmax = 0.

#             for lmax in localmax[i]: # was xcoords

#                 wave = wdat[wr, lmax[0],lmax[1]]

#                 wdff = (wave-wave[0])/wave[0]

#                 if np.max(wdff) > wmax:

#                     wmax = np.max(wdff)

#                 mpl.plot(np.linspace(0., len(wave)*1./self.framerate, num=len(wave)),

#                         wdff, color=clist2[i])

#             mpl.ylim(-0.1*wmax, wmax)

#         fig = mpl.figure(3)

#         for i in range(sigd.shape[0]):

#             if len(localmax[i]) == 0:

#                 continue

#             max_fr = np.max([m[2] for m in localmax[i]])

#             for lm in localmax[i]:

#                 mpl.scatter(lm[1], lm[0], marker='o', c=clist2[i], 

#                 s=scattersize*lm[2]/max_fr, alpha=0.5, edgecolors='k')

#         mpl.ylim(0, sigd.shape[2])

#         mpl.xlim(0, sigd.shape[1])

#         mpl.axis('equal')

        mpl.show()

        print ('   DF/F analysis finished.\n')



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

        The average time course over repitions for the region specified

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

        #scipy.ndimage.gaussian_filter(self.amplitudeImage1, 2, order=0, output=self.amplitudeImage1, mode='reflect')

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



    layout = None

    winsize = []

    ta = FFTImageAnalysis(layout=layout, winsize=winsize)  # create instance (for debugging)

    ta.parse_commands(sys.argv[1:])



    mpl.show()

