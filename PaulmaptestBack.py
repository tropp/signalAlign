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
import scipy.io
import scipy.signal
import scipy.ndimage
#import scipy.stsci.convolve
#import astropy.convolution
from astropy.convolution import convolve_fft, convolve, Box2DKernel, Box1DKernel
import pickle
try:
    import matplotlib
#TFR 11/13/15 inserted the following line to try to resolve issue with pylab.show
#matplotlib.rcParams['backend'] = "QtAgg"
    import matplotlib.mlab as mlab
    import pylab
    from matplotlib import rc
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    ## for Palatino and other serif fonts use:
    #rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)
    HAVE_MPL = True
except:
    HAVE_MPL = False

import pyqtgraph as pg #added to deal with plottng issues TFR 11/13/15
app = pg.Qt.QtGui.QApplication([])

from pyqtgraph.metaarray import MetaArray
#import pylibrary.Utility as Utils
#from pylibrary.Utility import SignalFilter_LPFBessel
from optparse import OptionParser

# frequency list for runs 15 May, 24 May and 2 June 2010, until #60 in 2-June
# freqlist = np.logspace(0, 4, num=17, base=2.0)
# fl1 = [3000*x for x in freqlist]
# #fl1=[1, 1.414, 2.0, 2.828, 4.0, 5.656, 8.0, 11.318, 16.0, 22.627, 32.0, 45.254]
# # frequency list for runs 2-June 2010, starting at #60 (heavier coverage of higher frequencies) and text note
# fl = [4.0, 4.756, 5.656, 6.727, 8.0, 9.5, 11.3, 13.45, 16.0, 19.02, 22.62, 26.91, 31.99, 38.09, 45.25, 53.8]

freqlist = np.logspace(0, 4, num=9, base=2.0)
fl2 = [3000*x for x in freqlist]
print 'freq list', fl2

freqlist = np.logspace(0, 4, num=5, base=2.0)
fl3 = [3000*x for x in freqlist]
print 'freq list', fl3

# dictionary of data sets
# Keys are first file #. Data are file name (up, down), wavelength, attn, period, date, frequency list, comment
# 15 May 10:  used amber LED (noisy) for 610 illumination
DB = {2: ('002', '001', 610, 15.0, 2.25, '29Apr16', fl2, 'no skull')}
#DB[4] = ('004', '003', 610, 30.0, 2.25, '29Apr16', fl2, 'no skull')
#DB[3] = ('003', '002', 610, 20.0, 1.25, fl2, 'no skull')
# DB = {11: ('011', '004', 610, 20.0, 4.5, '09Feb16', fl1, 'thinned skull')}
# DB[14] = ('014', '021', 610, 15.0, 4.25, '09Feb16', fl1, 'thinned skull') #Tessa's data
#DB[18] = ('018', '019', 610, 15.0, 6.444, '16May16', fl1, 'thinned skull')
DB[4] = ('004', '000', 610, 15.0, 4.5, '25May16', fl2, 'thinned skull')
DB[6] = ('006', '010', 610, 15.0, 4.5, '25May16', fl2, 'thinned skull')
DB[16] = ('016', '012', 610, 15.0, 4.5, '25May16', fl2, 'thinned skull')
DB[17] = ('017', '018', 610, 15.0, 2.5, '25May16', fl3, 'thinned skull')
DB[20] = ('020', '019', 610, 15.0, 2.5, '25May16', fl3, 'thinned skull')




# DB[22] = ('022', '023', 610, 8.0, 6.444, '15May10', fl1, 'dura, deeper focus')
# DB[24] = ('024', '025', 610, 29.0, 6.444, '15May10', fl1, 'dura, deeper focus')
# DB[26] = ('026', '027', 560, 29.0, 6.444, '15May10', fl1, 'dura, deeper focus') # light fluctuations; some phase shifts though
# DB[28] = ('028', '029', 560, 26.0, 6.444, '15May10', fl1, 'dura, focus near surface') # too many large fluctuations - can't trust
# DB[5] = ('005', '002', 610, 20, 4.25, '05Feb16', fl1, 'dura')
# # 24 May 10: used white LED and put 610 filter in front of camera (quieter illumination)
# # potential damage to cortex (bleeder)
# DB[32] = ('032', '033', 610, 5.0, 6.412, '24May10', fl1, 'dura, just below surface') # amplitude maps look similar; phase maps look good
# DB[34] = ('034', '035', 610, 30.0, 6.412, '24May10', fl1, 'dura, just below surface') #linear drift, but corrections ok; phase gradient
# DB[36] = ('037', '036', 610, 120.0, 6.412, '24May10', fl1, 'dura, just below surface') # no input to speaker; phase map somewhat flat
# DB[39] = ('039', '041', 610, 20.0, 6.482, '24May10', fl1, 'dura, just below surface') # illim steady; no clear resonse in phase map

# # 02 June 10: used white LED and green LED
# DB[42] = ('042', '043', 610, 5.0, 6.452, '02Jun10', fl1, 'thinned skull') # not too bad
# DB[44] = ('045', '044', 560, 5.0, 6.452, '02Jun10', fl1, 'thinned skull') # not bad; led is stable
# DB[48] = ('049', '048', 560, 5.0, 6.412, '02Jun10', fl1, 'thinned skull') # up has large drift - NG
# DB[50] = ('050', '051', 610, 20.0, 6.412, '02Jun10', fl1, 'thinned skull') # both have drift, but not many larg fluctuatiosn - map spotty
# DB[52] = ('052', '033', 610, 35.0, 6.422, '02Jun10', fl1, 'thinned skull, focussed slightly deeper') # many large light flucturtions
# DB[54] = ('054', '055', 610, 120.0, 6.412, '02Jun10', fl1, 'thinned skull') # no stim control
# DB[56] = ('056', '057', 560, 120.0, 6.412, '02Jun10', fl1, 'thinned skull') # no stim control
# # changed frequency map for next runs on 02 June 2010
# DB[60] = ('061', '060', 560, 10.0, 4.276, '02Jun10', fl2, 'thinned skull') # large drift on direction
# DB[62] = ('062', '063', 610, 10.0, 4.276, '02Jun10', fl2, 'thinned skull') # drift and noise
# DB[64] = ('064', '065', 610, 30.0, 4.274, '02Jun10', fl2, 'thinned skull') # possibly OK - clean illumination
# DB[66] = ('066', '067', 610, 15.0, 4.274, '02Jun10', fl2, 'thinned skull') # Might be good!!!!

# # 09 June 10: QuantEM512SC for imaging 
# DB[68] = ('068', '069', 610, 15.0, 4.228, '09Jun10', fl2, 'thinned skull') # focus near surface Might be good!!!! diagonal phase gradient
# DB[70] = ('070', '071', 610, 15.0, 4.228, '09Jun10', fl2, 'thinned skull') # focus deeper Might be good!!!! Diagonal phase gradient - but horizontal stripe too
# DB[73] = ('073', '074', 610, 15.0, 4.228, '09Jun10', fl2, 'thinned skull') # same as 68/70, but no stimulation Might be good!!!!
# DB[75] = ('075', '076', 560, 15.0, 4.224, '09Jun10', fl2, 'thinned skull') # Green light, 120 msec integration time phase map with structure
# DB[77] = ('077', '078', 560, 25.0, 4.248, '09Jun10', fl2, 'thinned skull') # Green light, 151 msec integration time - a little noisy ?
# DB[79] = ('079', '081', 610, 15.0, 4.204, '09Jun10', fl2, 'thinned skull') # 610, 30 fps, 30 msec integration time way too big to handle
# DB[82] = ('082', '085', 610, 15.0, 4.204, '09Jun10', fl2, 'thinned skull') # 610, 30 fps, 30 msec integration time broken down, gradient, but horizontal stripe
# DB[83] = ('083', '086', 610, 15.0, 4.204, '09Jun10', fl2, 'thinned skull') # 610, 30 fps, 30 msec integration time -- Diagonal gradient 
# DB[84] = ('084', '087', 610, 15.0, 4.204, '09Jun10', fl2, 'thinned skull') # 610, 30 fps, 30 msec integration time -- Diagonal gradient

# DB[18] = ('018','015', 610, 5.0, 4.25, '05Feb16', fl1, 'thinned skull')
# DB[19] = ('019','014', 610, 5.0, 4.25, '05Feb16', fl1, 'thinned skull')
# #DB[18] = ('018','015', 610, 5.0, 4.25, '05Feb16', fl1, 'thinned skull')
# DB[17] = ('017','016', 610, 5.0, 4.25, '05Feb16', fl1, 'thinned skull')

D = []
d = []
measuredPeriod = 2.25
binsize = 4
gfilt = 0
#freqlist = numpy.logspace(3, 4.7, 12, base=10)
homedir = os.getenv('HOME')
workingpath = 'Desktop/IntrinsicImaging/video_'
basepath = os.path.join(homedir, workingpath)
#basepath = '/Volumes/Promise Pegasus/ManisLab_Data3/IntrinsicImaging/'
#basepath = '/Volumes/Time_Machine_Backups/2016.05.12_000/Sound_Stimulation_Apr_2016_'
basepath = '/Users/tjropp/Desktop/data/2016.05.25_000/Intrinsic_Stimulation_Camera_'
#basepath = '/Volumes/TRoppData/data/2016.05.16_000/Intrinsic_Stimulation_Camera_'

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
        self.phasex = []
        self.phasey = []
        self.nPhases = 5
        self.nCycles = 10
        
    def parse_and_go(self, argsin = None):
        global measuredPeriod
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
        parser.add_option("-p", '--period', dest = "period", default=8.0, type="float",
                          help = "Stimulus cycle period")
        parser.add_option("-c", '--cycles', dest = "cycles", default=0, type="int",
                          help = "# cycles to analyze")
        parser.add_option("-b", '--binning', dest = "binsize", default=0, type="int",
                          help = "bin reduction x,y")
        parser.add_option("-g", '--gfilter', dest = "gfilt", default=0, type="float",
                          help = "gaussian filter width")
        parser.add_option("-f", '--fdict', dest = "fdict", default=0, type="int",
                          help = "Use dictionary entry")
        if argsin is not None:
            (options, args) = parser.parse_args(argsin)
        else:
            (options, args) = parser.parse_args()

        if options.period is not None:
            measuredPeriod = options.period
        if options.cycles is not None:
            self.nCycles = options.cycles
        if options.binsize is not None:
            binsize = options.binsize
        if options.gfilt is not None:
            gfilt = options.gfilt

            #TFR- this code generates a test signal for running a test analysis sequence
        print options.test
        if options.test is True:
            print "Running Test Sample"
            period = 8.0 # period and frame sample rate can be different
            framerate = 8.0
            nper = 1
            d = 10.0*numpy.random.normal(size=(2500,128,128)).astype('float32')
            ds = d.shape
            self.nFrames = d.shape[0]
            self.nPhases = 10
            maxdel = 50
            self.phasex = []
            self.phasey = []
            for i in range(0,self.nPhases):
                dx = i*ds[1]/self.nPhases # each phase is assigned to a region
                baseline = 0.0
                self.resp = numpy.zeros((self.nFrames,))
                phaseDelay = 0.25*period+period*(float(i)/self.nPhases) # phase delay for this region from 0 to nearly the stimulus repeat period
               # print '********phase delay: ', phaseDelay
                for j in range(0, nper): # for each period 
                    tdelay = (float(j) * period) + phaseDelay # time to phase delay point
                    idelay = int(numpy.floor(tdelay*framerate)) # convert to frame position in frame space
               #     print '     tdel: ', tdelay, '    idel: ', idelay
                #    if idelay < self.nFrames-maxdel:
                #        self.resp[idelay:idelay+maxdel] = (i+1)*numpy.exp(-numpy.linspace(0, 2, maxdel)) # marks amplitudes as well
                self.resp = numpy.sin(
                         numpy.linspace(0, 2.0*numpy.pi*self.nFrames/(period*framerate), self.nFrames)+i*numpy.pi/8 - numpy.pi/2.0)
                d[:, dx:dx+int(ds[1]/self.nPhases), 5:int(ds[2]/2)] += self.resp[:, numpy.newaxis, numpy.newaxis]
                self.phasex.append( (2+(dx+int(ds[1]/self.nPhases))/2))
                self.phasey.append((6+int(ds[2]/2)/2)) # make the signal equivalent of digitized one (baseline 3000, signal at 1e-4 of baseline)
            d = (d*3000.0*1e-4)+3000.0 # scale and offset to match data scaling coming in
            self.imageData = d.astype('int16') # reduce to a 16-bit map to match camera data type
            self.times = numpy.arange(0, self.nFrames/framerate, 1.0/framerate)
            print "Test Image Created"
            self.Analysis_FourierMap(period = period, target = 1, mode=1, bins=binsize)
            print "Completed Analysis FourierMap"
            self.plotmaps_pg(mode = 2, gfilter = gfilt)
            print "Completed plot maps"

            return

        if options.fdict is not None:
            if options.fdict in DB.keys(): # populate options 
                options.upfile = DB[options.fdict][0]
                options.downfile = DB[options.fdict][1]
                options.period = DB[options.fdict][4]
            else:
               print "File %d NOT in DBase\n" % options.fdict
               return
        if options.directory is not None:
            self.directory = options.directory

        if options.upfile is not None:
            self.upfile = options.upfile
            target = 1
        
        if options.downfile is not None:
            self.downfile = options.downfile
            target = 2

        target = 0
        upf = None
        dwnf = None
        if options.upfile is not None:
            upf = basepath + options.upfile + '/Camera/frames.ma'
        if options.downfile is not None:
            dwnf = basepath + options.downfile + '/Camera/frames.ma'
        measuredPeriod=options.period
        for file in (upf, dwnf):
#if options.upfile is not None and options.downfile is not None:
            if file is None:
               break
            im=[]
            self.imageData = []
            print "loading data from ", file
            try:
                im = MetaArray(file = file,  subset=(slice(0,2), slice(64,128), slice(64,128)))
            except:
                print "Error loading upfile: %s\n" % file
                return
            print "data loaded"
            target = target + 1
            self.times = im.axisValues('Time').astype('float32')
            self.imageData = im.view(np.ndarray).astype('float32')
            im=[]
            if file is upf:
               upflag = 1
            else:
               upflag = 0
            self.Image_Background
            self.Image_Divided 
            self.Analysis_FourierMap(period=measuredPeriod, target = target,  bins=binsize, up=upflag)
        if target > 0:
            self.plotmaps_pg(mode = 1, target = target, gfilter = gfilt)
            # self.firstframe = pg.image(self.imageData[0], title='first frame of image')

    def Image_Background(self):
        self.background=[]
        background = self.imageData[:50]
        print 'shape of background:', np.shape(background)
        pg.image(np.mean(background[10:49],axis=0), title='average background ')

        self.background = np.mean(background,axis=0)
        return

    def Image_Divided(self):

        divided = np.zeros(np.shape(self.imageData), float)
        for i in range(self.imageData.shape[0]):
            if self.times[i]>=1:
                divided[i,:,:] = (self.imageData[i,:,:])
                #divided[i,:,:] = (self.imageData[i,:,:]-self.background)/self.background
        self.imageData = divided[50:]        
        #self.divided = divided[50:]
        #pg.image(subtracted, title='subtracted')
        #pg.image(self.divided,title='divided')    
        return

    def Analysis_FourierMap(self, period = 2.25, target = 1, mode=0, bins = 1, up=1):
        global D
        global measuredPeriod
        D = []
        self.DF = []
        self.avgimg = []
        self.stdimg = []
        self.nFrames =self.imageData.shape[0]
        self.imagePeriod = 0
        if HAVE_MPL:
            pylab.figure(2)
        
        print "Analysis Starting"
# first squeeze the image to 3d if it is 4d
        maxt = self.times[-1] # find last image time
        print "Duration of Image Stack: %9.3f s (%8.3f min)\n" % (maxt, maxt/60.0)
        sh = self.imageData.shape
        if len(sh) == 4:
           self.imageData = self.imageData.squeeze()
           sh = self.imageData.shape
        dt = numpy.mean(numpy.diff(self.times)) # get the mean dt
        print 'dt', dt
        self.imagePeriod = period# image period in seconds.
        print 'period:', self.imagePeriod
        w = 2.0 * numpy.pi * self.imagePeriod
        n_Periods = int(numpy.floor(maxt/self.imagePeriod)) # how many full periods in the image set?
        if self.nCycles > 0 and self.nCycles < n_Periods:
            n_Periods = self.nCycles
        n_PtsPerCycle = int(numpy.floor(self.imagePeriod/dt)); # estimate image points in a stimulus cycle
        ndt = self.imagePeriod/n_PtsPerCycle
        self.imageData = self.imageData[range(0, n_Periods*n_PtsPerCycle),:,:] # reduce to only what we need
        self.timebase = numpy.arange(0, self.imageData.shape[0]*dt, dt)# reduce data in blocks by averaging
        if mode == 0:
            ipx = self.imageData.shape[1]/2
            ipy = self.imageData.shape[2]/2
        else:
            ipx = 64
            ipy = 64
 
        if bins > 1:
            redx=bins
            redy=bins
            nredx = int(sh[1]/redx)
            nredy = int(sh[2]/redy)
            newImage = numpy.zeros((self.imageData.shape[0], nredx, nredy))
            print sh, nredx, nredy
            print self.imageData.shape, newImage.shape
            for i in range(0, nredx-1):
                for j in range(0, nredy-1):
    #                print i,j,i*redx,(i+1)*redx-1,j*redx,(j+1)*redy-1
                    newImage[:,i,j] = numpy.mean(numpy.mean(self.imageData[:,i*redx:(i+1)*redx-1, j*redy:(j+1)*redy-1],axis=2),axis=1)
            self.imageData = newImage
            sh = self.imageData.shape
            ipx = ipx/redx
            ipy = ipy/redy

        else:
            redx = bins
            redy = bins
        print "# Periods: %d  Pts/cycle: %d Cycle dt %8.4fs (%8.3fHz) Cycle: %7.4fs" %(n_Periods, n_PtsPerCycle, ndt, 1.0/ndt, self.imagePeriod)
        
        # get the average image and the average of the whole image over time
        self.avgimg = numpy.mean(self.imageData, axis=0) # get mean image for reference later: average across all time
        self.stdimg = numpy.std(self.imageData, axis= 0) # and standard deviation
        # timeavg is calculated on the central region only:

        self.timeavg = numpy.mean(numpy.mean(self.imageData[:,int(sh[1]*0.25):int(sh[1]*0.75),int(sh[2]*0.25):int(sh[2]*0.75)], axis=2),axis=1) # return average of entire image over time
        # I want to define the central region in a different way.  I was to define it at the area in the middle, where the signal is 
        # less that two standard deviations from the mean- I believe that this is where the signal lies.
        print 'size of self.timeavg', np.shape(self.timeavg)
        print " >>Before HPF: Noise floor (std/mean): %12.6f  largest std: %12.6f" % (numpy.mean(self.stdimg)/numpy.mean(self.avgimg), 
               numpy.amax(self.stdimg)/numpy.mean(self.avgimg))

        # color scheme: magenta with symbol is "raw" data for one pixel
        #               black is after averaged signal over space is subtracted over time
        #               red is after both corrections (boxcar and time acverage)
        zid = self.imageData[:,ipx,ipy]-self.timeavg
        print 'size of zid', np.shape(zid)
        mta = scipy.signal.detrend(self.timeavg)
        mtaa = numpy.mean(mta, axis=0)
        stdta = numpy.std(mta)
        rjstd = 3.0*stdta
        pts = len(self.timeavg)
        reject = numpy.where(numpy.abs(mta) < rjstd)
        #print 'reject', reject
        trej = numpy.array(self.timebase[reject])
        LPF = 0.2/dt
        lfilt = SignalFilter_LPFBessel(scipy.signal.detrend(zid, axis=0), LPF, samplefreq=1.0/dt , NPole = 8, reduce = False)
        if HAVE_MPL:
            p1=pylab.subplot(3,1,1)
            p1.plot(self.timebase, self.imageData[:,ipx,ipy] - numpy.mean(self.imageData[:,ipx,ipy]), 'mo-') # prior to any correction
            
            p1.plot(self.timebase, zid-numpy.mean(zid), 'k-') # after subtracting time averaged
            p3=pylab.subplot(3,1,2)
            
            p3.plot(self.timebase, mta, 'g-')
        
        
            p3.plot([0,numpy.amax(self.timebase)], [mtaa+rjstd,mtaa+rjstd], 'g--' )
            p3.plot([0,numpy.amax(self.timebase)], [mtaa-rjstd,mtaa-rjstd], 'g--')
        
#        print trej.shape()
#        print mta[:,reject].shape()
#        p3.plot(trej, mta[:,reject], 'rx')
        # calculate PSD of data
            amp, freqs = mlab.psd(scipy.signal.detrend(zid, axis=0), Fs=1.0/dt )
        
            amp2, freqs2 = mlab.psd(scipy.signal.detrend(self.imageData[:,ipx,ipy], axis=0), Fs=1.0/dt )
            amp3, freqs3 = mlab.psd(scipy.signal.detrend(lfilt, axis=0), Fs=1.0/dt )
            p2 = pylab.subplot(3,1,3)
            p2.loglog(freqs, amp, 'k-')
            p2.loglog(freqs2, amp2, 'mo-')
            p2.loglog(freqs3, amp3, 'cs-')
        # subtract slow fluctuations
        flpf = float(LPF)
        sf = float(1.0/dt)
        wn = [flpf/(sf/2.0)]
        NPole=8
        filter_b,filter_a=scipy.signal.bessel(
                NPole,
                wn,
                btype = 'low',
                output = 'ba')
        print "boxcar HPF"
        for i in range(0, self.imageData.shape[1]):
            for j in range(0, self.imageData.shape[2]):
                self.imageData[:,i,j] = self.imageData[:,i,j] - self.timeavg
# OLD: stsci not available anymore
#               box_2D_kernel = astropy.convolve.Box2DKernel(2*n_PtsPerCycle)
#               box_2D_kernel = Box2DKernel(5)
                box_2D_kernel = Box1DKernel(2*n_PtsPerCycle)
#               print self.imageData[:,i,j]
#               print len(self.imageData[:,i,j])
#               print box_2D_kernel
                self.imageData[:,i,j] = self.imageData[:,i,j] - convolve_fft(self.imageData[:,i,j], box_2D_kernel) 
#                self.imageData[:,i,j] = self.imageData[:,i,j] - scipy.stsci.convolve.boxcar(self.imageData[:,i,j], (2*n_PtsPerCycle,)) 
                self.imageData[:,i,j]=scipy.signal.lfilter(filter_b, filter_a, scipy.signal.detrend(self.imageData[:,i,j], axis=0)) # filter the incoming signal
        zid = self.imageData[:,ipx,ipy]
        lfilt = SignalFilter_LPFBessel(scipy.signal.detrend(zid, axis=0), LPF, samplefreq=1.0/dt , NPole = 8, reduce = False)
        if HAVE_MPL:
            p1.plot(self.timebase, zid - numpy.mean(zid), 'r-')
            p1.plot(self.timebase, lfilt - numpy.mean(lfilt), 'c-')
            amp2, freqs2 = mlab.psd(scipy.signal.detrend(self.imageData[:,ipx,ipy], axis=0), Fs=1.0/dt )
            p2.loglog(freqs2, amp2, 'r')
            ymin, ymax = p2.get_ylim()
            p2.set_ylim((0.01, ymax))
        self.stdimg = numpy.std(self.imageData, axis= 0) # and standard deviation
        print " >>after HPF: Noise floor (std/mean): %12.6f  largest std: %12.6f" % (numpy.mean(self.stdimg)/numpy.mean(self.avgimg), 
               numpy.amax(self.stdimg)/numpy.mean(self.avgimg))
        
        print "now reshaping"
        self.n_times = numpy.arange(0, n_PtsPerCycle*ndt, ndt) # just one cycle
        #self.n_freqs = (1/ndt)*numpy.arange(0,n_PtsPerCycle*ndt,(1/))
        # put data into new shape to prepare for mean. "Folds" data by cycles". Also multiply to make average work
        self.imageData = numpy.reshape(self.imageData, 
                         (n_Periods, n_PtsPerCycle, sh[1], sh[2])).astype('float32')

        print "now calculating mean"
        # excluding bad trials
        trials = range(0, n_Periods)
        print 'trials', trials
        print n_PtsPerCycle
        reject = reject[0]
        for i in range(0,len(reject)):
            t = reject[i]/n_PtsPerCycle
            #if t in trials:
                #trials.remove(t)
        print "retaining trials: ", trials
        D = numpy.mean(self.imageData[trials,:,:,:], axis=0).astype('float32') # /divider # get mean of the folded axes.
        print "mean calculated, now detrend and fft"
        # detrend before taking fft
        D = scipy.signal.detrend(D, axis=0)
        # calculate FFT and get amplitude and phase
        self.DF = numpy.fft.fft(D, axis = 0)

        ampimg = numpy.abs(self.DF[1,:,:]).astype('float32') #changing this to 3 instead of 1 (same next line)
        phaseimg = numpy.angle(self.DF[1,:,:]).astype('float32')
        if target == 1:
            f = open('img_phase1.dat', 'w')
            pickle.dump(phaseimg, f)
            f.close()
            f = open('img_amplitude1.dat', 'w')
            pickle.dump(ampimg, f)
            f.close()
            self.amplitudeImage1 = ampimg
            self.phaseImage1 = phaseimg
            self.DF1=self.DF
            # f = open('DF1.mat', 'w')
            # f.close()
            # scipy.io.savemat('/Users/tessajonneropp/Desktop/data/signalAlign/DF1.mat', mdict={'DF1': self.DF})
   
            # f = open('times1.mat', 'w')
            # f.close()
            # scipy.io.savemat('/Users/tessajonneropp/Desktop/data/signalAlign/times1.mat', mdict={'times1': self.n_times})
          
        if target == 2:
            f = open('img_phase2.dat', 'w')
            pickle.dump(phaseimg, f)
            f.close()
            f = open('img_amplitude2.dat', 'w')
            pickle.dump(ampimg, f)
            f.close()
            self.amplitudeImage2 = ampimg
            self.phaseImage2 = phaseimg
            self.DF2=self.DF
            # f = open('DF2.mat', 'w')
            # f.close()
            # scipy.io.savemat('/Users/tessajonneropp/Desktop/data/signalAlign/DF2.mat', mdict={'DF2': self.DF})

            # f = open('times2.mat','w')
            # f.close()
            # scipy.io.savemat('/Users/tessajonneropp/Desktop/data/signalAlign/times2.mat', mdict={'times2': self.n_times})
            
        print "fft calculated, data  saveddata"
        # save most recent calculation to disk

    def sub_func(self, a, avg):
        return(a - avg)


# plot data
    def plotmaps(self, mode = 0, target = 1, gfilter = 0):
        global D
        max1 = numpy.amax(self.amplitudeImage1)
        if target > 1:
            max1 = numpy.amax([max1, numpy.amax(self.amplitudeImage2)])
        max1 = 10.0*int(max1/10.0)
        pylab.figure(1)
        pylab.subplot(2,3,1)
        pylab.title('Amplitude Map1')
        #scipy.ndimage.gaussian_filter(self.amplitudeImage1, 2, order=0, output=self.amplitudeImage1, mode='reflect')
        imga1 = pylab.imshow(scipy.ndimage.gaussian_filter(self.amplitudeImage1,gfilt, order=0, mode='reflect'))
        pylab.colorbar()
        imga1.set_clim = (0.0, max1)
        pylab.subplot(2,3,4)
        pylab.title('Phase Map1')
        imgp1 = pylab.imshow(scipy.ndimage.gaussian_filter(self.phaseImage1, gfilt, order=0,mode='reflect'), cmap=matplotlib.cm.hsv)
        imgp1.set_clim=(-numpy.pi/2.0, numpy.pi/2.0)
        pylab.colorbar()

        print "plotmaps Block 1"
        print "mode:", mode
        if mode == 0:
            pylab.subplot(2,3,3)
            for i in range(0, self.nPhases):
                pylab.plot(ta.n_times, D[:,5,5].view(ndarray))
                #pylab.plot(self.n_times, D[:,i*55+20, 60])
                pylab.hold('on')
            pylab.title('Waveforms')

            pylab.subplot(2,3,6)
            for i in range(0, self.nPhases):
                pylab.plot(ta.n_times, self.DF[:,5,5].view(ndarray))
                #pylab.plot(self.DF[:,i*55+20, 60])
                pylab.hold('on')
            pylab.title('FFTs')

        print "plotmaps Block 2"

        if mode == 1 and target > 1:
            pylab.subplot(2,3,2)
            pylab.title('Amplitude Map2')
            #scipy.ndimage.gaussian_filter(self.amplitudeImage2, 2, order=0, output=self.amplitudeImage2, mode='reflect')
            imga2 = pylab.imshow(scipy.ndimage.gaussian_filter(self.amplitudeImage2,gfilt, order=0, mode='reflect'))
            imga2.set_clim = (0.0, max1)
            pylab.colorbar()
            pylab.subplot(2,3,5)
            imgp2 = pylab.imshow(scipy.ndimage.gaussian_filter(self.phaseImage2, gfilt, order=0,mode='reflect'), cmap=matplotlib.cm.hsv)
            pylab.colorbar()
            imgp2.set_clim=(-numpy.pi/2.0, numpy.pi/2.0)
            pylab.title('Phase Map2')
            # doubled phase map
            pylab.subplot(2,3,6)
            #scipy.ndimage.gaussian_filter(self.phaseImage2, 2, order=0, output=self.phaseImage2, mode='reflect')
            np1 = scipy.ndimage.gaussian_filter(self.phaseImage1, gfilt, order=0, mode='reflect')
            np2 = scipy.ndimage.gaussian_filter(self.phaseImage2, gfilt, order=0, mode='reflect')
            dphase = (np1 + np2)/2
            #dphase = self.phaseImage1 - self.phaseImage2
           
            #scipy.ndimage.gaussian_filter(dphase, 2, order=0, output=dphase, mode='reflect')
            imgpdouble = pylab.imshow(dphase, cmap=matplotlib.cm.hsv)
            pylab.title('2x Phi map')
            pylab.colorbar()
            imgpdouble.set_clim=(-numpy.pi, numpy.pi)

        print "plotmaps Block 3"

        if mode == 2 or mode == 1:
            if self.phasex == []:
                self.phasex = numpy.random.randint(0, high=D.shape[1], size=D.shape[1])
                self.phasey = numpy.random.randint(0, high=D.shape[2], size=D.shape[2])

            pylab.subplot(2,3,3)
            sh = D.shape
            spr = sh[2]/self.nPhases
            for i in range(0, self.nPhases):
                Dm = self.avgimg[i*spr,i*spr] # diagonal run
                pylab.plot(self.n_times, 100.0*(D[:,self.phasex[i], self.phasey[i]]/Dm))
                pylab.hold('on')
            pylab.title('Waveforms')

        print "plotmaps Block 4"

        if mode == 2:
            pylab.subplot(2,3,6)
            for i in range(0, self.nPhases):
                pylab.plot(self.DF[1:,80, 80])
                pylab.hold('on')
            pylab.title('FFTs')

        print "plotmaps Block 5"

        pylab.show()

    def plotmaps_pg(self, mode = 0, target = 1, gfilter = 0):

        # # ## Set up plots/images in window
        # self.view = pg.GraphicsView()
        # l = pg.GraphicsLayout(border=(100,100,100))
        # self.view.setCentralItem(l)
        # self.amp1View = l.addViewBox(lockAspect=True)
        # self.amp2View = l.addViewBox(lockAspect=True)
        # self.waveformPlot = l.addPlot(title="Waveforms")
        # l.nextRow()
        # self.phase1View = l.addViewBox(lockAspect=True)
        # self.phase2View = l.addViewBox(lockAspect=True)
        # self.fftPlot = l.addPlot(title="FFTs")
        # self.phiView = l.addViewBox(lockAspect=True)

        global D
        max1 = numpy.amax(self.amplitudeImage1)
        if target > 1:
            max1 = numpy.amax([max1, numpy.amax(self.amplitudeImage2)])
        max1 = 10.0*int(max1/10.0)
        # pylab.figure(1)
        # pylab.subplot(2,3,1)
        # pylab.title('Amplitude Map1')
        # #scipy.ndimage.gaussian_filter(self.amplitudeImage1, 2, order=0, output=self.amplitudeImage1, mode='reflect')
        ampimg = scipy.ndimage.gaussian_filter(self.amplitudeImage1,gfilt, order=0, mode='reflect')
        #self.amp1View.addItem(pg.ImageItem(ampimg))
        self.amp1 = pg.image(ampimg, title="Amplitude Map 1", levels=(0.0, max1))
        #imga1 = pylab.imshow(ampimg)
        #pylab.colorbar()
        #imga1.set_clim = (0.0, max1)
        #pylab.subplot(2,3,4)
        #pylab.title('Phase Map1')
        phsmap=scipy.ndimage.gaussian_filter(self.phaseImage1, gfilt, order=0,mode='reflect')
        #self.phase1View.addItem(pg.ImageItem(phsmap))
        self.phs1 = pg.image(phsmap, title='Phase Map 1')
        #self.phs1.getHistogramWidget().item.gradient.
        #imgp1 = pylab.imshow(phsmap, cmap=matplotlib.cm.hsv)
        #pylab.colorbar()

        print "plotmaps Block 1"
        print "mode:", mode
        self.wavePlt = pg.plot(title='Waveforms')
        self.freqPlt = pg.plot(title='frequency axis Waveforms')
        if mode == 0 or mode == 2:
            self.fftPlt = pg.plot(title = 'FFTs')
        
        if mode == 0:
            #pylab.subplot(2,3,3)

            for i in range(0, self.nPhases):
                self.wavePlt.plot(ta.n_times, D[:,5,5].view(ndarray))
                print 'max wave', np.max(D[:,5,5])
                #pylab.plot(ta.n_times, D[:,5,5].view(ndarray))
                #pylab.plot(self.n_times, D[:,i*55+20, 60])
                #pylab.hold('on')
            #pylab.title('Waveforms')

            #pylab.subplot(2,3,6)
            for i in range(0, self.nPhases):
                self.fftPlt.plot(ta.n_times, self.DF[:,5,5].view(ndarray))
                #pylab.plot(ta.n_times, self.DF[:,5,5].view(ndarray))
                #pylab.plot(self.DF[:,i*55+20, 60])
                #pylab.hold('on')
            #pylab.title('FFTs')

        print "plotmaps Block 2"

        if mode == 1 and target > 1:
            #pylab.subplot(2,3,2)
            #pylab.title('Amplitude Map2')
            #scipy.ndimage.gaussian_filter(self.amplitudeImage2, 2, order=0, output=self.amplitudeImage2, mode='reflect')
            ampImg2 = scipy.ndimage.gaussian_filter(self.amplitudeImage2,gfilt, order=0, mode='reflect')
            #imga2 = pylab.imshow(ampImg2)
            #self.amp2View.addItem(pg.ImageItem(ampImg2))
            self.amp2 = pg.image(ampImg2, title='Amplitude Map 2', levels=(0.0, max1))
            #imga2.set_clim = (0.0, max1)
            #pylab.colorbar()
            #pylab.subplot(2,3,5)
            phaseImg2 = scipy.ndimage.gaussian_filter(self.phaseImage2, gfilt, order=0,mode='reflect') 
            #self.phase2View.addItem(pg.ImageItem(phaseImg2))
            self.phs2 = pg.image(phaseImg2, title="Phase Map 2", levels=(-np.pi/2.0, np.pi/2.0))
            #imgp2 = pylab.imshow(phaseImg2, cmap=matplotlib.cm.hsv)
            #pylab.colorbar()
            #imgp2.set_clim=(-numpy.pi/2.0, numpy.pi/2.0)
            #pylab.title('Phase Map2')
            ### doubled phase map
            #pylab.subplot(2,3,6)
            #scipy.ndimage.gaussian_filter(self.phaseImage2, 2, order=0, output=self.phaseImage2, mode='reflect')
            np1 = scipy.ndimage.gaussian_filter(self.phaseImage1, gfilt, order=0, mode='reflect')
            np2 = scipy.ndimage.gaussian_filter(self.phaseImage2, gfilt, order=0, mode='reflect')
            dphase = (np1 + np2)/2
            twophi = np.angle(self.DF1[1])*np.angle(self.DF2[1])/(np.pi)
            #dphase = self.phaseImage1 - self.phaseImage2
           
            #scipy.ndimage.gaussian_filter(dphase, 2, order=0, output=dphase, mode='reflect')
            #self.phiView.addItem(pg.ImageItem(dphase))
            self.phi = pg.image(dphase, title="2x Phi map", levels=(-np.pi, np.pi))
            self.twophi = pg.image(twophi, title='2X Phi Map, using multiplication')

            #self.dphaseReshape(data=dphase)
            #imgpdouble = pylab.imshow(dphase, cmap=matplotlib.cm.hsv)
            #pylab.title('2x Phi map')
            #pylab.colorbar()
            #imgpdouble.set_clim=(-numpy.pi, numpy.pi)

        print "plotmaps Block 3"

        if mode == 2 or mode == 1:
            if self.phasex == []:
                self.phasex = numpy.random.randint(0, high=D.shape[1], size=D.shape[1])
                self.phasey = numpy.random.randint(0, high=D.shape[2], size=D.shape[2])

            #pylab.subplot(2,3,3)
            sh = D.shape
            spr = sh[2]/self.nPhases
            wvfms=[]
            for i in range(0, self.nPhases):
                Dm = self.avgimg[i*spr,i*spr] # diagonal run
                #wvfms=self.n_times, 100.0*(D[:,self.phasex[i], self.phasey[i]]/Dm)
                #pylab.plot(self.n_times, 100.0*(D[:,self.phasex[i], self.phasey[i]]/Dm))
                self.wavePlt.plot(self.n_times, 100.0*(D[:,self.phasex[i], self.phasey[i]]/Dm))
                #self.wavePlt.plot(100.0*(D[:,self.phasex[i], self.phasey[i]]/Dm)) #plotting on Waveforms.
                #self.freqPlt.plot(self.n_freqs, 100.0*(D[:,self.phasex[i], self.phasey[i]]/Dm))
                #pylab.hold('on')
                #self.plotlist.append(pg.image(wvfms, title="Waveforms"))
                #print "it worked"
            #pylab.title('Waveforms')

        print "plotmaps Block 4"

        if mode == 2:
            #pylab.subplot(2,3,6)
            for i in range(0, self.nPhases):
                #pylab.plot(self.DF[1:,80, 80])
                #self.fftPlt.plot(self.DF[1:,80,80]) ## causing errors and i'm not sure what the desired thing is, Exception: Can not plot complex data types.
                pass
                #pylab.hold('on')
            #pylab.title('FFTs')

        print "plotmaps Block 5"

        #pylab.show()
        #self.view.show()

    def meanxy(self, indata, n, m):
        """ compute a mean in the xy plane of indata, over an area nxm
            the return is the reduced mean array. Note that rHS and bottom parts
            may be lost, depending on whether n and/or m are equally divisible into
            the x and y dimensions """
# there must be a more efficient way to do this... this is SLOW... 
        sh = indata.shape
        newsh = (sh[0], sh[1]/n, sh[2]/m)
        result = numpy.zeros(newsh) # the new array
        ji=[]
        ki = []
        for j in range(0, newsh[1]): # precalc indices
            ji.append(range(j*n,(j+1)*n))
        for k in range(0, newsh[2]):
            ki.append(range(k*m,(k+1)*m))
        for i in range(0, sh[0]): # do not flattend the planes
            for j in range(0, newsh[1]):
                for k in range(0, newsh[2]):
                    result[i, j, k] = indata[i, ji[j], ki[k]].mean()
        return result

    def dphaseReshape(self, data):
        altdphase = data
        for i in range(altdphase.shape[0]):
            for j in range(altdphase.shape[1]):
                #for k in range(dphase.shape[2]):
                if altdphase[i,j]<-4*np.pi/5 or altdphase[i,j]>4*np.pi/5:
                    altdphase[i,j] = 10
                elif altdphase[i,j]<-2*np.pi/5:
                    altdphase[i,j] = 20
                elif altdphase[i,j]<0:
                    altdphase[i,j] = 30
                elif altdphase[i,j]<2*np.pi/5:
                    altdphase[i,j] = 40
                else:
                    altdphase[i,j] = 50
        
        self.redphase = pg.image(altdphase, title="transformed 2X Phi map", levels=(10, 50))
        return
#### This function is copied from pylibrary.Utility. It is here locally so we don't need the dependencies that pylibrary requires
def SignalFilter_LPFBessel(signal, LPF, samplefreq, NPole=8, reduce=False, debugFlag=False):
    """Low pass filter a signal with a Bessel filter

        Digitally low-pass filter a signal using a multipole Bessel filter
        filter. Does not apply reverse filtering so that result is causal.
        Possibly reduce the number of points in the result array.

        Parameters
        ----------
        signal : a numpy array of dim = 1, 2 or 3. The "last" dimension is filtered.
            The signal to be filtered.
        LPF : float
            The low-pass frequency of the filter (Hz)
        samplefreq : float
            The uniform sampling rate for the signal (in seconds)
        NPole : int
            Number of poles for Butterworth filter. Positive integer.
        reduce : boolean (default: False)
            If True, subsample the signal to the lowest frequency needed to 
            satisfy the Nyquist critera.
            If False, do not subsample the signal.

        Returns
        -------
        w : array
            Filtered version of the input signal
    """

    if debugFlag:
        print "sfreq: %f LPF: %f HPF: %f" % (samplefreq, LPF)
    flpf = float(LPF)
    sf = float(samplefreq)
    wn = [flpf/(sf/2.0)]
    reduction = 1
    if reduce:
        if LPF <= samplefreq/2.0:
            reduction = int(samplefreq/LPF)
    if debugFlag is True:
        print "signalfilter: samplef: %f  wn: %f,  lpf: %f, NPoles: %d " % (
           sf, wn, flpf, NPole)
    filter_b,filter_a=scipy.signal.bessel(
            NPole,
            wn,
            btype = 'low',
            output = 'ba')
    if signal.ndim == 1:
        sm = np.mean(signal)
        w=scipy.signal.lfilter(filter_b, filter_a, signal-sm) # filter the incoming signal
        w = w + sm
        if reduction > 1:
            w = scipy.signal.resample(w, reduction)
        return(w)
    if signal.ndim == 2:
        sh = np.shape(signal)
        for i in range(0, np.shape(signal)[0]):
            sm = np.mean(signal[i,:])
            w1 = scipy.signal.lfilter(filter_b, filter_a, signal[i,:]-sm)
            w1 = w1 + sm
            if reduction == 1:
                w1 = scipy.signal.resample(w1, reduction)
            if i == 0:
                w = np.empty((sh[0], np.shape(w1)[0]))
            w[i,:] = w1
        return w
    if signal.ndim == 3:
        sh = np.shape(signal)
        for i in range(0, np.shape(signal)[0]):
            for j in range(0, np.shape(signal)[1]):
                sm = np.mean(signal[i,j,:])
                w1 = scipy.signal.lfilter(filter_b, filter_a, signal[i,j,:]-sm)
                w1 = w1 + sm
                if reduction == 1:
                    w1 = scipy.signal.resample(w1, reduction)
                if i == 0 and j == 0:
                    w = np.empty((sh[0], sh[1], np.shape(w1)[0]))
                w[i,j,:] = w1
        return(w)
    if signal.ndim > 3:
        print "Error: signal dimesions of > 3 are not supported (no filtering applied)"
        return signal

if __name__ == "__main__":
    

    ta=testAnalysis()  # create instance (for debugging)
    ta.parse_and_go(sys.argv[1:])

    app.exec_()
# ta.Analysis_FourierMap(sys.argv[1:])
   