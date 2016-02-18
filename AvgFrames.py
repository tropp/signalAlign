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

#import acq4.util
import pyqtgraph as pg #added to deal with plottng issues TFR 11/13/15
#import scipy.stsci.convolve
#import astropy.convolution
#from astropy.convolution import convolve_fft, convolve, Box2DKernel, Box1DKernel
#from astropy import image
import pickle
import matplotlib
import matplotlib.mlab as mlab
import pylab
from PyQt4 import QtGui
from skimage import feature
# try:
#     import matplotlib
#TFR 11/13/15 inserted the following line to try to resolve issue with pylab.show
# #matplotlib.rcParams['backend'] = "QtAgg"
#     import matplotlib.mlab as mlab
#     import pylab
#     from matplotlib import rc
#     rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#     ## for Palatino and other serif fonts use:
#     #rc('font',**{'family':'serif','serif':['Palatino']})
#     rc('text', usetex=True)
#     HAVE_MPL = True
# except:
#     HAVE_MPL = False


import pyqtgraph.configfile as configfile
from pyqtgraph.metaarray import MetaArray
#import pylibrary.Utility as Utils
#from pylibrary.Utility import SignalFilter_LPFBessel
from optparse import OptionParser

app = pg.Qt.QtGui.QApplication([])




#randomstuff= np.random.normal(size=(128,100,128))
D = []
d = []
measuredPeriod = 6.444
binsize = 4
gfilt = 0

freqlist = np.logspace(0, 4, num=17, base=2.0)
fl = [3000*x for x in freqlist]
print 'fl:', fl

# Keys are first file #. Data are file name (up, down), wavelength, attn, period, date, frequency list, comment
DB = {5: ('005', '003', 610,50.0, 4.25, '05Feb16', fl, 'thinned skull')}
DB[6] = ('006','003', 610, 30.0, 4.25, '05Feb16', fl, 'thinned skull')
DB[7] = ('007','003', 610, 30.0, 4.25, '05Feb16', fl, 'thinned skull')
DB[8] = ('008','003', 610, 30.0, 4.25, '05Feb16', fl, 'thinned skull')
DB[9] = ('009','003', 610, 30.0, 4.25, '05Feb16', fl, 'thinned skull')
DB[10] = ('010','003', 610, 30.0, 4.25, '05Feb16', fl, 'thinned skull')
DB[35] = ('035','008', 610, 30.0, 4.25, '05Feb16', fl, 'thinned skull')
DB[36] = ('036','008', 610, 30.0, 4.25, '05Feb16', fl, 'thinned skull')
DB[37] = ('037','003', 610, 30.0, 4.25, '05Feb16', fl, 'thinned skull')
DB[38] = ('038','008', 610, 30.0, 4.25, '05Feb16', fl, 'thinned skull')
DB[39] = ('039','008', 610, 30.0, 4.25, '05Feb16', fl, 'thinned skull')
DB[40] = ('040','008', 610, 30.0, 4.25, '05Feb16', fl, 'thinned skull')
DB[41]= ('041','008', 610, 30.0, 4.25, '05Feb16', fl, 'thinned skull')


# # Timestamps.  Keys are first file number.  Data are videoup, audioup, videodown, audiodown start times
# timestamp = {11: (1452278712.382, 1452278713.787, 1452278790.432, 1452278791.642)}
# timestamp[19] =  (1452713884.635, 1452713885.25, 1452713843.612, 1452713844.84)
# timestamp[4] = (1452712481.09, 1452712482.456, 1452712288.048, 1452712289.394)             
# timestamp[5] = (1452712556.376, 1452712557.921, 1452712371.056, 1452712373.436)
# timestamp[7] = (1452712735.321, 1452712736.654, 1452712778.485, 1452712779.631)
# timestamp[12] = (1452713276.699, 1452713277.68, 1452713150.485, 1452713151.894)
           

homedir = os.getenv('HOME')
videobasepath = '/Volumes/TRoppData/data/Intrinsic_data/2016.02.09_000/animal_000/video_'
basepath = '/Volumes/TRoppData/data/Intrinsic_data/2016.02.09_000/animal_000/'
audiobasepath = '/Volumes/TRoppData/data/Intrinsic_data/2016.02.09_000/animal_000/Sound_Stimulation_'
# videobasepath = '/Volumes/TRoppData/data/Intrinsic/2016.02.01_000/Second_round/video_'
# basepath = '/Volumes/TRoppData/data/Intrinsic/2016.02.01_000/Second_round/'
# audiobasepath = '/Volumes/TRoppData/data/Intrinsic/2016.02.01_000/Second_round/Sound_Stimulation_video_'
###  Don't forget to change the backgrounf file!

# videobasepath = '/Volumes/TRoppData/data/Intrinsic/2016.02.01_000/First_round/video_'
# basepath = '/Volumes/TRoppData/data/Intrinsic/2016.02.01_000/First_round/'
# audiobasepath = '/Volumes/TRoppData/data/Intrinsic/2016.02.01_000/First_round/Sound_Stimulation_video_'
# videobasepath = '/Volumes/TRoppData/data/Intrinsic/2016.01.13_000/video_'
# basepath = '/Volumes/TRoppData/data/Intrinsic/2016.01.13_000/'
# audiobasepath = '/Volumes/TRoppData/data/Intrinsic/2016.01.13_000/Sound_Stimulation_video_'
# videobasepath = '/Volumes/TRoppData/data/Intrinsic/2016.01.08_000/Intrinsic_Mapping/video_'
# basepath = '/Volumes/TRoppData/data/Intrinsic/2016.01.08_000/Intrinsic_Mapping/'
# audiobasepath = '/Volumes/TRoppData/data/Intrinsic/2016.01.08_000/Intrinsic_Mapping/Sound_Stimulation_video_'
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
        self.nPhases = 17
        self.nCycles = 3
        
    def parse_and_go(self, argsin = None):
        global period
        global binsize
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
                self.resp = 1000.0*numpy.sin(
                         numpy.linspace(0, 2.0*numpy.pi*self.nFrames/(period*framerate), self.nFrames)+i*numpy.pi/8.0 - numpy.pi/2.0)
                d[:, dx:dx+int(ds[1]/self.nPhases), 5:int(ds[2]/2)] += self.resp[:, numpy.newaxis, numpy.newaxis]
                self.phasex.append( (2+(dx+int(ds[1]/self.nPhases))/2))
                self.phasey.append((6+int(ds[2]/2)/2)) # make the signal equivalent of digitized one (baseline 3000, signal at 1e-4 of baseline)
            d = (d*3000.0*1e-4)+3000.0 # scale and offset to match data scaling coming in
            self.imageData = d.astype('int16') # reduce to a 16-bit map to match camera data type
            self.times = numpy.arange(0, self.nFrames/framerate, 1.0/framerate)
            print "Test Image Created"
            getout2 = fac_lift(self.imageData, self.times)
            self.imageData=getout2
            self.Analysis_FourierMap_TFR(period=period, target = 1, mode=1, bins=binsize)
            print "Completed Analysis FourierMap"
            self.plotmaps_pg(mode = 2, gfilter = 0)
            print "Completed plot maps"

        if options.period is not None:
            measuredPeriod = options.period
        if options.cycles is not None:
            self.nCycles = options.cycles
        if options.binsize is not None:
            binsize = options.binsize
        if options.gfilt is not None:
            gfilt = options.gfilt
        if options.upfile is not None:
            self.upfile = options.upfile
            target = 1
        
        if options.downfile is not None:
            self.downfile = options.downfile
            target = 2

        target = 0
        videoupf = None
        videodwnf = None
        audioupf = None
        audiodwnf = None

        
        print 'DB keys', DB.keys()
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
            videoupf = videobasepath + options.upfile + '.ma'
            audioupf = audiobasepath + options.upfile + '/DaqDevice.ma'
        if options.downfile is not None:
            videodwnf = videobasepath + options.downfile + '.ma'
            audiodwnf = audiobasepath + options.downfile + '/DaqDevice.ma'

        
        im=[]
        self.imageData = []
        print "loading data from ", videoupf
        try:
            im = MetaArray(file = videoupf,  subset=(slice(0,2), slice(64,128), slice(64,128)))
        except:
            print "Error loading upfile: %s\n" % videoupf
            return
        print "data loaded"
        
         
        rawtimes=[]
        rawimageData=[]
        rawtimes = im.axisValues('Time').astype('float32')

        rawimageData = im.view(np.ndarray).astype('float32')
#   
        #reads the timestamps from the files
        indexFile = configfile.readConfigFile(basepath+'.index') 
        timestampup = indexFile.__getitem__('video_'+DB[options.fdict][0]+'.ma')[u'__timestamp__']
        audioupindex = configfile.readConfigFile(audiobasepath+DB[options.fdict][0]+'/.index')
        audioupstamp = audioupindex.__getitem__(u'.')[u'__timestamp__'] 
 
       
        diffup = audioupstamp - timestampup

        
        
        audio = MetaArray(file = audioupf, subset=(slice(0,2), slice(64,128), slice(64,128)))
        audiotime = audio.axisValues('Time').astype('float32')
        audiomin = np.min(audiotime) + diffup
        audiomax = np.max(audiotime) + diffup
        
        print 'audiomin', audiomin
        print 'audiomax', audiomax

        adjustedtime = rawtimes[np.logical_and(rawtimes <= audiomax+5, rawtimes >= audiomin)]
        frame_start=np.amin(np.where(rawtimes >= audiomin))
        frame_end=np.amax(np.where(rawtimes <= audiomax+4))
        adjustedimagedata = rawimageData[frame_start:frame_end]
        #adjustedimagedata = rawimageData[np.logical_and(rawtimes <= audiomax+.5, rawtimes >= audiomin)]
 
        self.times = [x-np.min(adjustedtime) for x in adjustedtime]
        self.imageData = adjustedimagedata
        # self.imageData=np.mean(self.imageData, axis=0)
        

        #background image
        background = rawimageData[5:25]
        pg.image(background[0], title='first background slice')

        background = np.mean(background,axis=0)
        print 'dimensions of background', np.shape(background)
        pg.image(background, title='mean background')
        #subtract background from image files

        print 'dimensions of imagedata', np.shape(self.imageData)
        subtracted = np.zeros(np.shape(self.imageData), float)
        divided = np.zeros(np.shape(self.imageData), float)
        for i in range(self.imageData.shape[0]):
            subtracted[i,:,:] = (self.imageData[i,:,:]-background)
            divided[i,:,:] = self.imageData[i,:,:]/background
            #subtracted = self.imageData-background
        subtracted=subtracted/subtracted.mean()
        divided=divided/divided.mean()
        print 'dimensions of subtracted', np.shape(subtracted)
        print 'dimensions of divided', np.shape(divided)
        subtracted = np.mean(subtracted, axis=0)
        divided = np.mean(divided,axis=0)
        pg.image(subtracted, title='subtracted')
        pg.image(divided,title='divided')

        self.imageData = np.mean(self.imageData, axis=0)
        print 'dimensions of imagedata', np.shape(self.imageData)
        pg.image(self.imageData,title='mean raw image')
        edges=feature.canny(self.imageData, sigma=3)
        pg.image(edges,title='edges')
        # for background file
#         im2=[]
#         self.imageData2 = []
#         adjustedimagedata = []

#         print "loading data from ", videodwnf
#         try:
#             im2 = MetaArray(file = videodwnf,  subset=(slice(0,2), slice(64,128), slice(64,128)))
#         except:
#             print "Error loading upfile: %s\n" % videodwnf
#             return
#         print "data loaded"
        
         
#         rawtimes=[]
#         rawimageData=[]
#         rawtimes = im2.axisValues('Time').astype('float32')

#         rawimageData = im2.view(np.ndarray).astype('float32')
# #  

#         #reads the timestamps from the files
#         indexFile = configfile.readConfigFile(basepath+'.index') 
#         timestampdwn = indexFile.__getitem__('video_'+DB[options.fdict][1]+'.ma')[u'__timestamp__']
#         audiodwnindex = configfile.readConfigFile(audiobasepath+DB[options.fdict][1]+'/.index')
#         audiodwnstamp = audiodwnindex.__getitem__(u'.')[u'__timestamp__'] 
 
       
#         diffdwn = audiodwnstamp - timestampdwn

        
        
#         audio = MetaArray(file = audiodwnf, subset=(slice(0,2), slice(64,128), slice(64,128)))
#         audiotime = audio.axisValues('Time').astype('float32')
#         audiomin = np.min(audiotime) + diffdwn
#         audiomax = np.max(audiotime) + diffdwn
        
#         print 'audiomin', audiomin
#         print 'audiomax', audiomax

#         adjustedtime = rawtimes[np.logical_and(rawtimes <= audiomax+.5, rawtimes >= audiomin)]
#         frame_start=np.amin(np.where(rawtimes >= audiomin))
#         frame_end=np.amax(np.where(rawtimes <= audiomax+0.5))
#         adjustedimagedata = rawimageData[frame_start:frame_end]
 
#         self.times = [x-np.min(adjustedtime) for x in adjustedtime]
#         self.imageData2 = adjustedimagedata
#         self.imageData2=np.mean(self.imageData2, axis=0)
#         print 'size of imagedata', self.imageData2.shape
#         diffframes = self.imageData/self.imageData2
#         print 'mean:', diffframes.mean()
#         print 'std:', diffframes.std()
        # procimage=diffframes/diffframes.mean()/diffframes.std()

        # self.avgframes = pg.image(diffframes, title='Average across frames')
        # self.diff_frames = pg.image(procimage, title='Normalized Average across frames')
        # self.imagebck=pg.image(rawimageData[10])
        #self.avgframes = pg.image(procimage, title='Average across frames')
        return

    def subtract_Background(self, diffup=0.005):
        #loading background data
        print 'running subtractBackground'
        bckfile = videobasepath + '005.ma'
        bckaudio = audiobasepath + '005/DaqDevice.ma'
        # bckfile = videobasepath + '011.ma'
        # bckaudio = audiobasepath + '011/DaqDevice.ma'
        
        try:
            im = MetaArray(file = bckfile,  subset=(slice(0,2), slice(64,128), slice(64,128)))
        except:
            print 'no background file!'
        #correct for timing differences
        audio = []
        audio = MetaArray(file = bckaudio, subset=(slice(0,2), slice(64,128), slice(64,128)))
        audiotime = audio.axisValues('Time').astype('float32')

        audiomin = np.min(audiotime) + diffup
       
        audiomax = np.max(audiotime) + diffup
        rawtimes = im.axisValues('Time').astype('float32')
        adjustedtime = rawtimes[np.logical_and(rawtimes <= audiomax+.5, rawtimes >= audiomin)]
        bckimagedata = im[np.logical_and(rawtimes <= audiomax, rawtimes >= audiomin)]
        raw=self.imageData
        #check that the background image and the data are the same shape and average
        #then subtract the average from the stimulated data
        getout = fac_lift(bckimagedata,adjustedtime)
        bckimagedata=getout
        getout2 = fac_lift(raw, adjustedtime)
        self.imageData=getout2
        print 'shape of background', bckimagedata.shape
        print 'shape of imageData', self.imageData.shape
        bckimagedata=np.mean(bckimagedata,axis=0)
        # if bckimagedata.shape[0] <= self.imageData.shape[0]:
        #     print 'image is longer'
        #     stop = bckimagedata.shape[0]
        #     print 'stop'
        #     self.imageData=self.imageData[: stop,:,:,:]
        #     print 'stop2'
        #     subtractor = np.zeros(bckimagedata.shape, float)
        #     diffimage = np.zeros(bckimagedata.shape, float)
        #     subtractor = np.mean(np.array([self.imageData,bckimagedata]), axis=0)
        #     #diffimage=sub_func(self.imageData, subtractor)
        #     diffimage = self.imageData - subtractor
        #     print 'stop 3'     
        # else:
        #     print 'error! image is shorter, fix this code!'
        diffimage = scipy.signal.detrend(diffimage, axis=0)    
        self.imageData = diffimage    
        return

    
    def Analysis_FourierMap_TFR(self, period = 4.25, target = 1, mode=0, bins = 1, up=1):
        global D
        D = []
        self.DF = []
        self.avgimg = []
        self.stdimg = []
        self.nFrames =self.imageData.shape[0]
        self.imagePeriod = 0
        # if HAVE_MPL:
        #     pylab.figure(2)
        #self.imageData = self.imageData.squeeze()
 
#         print "now calculating mean"
#         # excluding bad trials
#         trials = range(0, n_Periods)
#         reject = reject[0]
#         # print 'trials', trials
#         # print 'reject', reject
#         # print 'nptpercycle', n_PtsPerCycle

#         #N.B.- Commenting this out seems to resolve issues.  Figure out why!
#         # for i in range(0,len(reject)):
#         #     t = reject[i]/n_PtsPerCycle
#         #     if t in trials:
#         #         trials.remove(t)

#         print "retaining trials: ", trials
#         D = numpy.mean(self.imageData[trials,:,:,:], axis=0).astype('float32') # /divider # get mean of the folded axes.
#         print "mean calculated, now detrend and fft"
#         # detrend before taking fft
#         D = scipy.signal.detrend(D, axis=0)
#         # calculate FFT and get amplitude and phase
#         self.DF = numpy.fft.fft(D, axis = 0)
        #self.reshapeImage()
        D=numpy.mean(self.imageData, axis = 0)
        print 'shape of D', D.shape
        self.DF = numpy.fft.fft(D, axis = 0)
        ampimg = numpy.abs(self.DF[1,:,:]).astype('float32')
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
        if target == 2:
            f = open('img_phase2.dat', 'w')
            pickle.dump(phaseimg, f)
            f.close()
            f = open('img_amplitude2.dat', 'w')
            pickle.dump(ampimg, f)
            f.close()
            self.amplitudeImage2 = ampimg
            self.phaseImage2 = phaseimg
        print "fft calculated, data  saveddata"
        # save most recent calculation to disk

    def sub_func(self, a, avg):
        return(a - avg)

    def plotmaps(self, mode = 0, target = 1, gfilter = 0):
        global D
        max1 = numpy.amax(self.amplitudeImage1)
        if target > 1:
            max1 = numpy.amax([max1, numpy.amax(self.amplitudeImage2)])
        max1 = 10.0*int(max1/10.0)
        pylab.figure(1)
        pylab.subplot(2,3,1)
        pylab.title('Amplitude Map 1')
        #scipy.ndimage.gaussian_filter(self.amplitudeImage1, 2, order=0, output=self.amplitudeImage1, mode='reflect')
        ampimg = scipy.ndimage.gaussian_filter(self.amplitudeImage1,gfilt, order=0, mode='reflect')
        print 'ampimg:', ampimg
        imga1 = pylab.imshow(ampimg)
        pylab.colorbar()
        imga1.set_clim = (0.0, max1)
        pylab.subplot(2,3,4)
        pylab.title('Phase Map 1')
        imgp1 = pylab.imshow(scipy.ndimage.gaussian_filter(self.phaseImage1, gfilt, order=0,mode='reflect'), cmap=matplotlib.cm.hsv)
        imgp1.set_clim=(-numpy.pi/2.0, numpy.pi/2.0)
        pylab.colorbar()

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
            dphase = np1 + np2
            #dphase = self.phaseImage1 - self.phaseImage2
           
            #scipy.ndimage.gaussian_filter(dphase, 2, order=0, output=dphase, mode='reflect')
            imgpdouble = pylab.imshow(dphase, cmap=matplotlib.cm.hsv)
            pylab.title('2x Phi map')
            pylab.colorbar()
            imgpdouble.set_clim=(-numpy.pi, numpy.pi)


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

        if mode == 2:
            pylab.subplot(2,3,6)
            for i in range(0, self.nPhases):
                pylab.plot(self.DF[1:,80, 80])
                pylab.hold('on')
            pylab.title('FFTs')

        pylab.show()

# plot data
    def plotmaps_pg(self, mode = 0, target = 1, gfilter = 0):

        pos = np.array([0.0, 0.33, 0.67, 1.0])
        color = np.array([[0,0,0,255], [255,128,0,255], [255,255,0,255],[0,0,0,255]], dtype=np.ubyte)
        maps = pg.ColorMap(pos, color)
        lut = maps.getLookupTable(0.0, 1.0, 256)
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
        if mode == 0 or mode == 2:
            self.fftPlt = pg.plot(title = 'FFTs')
        
        if mode == 0:
            #pylab.subplot(2,3,3)

            # for i in range(0, self.nPhases):
            #     self.wavePlt.plot(ta.n_times, D[:,5,5].view(ndarray))
            #     #pylab.plot(ta.n_times, D[:,5,5].view(ndarray))
            #     #pylab.plot(self.n_times, D[:,i*55+20, 60])
            #     #pylab.hold('on')
            # #pylab.title('Waveforms')

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
            dphase = np1 + np2
            print 'shape of dphase', dphase.shape
            #dphase = self.phaseImage1 - self.phaseImage2
            print 'min phase', np.amin(dphase)
            print 'max phase', np.amax(dphase)
            for i in range(dphase.shape[0]):
                for j in range(dphase.shape[1]):
                    #for k in range(dphase.shape[2]):
                    if dphase[i,j]<0:
                        dphase[i,j] = dphase[i,j]+2*np.pi

            print 'min phase', np.amin(dphase)
            print 'max phase', np.amax(dphase)
            
            self.win = pg.GraphicsWindow()
            view = self.win.addViewBox()
            view.setAspectLocked(True)
            item = pg.ImageItem(dphase)
            view.addItem(item)
            item.setLookupTable(lut)
            item.setLevels([0,2*np.pi])

            # self.colorlevels = pg.GradientEditorItem()
            # self.colorlevels.getLookupTable(17)
            # self.colorlevels.setColorMode('rgb')
            # self.colorlevels.setOrientation('right')
            # self.colorlevels.setPos(-10,0)
            # view.addItem(self.colorlevels)
            gradlegend = pg.GradientLegend((10,100),(0,0))
            #gradlegend.setIntColorScale(0,255)
            #gradlegend.setGradient(self.creategradient())
            gradlegend.setGradient(maps.getGradient())
            view.addItem(gradlegend)
            #self.phiView.addItem(pg.ImageItem(dphase))
            self.phi = pg.image(dphase, title="2x Phi map", levels=(0, 2*np.pi))
            
            #imgpdouble = pylab.imshow(dphase, cmap=matplotlib.cm.hsv)
            #pylab.title('2x Phi map')
            #pylab.colorbar()
            #imgpdouble.set_clim=(-numpy.pi, numpy.pi)

        print "plotmaps Block 3"

        # if mode == 2 or mode == 1:
        #     # if self.phasex == []:
        #     #     self.phasex = numpy.random.randint(0, high=D.shape[1], size=D.shape[1])
        #     #     self.phasey = numpy.random.randint(0, high=D.shape[2], size=D.shape[2])

        #     #pylab.subplot(2,3,3)
        #     sh = D.shape
        #     spr = sh[2]/self.nPhases
        #     wvfms=[]
        #     for i in range(0, self.nPhases):
        #         Dm = self.avgimg[i*spr,i*spr] # diagonal run
        #         wvfms=self.n_times, 100.0*(D[:,self.phasex[i], self.phasey[i]]/Dm)
        #         #pylab.plot(self.n_times, 100.0*(D[:,self.phasex[i], self.phasey[i]]/Dm))
        #         self.wavePlt.plot(self.n_times, 100.0*(D[:,self.phasex[i], self.phasey[i]]/Dm))
        #         #pylab.hold('on')
        #         self.plotlist.append(pg.image(wvfms, title="Waveforms"))
        #         print "it worked"
        #     pylab.title('Waveforms')

        # print "plotmaps Block 4"

        if mode == 2:
            #pylab.subplot(2,3,6)
            for i in range(0, self.nPhases):
                #pylab.plot(self.DF[1:,80, 80])
                spectrum = np.abs(self.DF)**2
                self.fftPlt.plot(spectrum[1:,80,80])
                #pyqtgraph.intColor(index, hues=17, values=1, maxValue=255, minValue=150, maxHue=360, minHue=0, sat=255, alpha=255, **kargs)
                
                #self.fftPlt.plot(self.DF[1:,80,80]) ## causing errors and i'm not sure what the desired thing is, Exception: Can not plot complex data types.
                #pass
                #pylab.hold('on')
                # self.plotlist.append(pg.image(wvfms, title="Waveforms"))
                print "waveform plotting worked"
            # pylab.title('Waveforms')

        print "plotmaps Block 4"

        # if mode == 2:
        #     #pylab.subplot(2,3,6)
        #     for i in range(0, self.nPhases):
        #         #pylab.plot(self.DF[1:,80, 80])
        #         spectrum = np.abs(self.DF)**2
        #         self.fftPlt.plot(spectrum[1:,80,80])
        #         #pyqtgraph.intColor(index, hues=17, values=1, maxValue=255, minValue=150, maxHue=360, minHue=0, sat=255, alpha=255, **kargs)
                
        #         #self.fftPlt.plot(self.DF[1:,80,80]) ## causing errors and i'm not sure what the desired thing is, Exception: Can not plot complex data types.
        #         #pass
        #         #pylab.hold('on')
        #     #pylab.title('FFTs')
        
        # print "plotmaps Block 5"
        # print "plotting complete"
        return
        #pylab.show()
        #self.view.show()

    def creategradient(self):
        g = QtGui.QLinearGradient()
        for i in range(17):
            g.setColorAt(float(i)/17.0, pg.intColor(i, hues=17))

        return g

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

def fac_lift(goingin, times, period=4.25,):
        period = 4.25
        print "reshape Starting"

        maxt = times[-1] # find last image time
        print "Duration of Image Stack: %9.3f s (%8.3f min)\n" % (maxt, maxt/60.0)
        dt = numpy.mean(numpy.diff(times)) # get the mean dt
        sh = np.shape(goingin)
    # #determine the number of periods in the timeseries of the data
        period# image period in seconds.

        n_Periods = int(numpy.floor(maxt/period)) # how many full periods in the image set?

        n_PtsPerCycle = int(numpy.floor(period/dt)); # estimate image points in a stimulus cycle
        ndt = period/n_PtsPerCycle

        goingin = goingin[range(0, n_Periods*n_PtsPerCycle),:,:] # reduce to only what we need
        times = numpy.arange(0,goingin.shape[0]*dt, dt)# reduce data in blocks by averaging

        goingin = numpy.reshape(goingin,(n_Periods, n_PtsPerCycle, sh[1], sh[2])).astype('float32')
        print 'shape of rescaled imagedata', goingin.shape
        
        return goingin


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