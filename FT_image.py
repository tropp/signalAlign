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
from astropy.convolution import convolve_fft, convolve, Box2DKernel, Box1DKernel
#from astropy import image
import pickle
import matplotlib
import matplotlib.mlab as mlab
import pylab
from PyQt4 import QtGui
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
DB = {17: ('017', '016', 610, 50.0, 4.25, '08Jan16', fl, 'thinned skull')}
#
DB[18] = ('018','015', 610, 10.0, 4.25, '05Feb16', fl, 'thinned skull')
DB[19] = ('019','014', 610, 30.0, 4.25, '05Feb16', fl, 'thinned skull')


homedir = os.getenv('HOME')
videobasepath = '/Volumes/TRoppData/data/Intrinsic/2016.02.05_000/slice_002/video_'
basepath = '/Volumes/TRoppData/data/Intrinsic/2016.02.05_000/slice_002/'
audiobasepath = '/Volumes/TRoppData/data/Intrinsic/2016.02.05_000/slice_002/Sound_Stimulation_video_'

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

        if options.period is not None:
            measuredPeriod = options.period
        if options.cycles is not None:
            self.nCycles = options.cycles
        if options.binsize is not None:
            binsize = options.binsize
        if options.gfilt is not None:
            gfilt = options.gfilt

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
        videoupf = None
        videodwnf = None
        audioupf = None
        audiodwnf = None

        if options.upfile is not None:
            videoupf = videobasepath + options.upfile + '.ma'
            audioupf = audiobasepath + options.upfile + '/DaqDevice.ma'
        if options.downfile is not None:
            videodwnf = videobasepath + options.downfile + '.ma'
            audiodwnf = audiobasepath + options.downfile + '/DaqDevice.ma'

        for file in (videoupf, videodwnf):
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
            self.times = im.axisValues('Time')
            self.nFrames = numpy.shape(im)[0]
            self.imageData = numpy.array(im).astype(numpy.float32, copy=False)
            print 'min diff', np.amin(np.diff(self.times))
            print 'miax diff', np.amax(np.diff(self.times))
            dt = numpy.mean(numpy.diff(self.times))
            print 'dt:',dt
            target = target + 1
            rawtimes=[]
            rawimageData=[]
            rawtimes = im.axisValues('Time').astype('float32')
            rawimageData = im.view(np.ndarray).astype('float32')
            print 'size of im', np.shape(rawimageData)
            indexFile = configfile.readConfigFile(basepath+'.index') 
            timestampup = indexFile.__getitem__('video_'+DB[options.fdict][0]+'.ma')[u'__timestamp__']
            timestampdown = indexFile.__getitem__('video_'+DB[options.fdict][1]+'.ma')[u'__timestamp__']
            audioupindex = configfile.readConfigFile(audiobasepath+DB[options.fdict][0]+'/.index')
            audioupstamp = audioupindex.__getitem__(u'.')[u'__timestamp__'] 
            audiodownindex = configfile.readConfigFile(audiobasepath+DB[options.fdict][1]+'/.index')
            audiodownstamp = audiodownindex.__getitem__(u'.')[u'__timestamp__'] 
           
            diffup = audioupstamp - timestampup
            diffdown = audiodownstamp - timestampdown 

            
            if file is videoupf:
                audio = MetaArray(file = audioupf, subset=(slice(0,2), slice(64,128), slice(64,128)))
                audiotime = audio.axisValues('Time').astype('float32')
                audiomin = np.min(audiotime) + diffup
                audiomax = np.max(audiotime) + diffup
            elif file is videodwnf:
                audio = MetaArray(file = audiodwnf, subset=(slice(0,2), slice(64,128), slice(64,128)))
                audiotime = audio.axisValues('Time').astype('float32')
                audiomin = np.min(audiotime) + diffdown
                audiomax = np.max(audiotime) + diffdown
            else:
                print 'ERROR!  Unable to load audio file'
            print 'audiomin', audiomin
            print 'audiomax', audiomax

            adjustedtime = rawtimes[np.logical_and(rawtimes <= audiomax+.5, rawtimes >= audiomin)]
            frame_start=np.amin(np.where(rawtimes >= audiomin))
            frame_end=np.amax(np.where(rawtimes <= audiomax))

            background = np.mean(rawimageData[10:25,:,:], axis=0)
            print 'size of background', np.shape(background)
            adjustedimagedata = rawimageData[frame_start:frame_end,:,:]
            # print 'adjtime', adjustedtime
            self.times = [x-np.min(adjustedtime) for x in adjustedtime]
            self.imageData = adjustedimagedata
            #print 'self.times:', self.times
            # print 'length of self.times', np.shape(self.times)
            # print 'shape of image data', np.shape(self.imageData)
            print 'size of im', np.shape(self.imageData)
            im=[]
            if file is videoupf:
               upflag = 1
            else:
               upflag = 0
            #print 'target:', target

            framerate=1/dt
            xsize=self.imageData.shape[1]
            ysize=self.imageData.shape[2]
            nframes=self.imageData.shape[0]
            freq=4
            frames=self.imageData.reshape(nframes,xsize*ysize)
            print 'seize of frames', np.shape(frames)
            const=np.arange(dt,nframes/framerate,nframes)
            fourvec=[]
            for ii in range(const.shape[0]):
                fourvec[ii]=numpy.exp(2*np.pi*np.complex(0,1)*const[ii]*freq)
            print 'size of fourvec', np.shape(fourvec)
            data=frames[:,:]*fourvec
            print 'size of data', np.shape(data)
            data=np.mean(data,axis=0)
            data=data.reshape(xsize,ysize)
            
            pg.image(data)
            
            # pg.image(background)
            # diffimage=np.zeros(np.shape(self.imageData))

            # print 'single:', np.shape(self.imageData[4])
            # for i in range(self.imageData.shape[0]):
            #     diffimage[i]=self.imageData[i]/background
            #     #self.imageData[i]/background
            # print 'size of diffimage', np.shape(diffimage)
            # refimage=self.imageData[frame_start]/self.imageData[10]
            # conddiff=np.mean(diffimage,axis=0)-refimage
            # graphme=conddiff/conddiff.mean()/conddiff.std()
            # pg.image(graphme,title='graphme')
            #self.subtract_Background(diffup=diffup)
            #self.Analysis_FourierMap_TFR(period=measuredPeriod, target = target,  bins=binsize, up=upflag)
        print 'target:', target
        # if target > 0:
        #     self.plotmaps_pg(mode = 1, target = target, gfilter = gfilt)

        # return

    def subtract_Background(self, diffup=0.005):
        #loading background data
        print 'running subtractBackground'

        bckfile = videobasepath + '008.ma'
        bckaudio = audiobasepath + '008/DaqDevice.ma'

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
        if bckimagedata.shape[0] <= self.imageData.shape[0]:
            print 'image is longer'
            stop = bckimagedata.shape[0]
            print 'stop'
            self.imageData=self.imageData[: stop,:,:,:]
            print 'stop2'
            subtractor = np.zeros(bckimagedata.shape, float)
            diffimage = np.zeros(bckimagedata.shape, float)
            subtractor = np.mean(np.array([self.imageData,bckimagedata]), axis=0)
            #diffimage=sub_func(self.imageData, subtractor)
            diffimage = self.imageData - subtractor
            print 'stop 3'     
        else:
            print 'error! image is shorter, fix this code!'
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
        dt = numpy.mean(numpy.diff(self.times))
        D=numpy.mean(self.imageData, axis = 0)
        LPF = 0.2/dt
        flpf = float(LPF)
        sf = float(1.0/dt)
        wn = [flpf/(sf/2.0)]
        NPole=8
        filter_b,filter_a=scipy.signal.bessel(
                NPole,
                wn,
                btype = 'low',
                output = 'ba')
        print "boxcar LPF"
        for i in range(0, self.imageData.shape[1]):
            for j in range(0, self.imageData.shape[2]):
               # self.imageData[:,i,j] = self.imageData[:,i,j] - self.timeavg
# OLD: stsci not available anymore
#               box_2D_kernel = astropy.convolve.Box2DKernel(2*n_PtsPerCycle)
#               box_2D_kernel = Box2DKernel(5)
                #box_2D_kernel = Box1DKernel(5)
#               print self.imageData[:,i,j]
#               print len(self.imageData[:,i,j])
#               print box_2D_kernel
                #self.imageData[:,i,j] = self.imageData[:,i,j] - convolve_fft(self.imageData[:,i,j], box_2D_kernel) 
#                self.imageData[:,i,j] = self.imageData[:,i,j] - scipy.stsci.convolve.boxcar(self.imageData[:,i,j], (2*n_PtsPerCycle,)) 
                self.imageData[:,i,j]=scipy.signal.lfilter(filter_b, filter_a, scipy.signal.detrend(self.imageData[:,i,j], axis=0)) # filter the incoming signal
    
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