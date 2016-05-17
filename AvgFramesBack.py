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

# Keys are first file #. Data are file name, number of reps, wavelength, attn, date, frequency, comment
DB = {9: ('009', 4, 610, 15.0, '16May16', 8.0, 'thinned skull')}
DB[7] = ('007', 4, 610, 15.0, '16May16', 16.0, 'thinned skull')
DB[5] = ('005', 4, 610, 15.0, '16May16', 32.0, 'thinned skull')
#DB[1] = ('001','002', 610, 30.0, 4.25, '05Feb16', fl, 'thinned skull')
#DB[9] = ('009','004', 610, 30.0, 4.25, '05Feb16', fl, 'thinned skull')
           
#basepath = '/Volumes/TRoppData/data/Intrinsic_data/2016.02.19_000/slice_000/SingleTone_Stimulation_'
basepath = '/Volumes/TRoppData/data/2016.05.16_000/SineAM_Stim_Camera_'

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
        
    def parse_and_go(self, argsin = None):
        global period
        global binsize
        global options
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
        done_deal=np.zeros((512,512),float)
        if argsin is not None:
            (options, args) = parser.parse_args(argsin)
        else:
            (options, args) = parser.parse_args()
        print 'DB keys', DB.keys()
        if options.fdict is not None:
            if options.fdict in DB.keys(): # populate options 
                options.upfile = DB[options.fdict][0]
                options.reps = DB[options.fdict][1]
                options.freq = DB[options.fdict][5]
            else:
               print "File %d NOT in DBase\n" % options.fdict
               return
        if options.directory is not None:
            self.directory = options.directory
        print 'options.upfile', options.upfile
        if options.reps is not None:
            for nn in range(options.reps):
                self.load_file(nn)
                self.Image_Background()
                self.Image_Divided()
                print 'divided', np.shape(self.divided)
                done_deal = done_deal+self.divided
            self.AvgFrames=done_deal/4
            pg.image(self.AvgFrames,title='Average Frames')
                 
        return

    def load_file(self,repnum):
        global options

        upf = basepath + options.upfile + '/00' + str(repnum) + '/Camera/frames.ma'
        im=[]
        self.imageData = []
        print "loading data from ", upf
        try:
            im = MetaArray(file = upf,  subset=(slice(0,2), slice(64,128), slice(64,128)))
        except:
            print "Error loading upfile: %s\n" % videoupf
            return
        print "data loaded"
 
        self.times = im.axisValues('Time').astype('float32')
        self.imageData = im.view(np.ndarray).astype('float32')
   
        return

    def Image_Background(self):
        self.background=[]
        background = self.imageData[self.times<1]
        pg.image(np.mean(background,axis=0), title='average background ')

        self.background = np.mean(background,axis=0)
        return

    def Image_Divided(self):

        divided = np.zeros(np.shape(self.imageData), float)
        for i in range(self.imageData.shape[0]):
            if self.times[i]>1:
                divided[i,:,:] = (self.imageData[i,:,:]-self.background)/self.background
        self.divided = np.mean(divided[self.times>=1],axis=0)
        #pg.image(subtracted, title='subtracted')
        pg.image(self.divided,title='divided')    
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
            dphase = (np1 + np2)/2
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
            dphase = (np1 + np2)/2
            print 'shape of dphase', dphase.shape
            #dphase = self.phaseImage1 - self.phaseImage2
            print 'min phase', np.amin(dphase)
            print 'max phase', np.amax(dphase)
            # for i in range(dphase.shape[0]):
            #     for j in range(dphase.shape[1]):
            #         #for k in range(dphase.shape[2]):
            #         if dphase[i,j]<0:
            #             dphase[i,j] = dphase[i,j]+2*np.pi

            print 'min phase', np.amin(dphase)
            print 'max phase', np.amax(dphase)
            
            self.win = pg.GraphicsWindow()
            view = self.win.addViewBox()
            view.setAspectLocked(True)
            item = pg.ImageItem(dphase)
            view.addItem(item)
            item.setLookupTable(lut)
            item.setLevels([-np.pi,np.pi])

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
            self.phi = pg.image(dphase, title="2x Phi map", levels=(-np.pi, np.pi))
            
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