import sys, os
import numpy
import numpy as np
import scipy.signal
import scipy.ndimage 
import pyqtgraph as pg #added to deal with plottng issues TFR 11/13/15
import pickle
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as mpl
import pylab
from PyQt5 import QtGui
from optparse import OptionParser
import tifffile as tf


app = pg.Qt.QtGui.QApplication([])

D = []
d = []
binsize = 4

basepath = 'micromanager'
homedir = '/Volumes/TROPPDATA/data/'
basepath = os.path.join(homedir, basepath)
fn = ''

class testAnalysis():
    def __init__(self):
        global d
        global measuredPeriod
        global gfilt
        global binsize
        self.times = []
        self.binsize=binsize
        self.upfile = []
        self.nrepetitions=5
        self.downfile = []
        self.period = 10.0
        self.avgimg = []
        self.imageData = []
        self.subtracted = []
        self.divided = []
        self.phasex = []
        self.phasey = []
        self.nPhases = 1
        self.nCycles = 1
        self.zbinsize = 4
        self.gfilter =0
        self.framerate = 10
        return


    def parse_commands(self, argsin=None):
        parser=OptionParser() # command line options
        parser.add_option("-D", "--directory", dest="directory", metavar='FILE',help="Use directory for data")
        parser.add_option("-T", "--tiff", dest="tifffile", default=fn, type="str",help="load a tiff file")
        
        if argsin is not None:
            (options, args) = parser.parse_args(argsin)
        else:
            (options, args) = parser.parse_args()

    
        if options.tifffile is not None:
            self.tifffile = options.tifffile
            n2 = self.tifffile + '_MMStack_Pos0.ome.tif'
            self.read_tiff_stack(filename=os.path.join(basepath, self.tifffile, n2))
            im=self.imageData
            pg.image(im,title='raw imageData')
            pg.image(np.mean(im,axis=0), title='mean')
            self.zBinning()
            return

        return

    def read_tiff_stack(self, filename=None):
        """
        Read a diff stack into the data space.
        Using preset parameters, populates the frames, repetitions and frame times
        """
        print filename
        if filename is None:
            raise ValueError('No file specified')
        print 'Reading tifffile: %s' % filename
        self.imageData = tf.imread(filename)
        self.nFrames = self.imageData.shape[0]
        print 'image shape: ', self.imageData.shape[:]
        self.nrepetitions = np.floor(self.nFrames/(self.period * self.framerate))
        print 'n_reps:', self.nrepetitions
        print 'period:', self.period
        print 'framerate', self.framerate
        print 'nFrames:', self.nFrames

        return

    def Image_Background(self):
        self.background=[]
        self.background = np.mean(self.imageData,axis=0)
        return

    def Image_Divided(self):
        self.divided = (self.imageData-self.background)/self.background
        self.imageData = self.divided
        pg.image(self.divided[1:],title='divide image')
        return

    def zBinning(self):
        
        self.imageData = self.imageData[20:,:,:]
        sh = np.shape(self.imageData)
        numbins = np.floor(sh[0]/4)
        if sh[0]>numbins*4:
            self.imageData=self.imageData[:numbins*4,:,:]
        bins = np.zeros([4,numbins,sh[1],sh[2]],float)
        bins = np.reshape(self.imageData,[4,numbins,sh[1],sh[2]])
        bins = np.mean(bin,axis=0)
        pg.image(bins, title='binned in z by 4')
        self.imageData = bins
        return


    def avg_over_trials(self):

        self.shaped = []
        single = int(self.period*self.framerate)
        print 'single: ',single
        self.shaped = np.reshape(self.imageData,[self.nrepetitions,single,self.imageData.shape[1],self.imageData.shape[2]])
        self.imageData = np.mean(self.shaped[1:],axis=0)
        return

if __name__ == "__main__":
    ta=testAnalysis()  # create instance (for debugging)
    ta.parse_commands(sys.argv[1:])
    app.exec_()
    mpl.show()