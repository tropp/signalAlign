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
        self.period = 1.0
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
        parser.add_option("-t", "--test", dest="test", action='store_true',help="Test mode to check calculations", default=False)
        parser.add_option("-u", "--upfile", dest="upfile", metavar='FILE',help="load the up-file")
        parser.add_option("-d", "--downfile", dest="downfile", metavar='FILE',help="load the down-file")
        parser.add_option("-D", "--directory", dest="directory", metavar='FILE',help="Use directory for data")
        parser.add_option("-T", "--tiff", dest="tifffile", default=fn, type="str",help="load a tiff file")
        parser.add_option("-p", '--period', dest = "period", default=self.period, type="float",help = "Stimulus cycle period")
        parser.add_option("-c", '--cycles', dest = "cycles", default=self.nrepetitions, type="int",help = "# cycles to analyze")
        parser.add_option("-b", '--binning', dest = "binsize", default=self.binsize, type="int", help = "bin reduction x,y")
        parser.add_option("-z", '--zbinning', dest = "zbinsize", default=self.zbinsize, type="int",help = "bin reduction z")
        parser.add_option("-g", '--gfilter', dest = "gfilt", default=self.gfilter, type="float",help = "gaussian filter width")
        parser.add_option("-f", '--fdict', dest = "fdict", default=0, type="int",help = "Use dictionary entry")
        parser.add_option("-s", '--skip', dest = "skip", default=0, type="float", help = "frame skip correction")
 
        if argsin is not None:
            (options, args) = parser.parse_args(argsin)
        else:
            (options, args) = parser.parse_args()

        if options.period is not None:
            self.measuredPeriod = options.period
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
        if options.skip is not None:
            self.skip = options.skip

        if options.test is True:
            self.make_test_data()
            return
        
        if options.tifffile is not None:
            n2 = self.tifffile + '_MMStack_Pos0.ome.tif'
            self.read_tiff_stack(filename=os.path.join(basepath, self.tifffile, n2))
            im=self.imageData
            self.Image_Background()
            self.Image_Divided()
            self.avg_over_trials()
            pg.image(self.imageData)
            return

        if options.fdict is not None:
            if options.fdict in DB.keys(): # populate options 
                options.upfile = DB[options.fdict][0]
                options.downfile = DB[options.fdict][1]
                options.period = DB[options.fdict][4]
            else:
                print "File %d NOT in DBase\n" % options.fdict
                return

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

    def avg_over_trials(self):

        self.shaped = []
        single = int(self.period*self.framerate)
        print 'single: ',single
        self.shaped = np.reshape(self.imageData,[self.nrepetitions,single,self.imageData.shape[1],self.imageData.shape[2]])
        self.imageData = np.mean(self.shaped[1:],axis=0)
        pg.image(self.imageData, title='folded and averaged')
        return

if __name__ == "__main__":
    ta=testAnalysis()  # create instance (for debugging)
    ta.parse_commands(sys.argv[1:])
    app.exec_()
    mpl.show()