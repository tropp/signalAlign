"""
This is an image conversion file that orients images from imageJ, pyqtgraph and matplotlib into the same orientation so they can be easily overlayed
This will also convert files from the AMScope camera into a fixed orientation.
Our desired orientation is ROSTRAL pointing left and LATERAL pointing down.
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

basepath = 'micromanager'
#homedir = '/media/ropphouse/TROPPDATA/data'
#homedir = '/Volumes/TROPPDATA/data'
homedir = '/Users/tjropp/Desktop/data'
basepath = os.path.join(homedir, basepath)

class ImageCorrection():
    """
    Provide routines for analyzing image stacks by extracting phase and amplitude 
    across the image plane from the time data series
    """
    def __init__(self, layout, winsize, d=[], measurePeriod=8.0, binsize=1, gfilter=0):

        self.layout=layout
        self.winsize = winsize
        self.d = d
        self.times = []
        self.imageData =[]
        self.framerate = []
        self.period = []
        self. nrepetitions = 1


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
        parser.add_option("-T", "--tiff", dest="tifffile", default=None, type="str",
                          help="load a tiff file")

        if argsin is not None:
            (options, args) = parser.parse_args(argsin)
        else:
            (options, args) = parser.parse_args()


        if options.tifffile is not None:
            self.tifffile = options.tifffile

        
        
        if options.tifffile is not None:
            n2 = self.tifffile + '_MMStack_Pos0.ome.tif'
            self.read_tiff_stack(filename=os.path.join(basepath, self.tifffile, n2))
            return

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

        self.imageData = tf.imread(filename)


        if len(np.shape(self.imageData)) > 2:
            self.avgimg = np.mean(self.imageData, axis=0) # get mean image for reference later: average across all time
        else:
            self.avgimg = self.imageData

        imagepg = ndimage.rotate(self.avgimg, 180, reshape = 'False')
        imagempl = ndimage.rotate(self.avgimg,-90, reshape = 'False')
        pg.image(imagepg)
        mpl.imshow(imagempl,origin = 'lower')


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
            #            print self.frequencies
        else:
            print ('*** No matfile found ***\n')


        

if __name__ == "__main__":

    layout = None
    winsize = []
    ta = ImageCorrection(layout=layout, winsize=winsize)  # create instance (for debugging)
    ta.parse_commands(sys.argv[1:])
    mpl.show()
    app.exec_()