import numpy as np
import scipy
import os
import pickle
import matplotlib.pyplot as mpl

basepath = '/Volumes/TROPPDATA/data/micromanager/'
upf = 'camera_updown_20161220_1/'
dwnf = 'camera_updown_20161220_2/'
remnant = '_MMStack_Pos0.ome.tif'
upfile = os.path.join(basepath,upf)
downfile = os.path.join(basepath, dwnf)
print 'upfile: ', upfile
print 'downfile: ',downfile
imup = open('img_phase1.dat', 'r')
imdwn = open('img_phase2.dat', 'r')
# imup = open(os.path.join(upfile,'img_phase1.dat'), 'r')
# imdwn = open(os.path.join(downfile,'img_phase1.dat'), 'r')
up = pickle.load(imup)
print 'size of up: ', np.shape(up)
down = pickle.load(imdwn)
print 'size of down: ', np.shape(down)

delphase = -(up-(2*np.pi-down))/2
delphase2 = (up-down)/2
mpl.figure(1)
mpl.imshow(delphase)
mpl.colorbar()
mpl.figure(2)
mpl.imshow(delphase2)
mpl.colorbar()
mpl.show()

imup.close()
imdwn.close()