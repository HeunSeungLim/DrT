import numpy as np
import cv2
import h5py



f = h5py.File('./test/002.h5', 'r')
data = f['IN'][:]
label = f['GT'][:]
f.close()

cv2.imwrite('data')


