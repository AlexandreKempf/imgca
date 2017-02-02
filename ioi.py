import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.ndimage import zoom, sobel
import scipy.ndimage.filters as filt
from PIL import Image, ImageFilter
from skimage import feature
import roipoly
from imgca import exist, smooth, binArray

def inputAxis(axis):
    if isinstance(axis, str):
        return ["stim","rep","px","py","time"].index(axis)
    else:
        return int(axis)

def importRaw(dirpath, stim=[]):
    dirpath= "/home/alexandre/docs/code/dev/pkg_lab/ioi/M07/20160106"
    vessels = np.array(Image.open(exist(dirpath,"*.tif"))).astype(float)
    # change scale to 0-255 in uint8
    vessels -= np.min(vessels)
    vessels *= 255/np.max(vessels)
    vessels = vessels.astype(float).transpose()

    name = os.path.basename(dirpath)
    if len(stim)==0:
        stim = np.array(["Whitenoise","4kHz", "8kHz", "16kHz", "32kHz"])
    else:
        stim = np.array(stim)
    datafile = exist(dirpath, """*data.mat""")
    stimfile = exist(dirpath, """*.stim.mat""")
    data = loadmat(datafile, mat_dtype=True)
    conds = loadmat(stimfile, mat_dtype=True)

    idx = np.array(np.ravel(conds["list"]["idx"])[0][0]-1)
    data = np.array([data["frame"][:, :, :, idx==sound] for sound in np.arange(len(stim))]).astype(float)
    data = data.transpose((0,4,1,2,3)) # transform to (stim, rep, px, py, t)
    data = data[:,:,::-1,:,:]
    data = data[:,:,:,::-1,:]
    data -= np.min(data)
    data *= 255/np.max(data)
    px = int(data.shape[2]*4)
    py = int(data.shape[3]*4)
    vessels = vessels[:px,:py]
    return data, stim, vessels

def dRoverR(data):
    baseline = np.median(data[:,:,:,:,:40],4,keepdims=True)
    data -= baseline
    data /= baseline
    return data

def focusROI(data, vessels):
    plt.imshow(vessels, origin="lower")
    roi = roipoly.roipoly(roicolor='r')
    mask = roi.getMask(vessels).astype(float)
    mask[mask==False] *= np.nan
    vessels *= mask
    maskdata = binArray(mask,1,4,4,np.mean)
    maskdata = binArray(maskdata,0,4,4,np.mean)
    data *= maskdata.reshape((1,1,maskdata.shape[0],maskdata.shape[1],1))
    return data, vessels

def estimateBloodVessels(vessels, sigma=2):
    return feature.canny(vessels, sigma=sigma)

def estimateActivity(data,vessels,stims):
    if isinstance(stims, int):
        stims = [stims]
    img = np.zeros((len(stims), data.shape[2], data.shape[3]))
    for i in np.arange(len(stims)):
        img[i] = -data[stims[i],:,:,:,70:85].mean(3).mean(0)
    return img

def plotActivity(data, stim, vessels, stimsel, sigma=2, cmap = plt.cm.RdBu_r):
    imgs = estimateActivity(data,vessels,stimsel)
    canny = estimateBloodVessels(vessels, sigma)
    cmap.set_bad('black',1.)
    for i in np.arange(imgs.shape[0]):
        img = zoom(imgs[i],(4,4), order=0)
        img[canny] = np.nan
        plt.figure()
        plt.title(stim[stimsel[i]])
        plt.imshow(img, origin="lower", interpolation="none", cmap = cmap)
        plt.clim(np.nanpercentile(img,10),np.nanpercentile(img,99));
    plt.show()

def estimateTonotopy(data, vessels, stimsel, weights, smoothfactor = 5, sigma=2):
    imgs = estimateActivity(data, vessels, stimsel)
    weights = np.reshape(weights, (len(weights), 1, 1))
    return np.sum(imgs*weights,0)


def plotTonotopy(data, vessels, stimsel, weights, sigma=2, cmap = plt.cm.RdBu_r):
    img = estimateTonotopy(data,vessels,stimsel,weights)
    canny = estimateBloodVessels(vessels, sigma)
    cmap.set_bad('black',1.)
    img = zoom(img,(4,4), order=0)
    img[canny] = np.nan
    plt.imshow(img, origin="lower", interpolation="none", cmap = cmap)
    plt.clim(np.nanpercentile(img,1),np.nanpercentile(img,99));
    plt.show()


dirpath = "/run/user/1001/gvfs/smb-share:server=157.136.60.15,share=eqbrice/IntrinsicImaging/thibault/cage6mouse2/"
data, stim, vessels = importRaw(dirpath, stim=["Whitenoise","4kHz", "8kHz", "16kHz", "32kHz"])
data = smooth(data,2,inputAxis("time"))
data = smooth(data,2,inputAxis("px"))
data = smooth(data,2,inputAxis("py"))

data = dRoverR(data)
data, vessels = focusROI(data, vessels)

plotActivity(data, stim, vessels,[1,2,3,4], sigma=2, cmap = plt.cm.jet)
plotTonotopy(data, vessels, stimsel= [1,2,3,4], weights=[-6,-2,2,6], sigma=2, cmap = plt.cm.jet)
