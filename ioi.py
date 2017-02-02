# import cv2
import numpy as np
from scipy.io import loadmat
# import glob
import scipy.ndimage.filters as filt
from imgca import exist, smooth, binArray
#
#
import roipoly
from scipy.ndimage import zoom, sobel
# from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d, maximum_filter
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import os
# from copy import deepcopy
# from skimage.measure import block_reduce
from skimage import feature
# import scipy.ndimage as ndimage

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

def estimateBloodVessels(vessels, smoothfactor = 5, sigma=2):
    vess = smooth(vessels,smoothfactor,0)
    vess = smooth(vess,smoothfactor,1)
    cannyvessels=feature.canny(vess, sigma=sigma)
    return cannyvessels

def estimateActivity(data,vessels,stims):
    if isinstance(stims, int):
        stims = [stims]
    img = np.zeros((len(stims), data.shape[2], data.shape[3]))
    for i in np.arange(len(stims)):
        img[i] = -data[stims[i],:,:,:,70:85].mean(3).mean(0)
    return img

def plotActivity(data, stim, vessels, stimsel, smoothfactor = 5, sigma=2):
    imgs = estimateActivity(data,vessels,stimsel)
    canny = estimateBloodVessels(vessels, smoothfactor, sigma)
    cmap = plt.cm.RdBu_r
    cmap.set_bad('black',1.)
    for i in np.arange(imgs.shape[0]):
        img = zoom(imgs[i],(4,4), order=0)
        img[canny] = np.nan
        plt.figure()
        plt.title(stim[stimsel[i]])
        plt.imshow(img, origin="lower", cmap = cmap)
        plt.clim(np.nanpercentile(img,10),np.nanpercentile(img,99));
    plt.show()



dirpath = "/run/user/1001/gvfs/smb-share:server=157.136.60.15,share=eqbrice/IntrinsicImaging/thibault/cage6mouse2/"
data, stim, vessels = importRaw(dirpath, stim=["Whitenoise","4kHz", "8kHz", "16kHz", "32kHz"])
# plt.imshow(vessels, origin="lower", interpolation='none');plt.show()
data = smooth(data,2,4)
data = smooth(data,1,2)
data = smooth(data,1,3)

data = dRoverR(data)
# data, vessels = focusROI(data, vessels)
# imgs = estimateActivity(data,vessels,[0,1,2,3])


plotActivity(data, stim,vessels, [0,1,2], smoothfactor = 5, sigma=2)




















1+1
    def return_image(self, sound, sigma=0.1, plot=False):
        vessels=feature.canny(self.vessels, sigma=sigma)
        img = zoom(-self.data[sound,:,:,70:85,:].mean(3).mean(2),4,order=0)
        img[vessels] = np.nan
        # img = np.ma.array(img, mask=np.isnan(img))
        cmap = plt.cm.RdBu_r
        cmap.set_bad('black',1.)
        if plot:
            plt.imshow(img, cmap = cmap)
            plt.clim(np.nanpercentile(img,15),np.nanpercentile(img,98));
        return img

    def return_tonotopy(self, sigma=0.8, plot=False):
        imgs = -self.data[:,:,:,70:85,:].mean(4).mean(3)

        imgs -= np.nanmin(np.nanmin(imgs,2,keepdims=True),1,keepdims=True)
        imgs *= 1./np.nanmax(np.nanmax(imgs,2,keepdims=True),1,keepdims=True)
        imgs[imgs<0.5] = np.nanmedian(imgs)

        vessels=feature.canny(self.vessels, sigma=sigma)
        img = 6*imgs[4]+imgs[3]-imgs[2]-6*imgs[1]
        img = zoom(img,4,order=0)
        img = gaussian_filter(img,(2,2))
        img[vessels] = np.nan
        if plot:
            plt.imshow(img)
        return img



#### FAIRE LE ALIGN_IOI ICI AUSSI


# dirpath = "/home/alexandre/docs/code/dev/pkg_lab/ioi/seb"
dirpath = "/run/user/1001/gvfs/smb-share:server=157.136.60.15,share=eqbrice/IntrinsicImaging/thibault/cage6mouse2/"
a = ioio(dirpath)
a.dRoR()
b = a.copy();
b.smooth((0,3,3,2,0));
# b.focusROI(30)

for i in np.arange(5):
    if i==0:
        ax = plt.subplot(2,3,i+1);
    else:
        plt.subplot(2,3,i+1, sharex=ax,sharey=ax);
    plt.title(b.stim[i])
    b.return_image(i,sigma = 0.2,plot=True);
plt.show()

img = b.return_tonotopy(sigma = 0.2, plot=True)
plt.show()















def loadioi(paths, show=False,thr_down=10,thr_up=40):
    """ Path is a vector of multiple paths"""
    resultmatall = []
    for path in paths:
        vessels = cv2.imread(exist(path,"*.tif"), 0)
        vessels = cv2.GaussianBlur(vessels, (5,5), 2)
        vessels = vessels[1:-1,1:-2]
        ioipath = exist(path, """*data.mat""")
        ioipathstim = exist(path, """*.stim.mat""")
        name = path.split("/")[-1]
        title = [name+"_Whitenoise", name+"_4kHz", name+"_8kHz", name+"_16kHz", name+"_32kHz"]
        resultmat = []
        resultmat.append(vessels)
        for indexsound in np.arange(5):
            ## Data extraction
            ioimat = loadmat(ioipath, mat_dtype=True)
            ioimatstim = loadmat(ioipathstim, mat_dtype=True)
            idx = np.array(np.ravel(ioimatstim["list"]["idx"])[0][0]-1)
            ioimat = ioimat["frame"][:, :, :, idx==indexsound]

            ## Data preprocess
            if ioimat.shape[2]==2:
                ioiimg = np.mean((ioimat[:,:,0,:]/ioimat[:,:,1,:])-1, 2) # Old technique
            else:
                bef = np.mean(ioimat[:,:,:60,:],2)
                after = np.mean(ioimat[:,:,60:,:],2)
                ioiimg = np.mean((bef/after)-1, 2)
            ioiimg = zoom(ioiimg, 4, order=0)
            ioiimg = gaussian_filter(ioiimg, 2*4)
            ioiimg = maximum_filter(ioiimg, 3*4)
            ioiimg *= 255/np.max(ioiimg)

            # Detect Blood vessels
            ves = cv2.Canny(vessels,thr_down,thr_up)

            # Detect signal contours
            imgplot = np.zeros_like(ioiimg)
            for i in np.arange(0,255,15):
                imgplot[ioiimg >= i] = i

            # Add the two images
            imgplot[ves!=0] = 0
            imgplot = np.array(imgplot,np.uint8)
            imgplot = cv2.applyColorMap(imgplot, cv2.COLORMAP_JET)
            resultmat.append(imgplot)
            # Show
            if show==True:
                cv2.imshow(title[indexsound], imgplot)
            print(path+'/'+title[indexsound]+".png")
            cv2.imwrite(path+'/'+title[indexsound]+".png", imgplot)

        resultmatall.append(resultmat)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return resultmatall



# exemple :
a=['/run/user/1001/gvfs/smb-share:server=157.136.60.15,share=eqbrice/IntrinsicImaging/Alexandre/161202_C01M01']
loadioi(a, show=False,thr_down=7,thr_up=30)
