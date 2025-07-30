"""
Class and method dev for image handling.

Read images and process for use with VMI class.

Note this is written to replace some base functionality in tmoDataBase.py, which reads hit-based data.

14/07/25 v0.0.1-vmi-img, adding file IO + mods for use with image-based VMI datasets (previously assumed hit-based raw data)

Paul Hockett

https://github.com/phockett/tmo-dev

"""
from pathlib import Path

import numpy as np
import skimage as ski
import skimage.io as io

def getFilesImg(self, ext='png', fileSchema=None):
    """
    Basic getFiles() routine for image-based datasets.

    Currently duplicates functionality of tmoDataBase.getFiles() for compatibility.

    14/07/25
    """
    
    # July 2023 - set for updated scheme, with subdirs per run
    # runs['fileParts'] = {N:list(Path(fileBase, fileSchema.format(N=N)).rglob(f"**/*.{ext}")) for N in runList}   # Reformat existing style to list per run - this gives issues later however.
    print("Importing images from file...")
    self.runs['files'] = {}
    
    for N in self.runs['runList']:
        fileList = list(Path(self.runs['fileBase']).rglob(f'{fileSchema.format(N=N)}.{ext}'))
        self.runs['fileParts'][N] = fileList
        
        if len(fileList) == 1:
            self.runs['files'][N] = fileList[0]
        else:
            # Flatten fileList for use with existing methods.
            # NOTE this is likely problematic - no concat for parts here

            # For HDF5 Can try:
            # (a) link style (https://docs.h5py.org/en/stable/high/group.html?highlight=external#external-links)
            # (b) virtual dataset (https://docs.h5py.org/en/stable/vds.html)
            # (c) convert to numpy and then concat (difficult to patch in current tmo routines).
            
            # FOR IMG case, assume format is t.run, or vice versa, for dict keys
            # self.runs['files'].update({f'{N}.{m+1}':item for m,item in enumerate(self.runs['fileParts'][N])})
            self.runs['files'].update({f'{m+1}.{N}':item for m,item in enumerate(self.runs['fileParts'][N])})


def readImgFiles(self, keyDims = ['run','X','Y'], subtractBG = False):
    """
    Read image files.

    For 'hit' data use base class readFiles().
    """

    # Read from "fileParts", indexed by (t,run)
    for k,fileList in self.runs['fileParts'].items():
        # for fileIn in fileList:
        #     imgIn = io.ImageCollection(fileIn)

        fileStrList = [item.as_posix() for item in fileList]
        imgsIn = io.ImageCollection(fileStrList)

        # Concat and subtract
        # This gives subtracted results PER CYCLE
        # NOTE: raw are uint8, so need to convert to float, either with ski or np.
        # NOTE: may also want to rescale? See https://scikit-image.org/docs/stable/user_guide/data_types.html#rescaling-intensity-values
        if subtractBG:
            # imgCycle = ski.util.img_as_float(imageS.concatenate()) - ski.util.img_as_float(imageB.concatenate())
            print("BG subtraction not yet implemented - need to set multiple file patterns.")
        else:
            imgCycle = ski.util.img_as_float(imgsIn.concatenate())

        # Set to output Xarray datastructure
        # imgT = xr.DataArray(imgCycle, dims=['cycle','X','Y'])
        # Q: stack initially by t or cycle? Cycle seems easier...
        coords = {k:np.arange(1,imgCycle.shape[n]+1) for k,n in enumerate(keyDims)}  # Basically same as XR defaults, but should pull from file names here.
        imgT = xr.DataArray(imgCycle, coords=coords)

        imgT.name = k

        # imgT.attrs['files'] = {'Signal':imageS.files,
        #                     'Background':imageB.files}
    
    # Subselect on cycles if set - UPDATE: now set in getFiles
    # if cycles is not None:
    #     imgT = imgT.sel(cycle=cycles)
    
    # Stack per t to dict
    # imgStackDict[t] = imgT.copy()
    
    # Stack per t to Xarray directly
    if n==0:
        # XR dataset with (cycle,X,Y) array per t
        imgStack = xr.Dataset({t:imgT})
    else:
        imgStack = imgStack.assign({t:imgT})

    self.imgStack = imgStack