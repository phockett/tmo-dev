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
import xarray as xr
import skimage as ski
import skimage.io as io
import scipy

def getFilesImg(self, ext='png', fileSchema=None):
    """
    Basic getFiles() routine for image-based datasets.

    Currently duplicates functionality of tmoDataBase.getFiles() for compatibility.


    31/07/25  v2  Implement dict schema to allow multiple file patterns, e.g. signal and bg.

    14/07/25  v1
    """
    
    # July 2023 - set for updated scheme, with subdirs per run
    # runs['fileParts'] = {N:list(Path(fileBase, fileSchema.format(N=N)).rglob(f"**/*.{ext}")) for N in runList}   # Reformat existing style to list per run - this gives issues later however.
    print("Importing images from file...")
    
    if not isinstance(fileSchema, dict):
        fileSchema = {'signal':fileSchema}
    
    
    self.runs['files'] = {}
    self.runs['fileParts'] = {}
    
    for k, v in fileSchema.items():
        print(f"Checking schema {k}:{v}.")
        for N in self.runs['runList']:
            
            if not N in self.runs['fileParts'].keys():
                self.runs['fileParts'][N] = {}
                
            fileList = list(Path(self.runs['fileBase']).rglob(f'{v.format(N=N)}.{ext}'))
            self.runs['fileParts'][N][k] = fileList

            if len(fileList) == 1:
                self.runs['files'][N][k] = fileList[0]
            else:
                # Flatten fileList for use with existing methods.
                # NOTE this is likely problematic - no concat for parts here

                # For HDF5 Can try:
                # (a) link style (https://docs.h5py.org/en/stable/high/group.html?highlight=external#external-links)
                # (b) virtual dataset (https://docs.h5py.org/en/stable/vds.html)
                # (c) convert to numpy and then concat (difficult to patch in current tmo routines).

                # FOR IMG case, assume format is t.run, or vice versa, for dict keys
                # self.runs['files'].update({f'{N}.{m+1}':item for m,item in enumerate(self.runs['fileParts'][N])})
                self.runs['files'].update({f'{m+1}.{N}.{k}':item for m,item in enumerate(self.runs['fileParts'][N][k])})


def readImgFiles(self, keyDims = ['run','X','Y'], subtract = None,
                    crop = None, rotation = None):
    """
    Read image files.

    keyDims : list
        Dims to use for output data, default case = ['run','X','Y']
        
    crop : dict, optional, default = None
        Dictionary of params to use for image crop if supplied.
        Supply with dict values: {'XC':,'YC':,'W':,'H':}
        This just uses raw array values (i.e. pixel values from raw files).
        
    rotation : float, optional, default = None
        Apply rotation correction (degrees) to image stack using scipy.ndimage.rotate.
        Note this currently assumes axes=(2, 1).
        
    subtract : list, optional, defualt = None
        Apply subtraction(s) to images.
        This will subtract all list items from the initial item.
        E.g. subtract = ['signal','bg'] will give signal-bg.
        Subtracted results will be set to 'subtracted' in the output image stack.


    For 'hit' data use base class readFiles().
    
    31/07/25  v2  Implement dict schema to allow multiple file patterns, e.g. signal and bg.
        NOTE: this is similar to existing VMI image gen with filters, and should have compatible output.

    14/07/25  v1
    
    """
    
#     if subtractBG:
#         self.runs['filePartsSignal']=self.runs['fileParts'].copy()
#         self.

    imgStack = {}
    
#     imgStack = xr.Dataset()  # May be able to just stack to Dataset? 
                                # See genVMIX() which stacks by filter. 

    # Read from "fileParts", indexed by (t,run)
    for k,fileListByType in self.runs['fileParts'].items():
        # for fileIn in fileList:
        #     imgIn = io.ImageCollection(fileIn)

        imgsIn = {}
        
        for fileType, fileList in fileListByType.items():
            
            fileStrList = [item.as_posix() for item in fileList]
            imgsIn[fileType] = io.ImageCollection(fileStrList)

        # Concat and subtract
        # This gives subtracted results PER CYCLE
        # NOTE: raw are uint8, so need to convert to float, either with ski or np.
        # NOTE: may also want to rescale? See https://scikit-image.org/docs/stable/user_guide/data_types.html#rescaling-intensity-values
#         if subtractBG:
#             # imgCycle = ski.util.img_as_float(imageS.concatenate()) - ski.util.img_as_float(imageB.concatenate())
# #             print("BG subtraction not yet implemented - need to set multiple file patterns.")
            
#         else:
#             imgCycle = ski.util.img_as_float(imgsIn.concatenate())

            imgCycle = ski.util.img_as_float(imgsIn[fileType].concatenate())
    
            # Crop
            if crop is not None:
                imgCycle = imgCycle[:,crop['XC']-crop['W']:crop['XC']+crop['W'],crop['YC']-crop['H']:crop['YC']+crop['H']]
                
            # Rotate if set
            if rotation is not None:
            #     tform = ski.transform.EuclideanTransform(rotation=imgRot*np.pi/180)
            #     imgCycle

                # *** Per https://scikit-image.org/docs/stable/user_guide/geometrical_transform.html#projective-transforms-homographies
                # Runs, but not sure how to define rotation axis here? Seems to rotate ALL axes?
                # tform = ski.transform.EuclideanTransform(rotation=imgRot*np.pi/180)
                # tf_img = ski.transform.warp(imgCycle, tform.inverse)

                # *** Basic version per https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.rotate
                # Also not clear if this rotates as expected, although seems closer to expectations!
                # Note this preserves array size by default (crops)
                # tf_img = ski.transform.rotate(imgCycle, imgRot)  #*np.pi/180,)

                #  *** Use scipy!
                #  Scipy: See https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.rotate.html#rotate
                #  Note this changes to array size by default, unless reshape=False is passed
                # TODO: futher tests here!
                imgCycle = scipy.ndimage.rotate(imgCycle, rotation, axes=(2, 1), reshape=False)
    
            # Set to output Xarray datastructure
            # imgT = xr.DataArray(imgCycle, dims=['cycle','X','Y'])
            # Q: stack initially by t or cycle? Cycle seems easier...
            coords = {k:np.arange(1,imgCycle.shape[n]+1) for n,k in enumerate(keyDims)}  # Basically same as XR defaults, but should pull from file names here.
            imgT = xr.DataArray(imgCycle, coords=coords)

            imgT.name = k
            
#             imgT.attrs['files'] = fileList
            # imgT.attrs['files'] = {'Signal':imageS.files,
            #                     'Background':imageB.files}

            # Subselect on cycles if set - UPDATE: now set in getFiles
            # if cycles is not None:
            #     imgT = imgT.sel(cycle=cycles)

            # Stack per t to dict
            # imgStackDict[t] = imgT.copy()

            # Stack per t to Xarray directly
#             if n==0:
            if fileType not in imgStack.keys():
                # XR dataset with (cycle,X,Y) array per t
                imgStack[fileType] = xr.Dataset({k:imgT})
                # Force attr - dropped on assignment otherwise
                imgStack[fileType].attrs['files'] = {k:fileList}
                
            else:
                imgStack[fileType] = imgStack[fileType].assign({k:imgT})
                imgStack[fileType].attrs['files'][k] = fileList
                

    
#             imgStack[fileType] = imgStack.copy() 
            
    # Restack from dict to full dataset...
    imgDS = xr.Dataset()
    for k in imgStack.keys():
        imgDS[k] = imgStack[k].to_array()
    
    imgDS = imgDS.rename({'variable':'t'})
    
    # Additional attrs
    imgDS.attrs['crop'] = crop
    imgDS.attrs['rotation'] = rotation
    
    
    self.imgStack = imgDS
    
    print("Assigned images to self.imgStack.")
    
    if subtract is not None:
        baseImg = imgDS[subtract[0]]
        
        for k in subtract[1:]:
            baseImg = baseImg - imgDS[k]
        
        self.imgStack['subtracted'] = baseImg
        print("Assigned subtracted images to self.imgStack['subtracted'].")