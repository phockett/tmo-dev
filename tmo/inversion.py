"""
Wrappers & methods for vmi image inversion.

"""


from pathlib import Path
import sys
import inspect

import numpy as np
import xarray as xr
import holoviews as hv

import vmi as vmi

# Set cpbasex imports
# Imports adapted from Elio's code from tmolw0618/scratch/results/tmo_ana/elio/elec_pbasex.ipynb
# Code adapted from similar method in ePSproc.orbPlot.importChemLabQC
# This will import pbasex + helper functions if found

def importCPBASEX(pbasexPath = None, basisPath = None, imgFlag = True):
    """
    Import cpBasex & additional functions.

    This is set as optional, and a local path can be defiened, since it may already be installed in local env., or in data processing location.

    A different path can also be specified for the basis sets required.

    Code:
    - CPBASEX: https://github.com/e-champenois/CPBASEX
    - Quadrant: https://github.com/e-champenois/quadrant
    - Polar-Rebinning: https://github.com/e-champenois/Polar-Rebinning

    For local install instructions, see https://github.com/e-champenois/CPBASEX#installation

    To do: add more error parsing and logging, see, e.g., https://airbrake.io/blog/python-exception-handling/importerror-and-modulenotfounderror

    NOTE: in testing, some versions had issues with loading the basis functions - possibly a version mismatch issue? TBC.

    For LW06 processing, setting local imports as follows is working:
    `pbasexPath = '/reg/d/psdm/tmo/tmolw0618/results/modules/pbasex'`
    `basisPath = '/reg/d/psdm/tmo/tmolw0618/scratch/results/tmo_ana/calc/G_r512_k128_l4.h5'`

    To debug, use inspect to check imported module paths, e.g.
    ```
        import inspect
        inspect.getfile(pbasex)
    ```

    Or importlib
    ```
        # Try setting specific version for import...
        # See https://stackoverflow.com/a/67692
        import importlib.util
        spec = importlib.util.spec_from_file_location('pbasex', '/reg/d/psdm/tmo/tmolw0618/results/modules/pbasex/pbasex.py')
        foo = importlib.util.module_from_spec(spec)
    ```

    """

    cpImport = False

    # If pbasex is locally installed, this should just work...
    if pbasexPath is None:
        try:
            import pbasex
            import quadrant

            cpImport = True

        except ImportError:
            pass

    if pbasexPath is not None:
        try:
            # Load specific version from path
            # This might fail if there is also a version installed in the current environment
            # sys.path.append(pbasexPath)
            # Force to head of path to ensure local version loaded
            sys.path.insert(0, pbasexPath)

            # Optional - to force load from specific file location
            # See https://stackoverflow.com/a/67692
            # import importlib.util
            # spec = importlib.util.spec_from_file_location('pbasex', '/reg/d/psdm/tmo/tmolw0618/results/modules/pbasex/pbasex.py')
            # foo = importlib.util.module_from_spec(spec)


            # Import from local dir, assuming both modules present for the moment
            import pbasex
            import quadrant

            cpImport = True

        except ImportError:
            pass
    # else:
    #     pbasexPath = Path(inspect.getfile(pbasex)).parent.as_posix()

    if not cpImport:
        print('cpBasex not found.')
        return None
    else:
        print(f'cpBasex import OK, checking for basis functions...')

    try:
        # if basisPath is None:
        #     basisPath = pbasexPath  # This could work if files are parsed.

        if basisPath is not None:
            gBasis = pbasex.loadG(basisPath, make_images = imgFlag)
            print(f'Found basis at {basisPath}.')
        else:
            gBasis = None
            print('Basis file not set, please run importer again with basisPath=<gBasis file location>')

    except OSError:
        print(f'*** Failed to load basis from {basisPath}.')

    return (pbasex, quadrant, gBasis)


# Class to hold inversion methods.
# Inherit from base VMI class.
class VMIproc(vmi.VMI):

    def __init__(self, method='cpbasex', **kwargs):
        # Run __init__ from base class
        super().__init__(**kwargs)

        self.method = method

    # Import specific method
    def setMethod(self, **kwargs):
        if self.method == 'cpbasex':
            method = importCPBASEX(**kwargs)

            # Unpack tuple if returned
            if method is not None:
                self.cp, self.qu, self.gBasis = method

        else:
            print(f'No method {self.method}.')

    # Set & check centre coods
    def setCentre(self, imgCentre = None, dims = None, filterSet = 'signal'):
        """Set and check image centre coords"""

        # If dims is none, use 1st two axes from image stack as default
        if dims is None:
#             dims = self.imgStack[filterSet].dims[0:2]
            dims = list(self.imgStack.dims)[-1:0:-1]  # Use this for consistency with base class.
                                                    # Note order is REVERSED!

        # Take centre as max point if not passed
        if imgCentre is None:
            imgInd = np.where(self.imgStack[filterSet] == self.imgStack[filterSet].max())  # Return inds to max point
            # Bit circular, but get back pixel coords here for consistency with rest of function
            # Note ordering reversed since dims reversed above!
            imgCentre = [self.imgStack[filterSet][dims[1]][imgInd[1]].data.item(),
                         self.imgStack[filterSet][dims[0]][imgInd[0]].data.item()]

        self.centreInds = {'input':imgCentre, 'dims':dims}
        cList = []
        dMax = []
        for n, dim in enumerate(dims):
    #         print(dim)
            # Find nearest point, then indicies with np.where?  Bit ugly , but seems to work, although it shouldn't???
            # Should be a native Xarray method for this?
            nearestPoint = self.imgStack[filterSet][dim].sel({dim:imgCentre[n]}, method='nearest')
            ind = np.where(self.imgStack[filterSet][dim] == nearestPoint)[0]
            # xInd = np.nonzero(data.imgStack['signal'].xc == nearestX)
            self.centreInds[dim] = {'coord':nearestPoint.data, 'index':ind}
            cList.append(int(ind))  # Force to int
            dMax.append((self.imgStack[filterSet][dim].max() - nearestPoint).data.item())  # Roughly set max dist to edge

        self.centreInds['cList'] = cList
        self.centreInds['dMax'] = dMax

    #         data.imgStack['signal'].xc[xInd]  # Can index INTO array with index.

    def checkCentre(self, nContour = 15, rMax = None, **kwargs):
        # Set circles
        def circle(radius):
            angles = np.linspace(0, 2*np.pi, 300)
            return {'x': radius*np.sin(angles) + self.centreInds['input'][0],
                    'y': radius*np.cos(angles) + self.centreInds['input'][1], 'radius': radius}

        # Set img
        img=self.showImg(returnImg = True, **kwargs)

        # Circle overlay as contours
        # As per http://holoviews.org/reference/elements/bokeh/Contours.html
        if rMax is None:
            rMax = self.centreInds['dMax'][0] * 0.7
        contours = hv.Contours([circle(i) for i in np.linspace(0, rMax, nContour).round()], vdims='radius')

        # As polygons
    #     hv.Polygons([{('x', 'y'): hv.Ellipse(0, 0, (i, i)).array(), 'z': i} for i in range(1, 10)[::-1]], vdims='z')

        # Centre point
        point = hv.Points([img.closest([(tuple(self.centreInds['input']))])]).opts(color='b', marker='x', size=10)


    #     sampleStep = 2
    #     display(hv.Image(im[:sampleStep:-1,:sampleStep:-1]) * contours)
        display(img.hist() * contours * point)


    def renorm(self, data = None, filterSet = None, norm={'type':'max', 'scope':'global'}, rMask = slice(1,-1)):
        """Ugly renorm routine by spectrum, needs some dim checking!

        rMask : optional, slice, default = slice(1,-1)
            Radial mask used for data norm. Default avoids centre spot.

        """

        if (data is None) and (filterSet is None):
            # Default to 1st dataset
            data = self.proc[self.proc.keys()[0]]['xr']
        elif data is None:
            data = self.proc[filterSet]['xr']

        # Reset values to raw
        data.attrs['normType'] = norm
        data[0,:,:] = data['XS'].copy()  # Reset to raw XS values

        if norm['type'] is 'max':
            data['norm'] = ('run', data['XS'][rMask,:].max('E'))  # Store per run values
            gVal = data['XS'][rMask,:].max()

        elif norm['type'] is 'sum':
            data['norm'] = ('run', data['XS'][:,rMask].sum('E'))
            gVal = data['XS'][rMask,:].sum()

    #     elif norm['type'] is 'raw':
    #         data['norm'] = ('run', np.ones(data.run.size))

        else:
            data['norm'] = ('run', np.ones(data.run.size))

        # Renorm B00 (intensities) per run or globally.
        if norm['scope'] is 'global':
            data[0,:,:] /= gVal  # Renorm - note this currently assume dim ordering, not optimal!
        else:
            data[0,:,:] /= data['norm']




    def setRmask(self, filterSet, XSmin = 1e-3, rPix = [1, -1]):
        """Set radial masking based on XS values and pixel range - VERY CRUDE, needs work."""
        mask = (self.proc[filterSet]['xr']['XS'] > XSmin)  # Set mask per run, bool
        # mask = data.proc['signal']['xr'].where(data.proc['signal']['xr']['XS'] > 1e-2) # Return values
        mask[rPix[0]:rPix[1]] = False  # Add pixel range
        self.proc[filterSet]['xr']['mask'] = mask


    def inv(self, filterSet = None, run = None, norm={'type':'max', 'scope':'global'}, step = [5,5],
            fold = True, quadFilter=[1, 1, 0, 0], basisR = 512, alpha=3.59e-4):
        """Basic wrapper for pbasex + fold routine.

        TODO:
        - add smoothing option.
        - move defaults to self.<cpbasex options>
        - outputs to Xarray
        """


        if filterSet is None:
            imgIn = self.restackVMIdataset(reduce=False)  # Pass NxNx(runsxfilters) data
                                                        # Note this includes transpose, may be problematic here?
            filterSet = 'All'

        else:
            imgIn = self.imgStack[filterSet].data # Pass as NxNxruns data


        # Symmetrize image with Fold?
        if fold:
            fold = self.qu.resizeFolded(self.qu.foldQuadrant(imgIn, self.centreInds['cList'][0], self.centreInds['cList'][1],
                                         quadrant_filter=quadFilter), basisR)
        else:
            fold = imgIn

        # Transpose images to keep pol axis on vertical (?)
        # Note this currently assumes dims (img1,img2,runs)
        fold = fold.transpose(1,0,2)

        # Run inversion
        out = self.cp.pbasex(fold, self.gBasis, alpha=alpha, make_images=True)


        # Convert to Xarray
        betaList = np.arange(2, 2*out['betas'].shape[1]+1,2)  # Inferred...?
        betas = xr.DataArray(out['betas'], coords={'E':out['E'], 'BLM':betaList, 'run':self.runs['proc']},
                            dims = ['E','BLM','run'], name='betas')

        IE = xr.DataArray(out['IE'], coords={'E':out['E'], 'run':self.runs['proc']},
                            dims = ['E','run'], name="IE")  #.expand_dims({'BLM':[0]})

    #     procXR = xr.merge([IE, betas]).to_array().rename({'variable':'type'})  # With xr.merge
        procXR = IE.expand_dims({'BLM':[0]}).combine_first(betas)  # Direct combine to da
        procXR['XS'] = IE.copy()  # Set raw XS data too
    #     procXR['rMask'] = procXR['XS'].where()

        # Norm intensities - NOW SET IN SEPARATE FUNCTION
    #     procXR.attrs['normType'] = norm
    #     if norm['type'] is 'max':
    #         procXR['norm'] = ('run', procXR.sel(BLM=0).max('E'))  # Store per run values
    #         gVal = procXR.sel(BLM=0).max()
    #     elif norm['type'] is 'sum':
    #         procXR['norm'] = ('run', procXR.sel(BLM=0).sum('E'))
    #         gVal = procXR.sel(BLM=0).sum()
    #     else:
    #         procXR['norm'] = ('run', np.ones(procXR.run.size))

    #     # Renorm B00 (intensities) per run or globally.
    #     if norm['scope'] is 'global':
    #         procXR[0,:,:] /= gVal  # Renorm - note this currently assume dim ordering, not optimal!
    #     else:
    #         procXR[0,:,:] /= procXR['norm']


        # Log results
        if not hasattr(self, 'proc'):
            self.proc = {}

        self.proc[filterSet] = {}
        self.proc[filterSet]['fold'] = fold  # Set here for now, but should pass back to imgStack for use with plotting methods.
        self.proc[filterSet]['pbasex'] = out
        self.proc[filterSet]['xr'] = procXR

        self.setRmask(filterSet = filterSet)
        self.renorm(filterSet = filterSet, norm = norm)  # Run renorm routine

        # Set image stacks for use with current plotters
        # Bit ugly! Also not sure how to fix/force dims here as yet

        dims = list(self.imgStack.dims)[-1:0:-1]  # Use this for consistency with base class.
                                                        # Note order is REVERSED!
        for name in ['fold','inv','fit']:
            dset = filterSet + '-' + name

            if name is 'fold':
                data = fold
            else:
                data = out[name]

            # Set arb bins here to allow for resized arrays - may need some work!
            # ACTUALLY, use existing bins, which should be OK...?
            b0 = self.imgStack[dims[0]][np.arange(0,data.shape[0])]
            b1 = self.imgStack[dims[1]][np.arange(0,data.shape[1])]

            self.imgStack[dset] = xr.DataArray(data, dims=[dims[0],dims[1],'run'],
                                                        coords={dims[0]:b0, dims[1]:b1, 'run':self.runs['proc']},
                                                        name = dset)

    #     self.imgStack[filterSet + '-inv'] = out['inv']
    #     self.imgStack[filterSet + '-fit'] = out['fit']

        # Rerun downsample to update self.reduce dataset
        self.downsample(step = step)


    # Similar to showImgSet code, but for spectral datasets [E,beta,run]
    def plotSpectra(self, filterSet = 'signal', overlay = 'BLM', returnMap = False):

        # Firstly set to an hv.Dataset
        eSpecDS = hv.Dataset(self.proc[filterSet]['xr'])

        # Then a HoloMap of curves
        # Crude radial mask for plot (assumes dims)
        # TODO: unify mask settings with setRmask()
        hmap = eSpecDS[:,1:50,:].to(hv.Curve, kdims=['E'])


        # Code from showPlot()
        if self.__notebook__ and (not returnMap):
            if overlay is None:
                display(hmap)  # If notebook, use display to push plot.
            else:
                display(hmap.overlay(overlay))

        # Is this necessary as an option?
        if returnMap:
            return hmap  # Otherwise return hv object.
