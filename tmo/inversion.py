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
        hplot = img.hist() * contours * point

        # Code from showPlot()
        if self.__notebook__:
            display(hplot)
        else:
            return hplot



    def renorm(self, data = None, filterSet = None, norm={'type':'max', 'scope':'global'},
                XSmin = None, rPix = None, eRange = None):
        """Ugly renorm routine by spectrum, needs some dim checking!

        rMask : optional, slice, default = slice(1,-1)
            Radial mask used for data norm. Default avoids centre spot.
            NOTE: this is currently set by index (pixel).

        Norm options set in a dictionary, default = {'type':'max', 'scope':'global'}

        Currently the norm options are:

        type:
        - max
        - sum
        - raw

        scope:
        - global
        - local

        """

        if (data is None) and (filterSet is None):
            # Default to 1st dataset
            filterSet = self.proc.keys()[0]
            data = self.proc[filterSet]['xr']
        elif data is None:
            data = self.proc[filterSet]['xr']

        # Check masking & set if required
        # TODO: fix UGLY logic
        if (not hasattr(data, 'mask')) or (XSmin is not None) or (rPix is not None) or (eRange is not None):
            self.setRmask(filterSet, XSmin = XSmin, rPix = rPix, eRange = eRange)

        # Reset values to raw
        data.attrs['normType'] = norm
        data[0,:,:] = data['XS'].copy()  # Reset to raw XS values

        if norm['type'] is 'max':
            # data['norm'] = ('run', data['XS'][rMask,:].max('E'))  # Store per run values
            # gVal = data['XS'][rMask,:].max()
            data['norm'] = data['XS'].where(data['mask']).max('E')  # With preset mask
            gVal = data['norm'].max()

        elif norm['type'] is 'sum':
            # data['norm'] = ('run', data['XS'][rMask,:].sum('E'))
            # gVal = data['XS'][rMask,:].sum()
            data['norm'] = data['XS'].where(data['mask']).sum('E')  # With preset mask
            gVal = data['norm'].sum()

    #     elif norm['type'] is 'raw':
    #         data['norm'] = ('run', np.ones(data.run.size))

        else:
            data['norm'] = ('run', np.ones(data.run.size))
            gVal = 1

        # Renorm B00 (intensities) per run or globally.
        if norm['scope'] is 'global':
            data[0,:,:] /= gVal  # Renorm - note this currently assume dim ordering, not optimal!
        else:
            data[0,:,:] /= data['norm']




    def setRmask(self, filterSet, XSmin = 1e-2, rPix = [0,10], eRange = None):
        """Set radial masking based on XS values and pixel range - VERY CRUDE, needs work.

        Currently:

        XSmin : XS threshold, default = 1e-2.
        rPix : a pixel index range to mask out as a list, [rStart, rStop], default = [0,10]
        eSlice : an energy range to KEEP as a list, [eStart, eStop], default = None

        Note these are applied in order and the logic is additive.

        TODO:
        - add multiple ROI options.
        - mask in or mask out.
        - pass masks directly.
        - threshold by abs or percentage value.

        """

        # Init mask based on XS, this has (E, run) dims.
        if XSmin is not None:
            mask = (self.proc[filterSet]['xr']['XS'] > XSmin)  # Set mask per run, bool, currently will be set by (E, run)
        # mask = data.proc['signal']['xr'].where(data.proc['signal']['xr']['XS'] > 1e-2) # Return values
        else:
            mask = self.proc[filterSet]['xr']['XS']

        if rPix is not None:
            # mask[rPix[0]:rPix[1]] = False  # Mask pixel range
            mask = xr.where((mask['pixel']>rPix[0]) & (mask['pixel']<rPix[1]), False, mask) # Ensure size & dims consistent

        if eRange is not None:
            # self.proc[filterSet]['xr'].sel(E=slice(0.5,40))  # OK for subselecting
            # mask = xr.where(mask.sel(E=slice(0.5,10)), True, False)  # Use this to modify inplace values
            # mask = xr.where((mask['E']>eRange[0]) & (mask['E']<eRange[1]), True, False)  # Ensure size consistent
            mask = xr.where((mask['E']>eRange[0]) & (mask['E']<eRange[1]), mask, False)  # Ensure size & dims consistent

        self.proc[filterSet]['xr']['mask'] = mask



    def inv(self, filterSet = None, run = None, norm={'type':'max', 'scope':'global'}, step = [5,5],
            fold = True, quadFilter=[1, 1, 0, 0], basisR = 512, alpha=3.59e-4,
            sigma = None):
        """Basic wrapper for pbasex + fold routine.

        General options:
        filterSet = None, run = None, norm={'type':'max', 'scope':'global'}, step = [5,5]

        (sigma option for smoothing yet to be implemented.)

        cpBasex options:
        fold = True, quadFilter=[1, 1, 0, 0], basisR = 512, alpha=3.59e-4,

        TODO:
        - add smoothing option (need to propagate through restackVMIdataset() first)
        - move defaults to self.<cpbasex options>
        - outputs to Xarray
        - implement `quadrant.unfoldQuadrant` for full symmetrized images.
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

        procXR['pixel'] = ('E', np.arange(0, procXR['E'].size))  # Add pixel value index

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
            try:
                b0 = self.imgStack[dims[0]][np.arange(0,data.shape[0])]
                b1 = self.imgStack[dims[1]][np.arange(0,data.shape[1])]

            # Default to int coords, this allows for cases where sizes don't match (if resized for instance)
            except IndexError:
                b0 = np.arange(0,data.shape[0])
                b1 = np.arange(0,data.shape[1])

                if self.verbose['main']:
                    print(f"*** Inversion routine for filterSet = {filterSet} dropping back to default int coords.")


            self.imgStack[dset] = xr.DataArray(data, dims=[dims[0],dims[1],'run'],
                                                        coords={dims[0]:b0, dims[1]:b1, 'run':self.runs['proc']},
                                                        name = dset)

    #     self.imgStack[filterSet + '-inv'] = out['inv']
    #     self.imgStack[filterSet + '-fit'] = out['fit']

        # Rerun downsample to update self.reduce dataset
        self.downsample(step = step)



    # Similar to showImgSet code, but for spectral datasets [E,beta,run]
    def plotSpectra(self, filterSet = 'signal', overlay = 'BLM', returnMap = False, ePlot = None, useMask = True): # , rMask = slice(0.3,-1)):
        """
        Plot outputs from image inversion using Holoviews.

        By default use the existing mask. Pass `useMask=False` to ignore. An energy plot range can also be set as ePlot=[eStart,eStop].
        """


        # Firstly set to an hv.Dataset
        if useMask:
            eSpecDS = hv.Dataset(self.proc[filterSet]['xr'].where(self.proc[filterSet]['xr']['mask'], drop=True))
        else:
            eSpecDS = hv.Dataset(self.proc[filterSet]['xr'])

        # Then a HoloMap of curves
        # Crude radial mask for plot (assumes dims)
        # NOTE - slicing for hv.Dataset is set by VALUE not index!
        # TODO: unify mask settings with setRmask()
        # hmap = eSpecDS[:,rMask,:].to(hv.Curve, kdims=['E'])  # Version with basic mask

        if ePlot is not None:
            hmap = eSpecDS[:,slice(ePlot[0],ePlot[1]),:].to(hv.Curve, kdims=['E'])  # Version with basic mask
        else:
            hmap = eSpecDS.to(hv.Curve, kdims=['E'])


        # Code from showPlot()
        if self.__notebook__ and (not returnMap):
            if overlay is None:
                display(hmap)  # If notebook, use display to push plot.
            else:
                display(hmap.overlay(overlay))

        # Is this necessary as an option?
        if returnMap:
            return hmap  # Otherwise return hv object.


# ********* IN PROGRESS
    # def multiFilterInv(self, filterSet = None, sub = ['signal','bg']):
    #     """
    #     Basic multifilter automated routine for VMI images & processing.
    #
    #     NOTE - this currently assumes that analysis setup is already completed.
    #
    #     NOTE - filterSet not yet fully implemented (in genVMIXmulti).
    #
    #     Method: loop over filter parameter settings and recalculate images + inversion. This is a little inefficient, but works.
    #
    #     TODO:
    #     - parallelize.
    #     - more options for additional processing/subtraction.
    #     - stacking to existing array - currently just overwrites existing values.
    #
    #     """
    #
    #     # if not hasattr(self, 'stats'):
    #     #     self.stats = xr.Dataset()  # Create empty dataset
    #     self.multiFilter = xr.Dataset()  # Create empty dataset
    #
    #     if self.verbose['main']:
    #         print(f"*** Multi-filter run for filterSet={filterSet}")
    #
    #     # Repeat N analysis routines
    #     for n in np.arange(0,N):
    #
    #         if self.verbose['main']:
    #             print(f"Running set {n+1} of {N}")
    #
    #         # Generate VMI images with Poissionian weights (sampling)
    #         self.genVMIXmulti(bootstrap = True, lambdaP = lambdaP)
    #
    #         # Subtract datasets, or other processing?
    #         if sub is not None:
    #             self.imgStack['sub'] = self.imgStack[sub[0]] - self.imgStack[sub[1]]
    #
    #         # Invert image set
    #         self.inv(filterSet=filterSet)
    #
    #         # Restack results for run n
    #         self.stats[n] = self.proc[filterSet]['xr'].copy()


    def bootstrapInv(self, N = 5, lambdaP = 1.0, filterSet = None, sub = ['signal','bg']):
        """
        Basic bootstrap routine for VMI images & processing.

        NOTE - this currently assumes that analysis setup is already completed.

        NOTE - filterSet not yet fully implemented (in genVMIXmulti).

        For method, see https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Poisson_bootstrap

        TODO:
        - parallelize.
        - more options for additional processing/subtraction.
        - stacking to existing array - currently just overwrites existing values.

        """

        # if not hasattr(self, 'stats'):
        #     self.stats = xr.Dataset()  # Create empty dataset
        self.stats = xr.Dataset()  # Create empty dataset

        if self.verbose['main']:
            print(f"*** Bootstrap run for N={N}, lambda={lambdaP}, filterSet={filterSet}")

        # Repeat N analysis routines
        for n in np.arange(0,N):

            if self.verbose['main']:
                print(f"Running set {n+1} of {N}")

            # Generate VMI images with Poissionian weights (sampling)
            self.genVMIXmulti(bootstrap = True, lambdaP = lambdaP)

            # Subtract datasets, or other processing?
            if sub is not None:
                self.imgStack['sub'] = self.imgStack[sub[0]] - self.imgStack[sub[1]]

            # Invert image set
            self.inv(filterSet=filterSet)

            # Restack results for run n
            self.stats[n] = self.proc[filterSet]['xr'].copy()


    def plotUncertainties(self, returnMap = False):
        """
        Quick hack for plotting spectra + uncertainties. Needs some work.
        """

        # Set XR dataarray and convert to Holoviews dataset
        statsXR = self.stats.to_array().rename({'variable':'n'})
        statsXR.name = 'IE'
        statsHV = hv.Dataset(statsXR)[:,1:50,:]
        # statsHV = hv.Dataset(statsXR.where(self.mask))  # TODO: set to mask here, need to check statsXR structure first!

        # Set stats using HV functionality
        red = statsHV.reduce('n', np.mean, spreadfn=np.std)  # Reduce works... but only for a single series?

        # Manual stack for Spread + Curve objects
        # http://holoviews.org/user_guide/Building_Composite_Objects.html

        # ACTUALLY, doubly-nested dict more versatile (as used in main hist2d() function)
        # hmap = hv.HoloMap(kdims='run')
        #
        # for run in statsXR['run']:
        #     spreadDict = {2*i:hv.Spread(red.select(BLM=2*i, run = run), kdims=['E']) for i in range(0,3)}
        #     curveDict = {2*i:hv.Curve(red.select(BLM=2*i, run = run), kdims=['E']) for i in range(0,3)}
        #
        # #     group[run] = [spreadDict,curveDict]
        # #     (hv.HoloMap(spreadDict) * hv.HoloMap(curveDict)).overlay('Default')
        #     nboverlay = hv.NdOverlay(spreadDict, kdims = 'BLM') * hv.NdOverlay(curveDict, kdims = 'BLM') #.relabel(group='run', label = str(run))
        #
        #     hmap[run] = nboverlay

        # Generate set of
        spreadDict = {}
        curveDict = {}
        for run in statsXR['run'].data:
            spreadDict.update({(BLM, run):hv.Spread(red.select(BLM=BLM, run = run), kdims=['E']) for BLM in statsXR['BLM'].data})
            curveDict.update({(BLM, run):hv.Curve(red.select(BLM=BLM, run = run), kdims=['E']) for BLM in statsXR['BLM'].data})

        hmap = (hv.HoloMap(spreadDict, kdims=['BLM','run']) * hv.HoloMap(curveDict, kdims=['BLM','run']))

        # Code from showPlot()
        if self.__notebook__ and (not returnMap):
            display(hmap)  # If notebook, use display to push plot.

        # Is this necessary as an option?
        if returnMap:
            return hmap  # Otherwise return hv object.
