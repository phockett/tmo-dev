"""
Class and method dev for SLAC TMO data (Run 18) - VMI & image handling class.

Preprocessed data (h5py IO) + further processing + Holoviews.

20/11/20 v0.0.1

Paul Hockett

https://github.com/phockett/tmo-dev

"""

# Dev code for new class
# Inherit from base class, just add evmi functionality here
import tmo.tmoDataBase as tb
# from tmo.utils import _checkDims

import xarray as xr
import holoviews as hv

import numpy as np
from scipy.ndimage import gaussian_filter

class VMI(tb.tmoDataBase):

    from .utils import _checkDims

    def __init__(self, **kwargs):
        # Run __init__ from base class
        super().__init__(**kwargs)

        # Filter update - add multilevel filtering here.
        # SHOULD propagate back to base class, but keep here for now as they're only applied to image processing.
        # Set some default filters. Use self.setFilter() to change.
        self.filters = {'signal':{'gas':True,
                                  'desc': 'Signal filter.'},
                        'bg':{'gas':False,
                              'desc': 'Background filter.'},
                        }

    # def _checkDims(self):
    #     _checkDims

    def filterData(self, filterOptions = {}, keys = None, dim = 'energies'):
        """Wrapper for filterData when using nested filter (v2, 23/11/20)"""

        # Update filters if required
        if filterOptions:
            self.setFilter(filterOptions)

        # Default to all datasets
        if keys is None:
            keys = self.runs['proc']

        # Loop over filter sets and pass to base filterData() method.
        for key in self.filters.keys():
#             print(self.filters[key])
            super().filterData(filterOptions = self.filters[key], keys = keys, dim = dim)

            # Sort outputs to nested format
            # Note this leaves current settings in self.data[key]['mask']
            # These are STILL USED by histogram functions
            for runKey in keys:
                if key not in self.data[runKey].keys():
                    self.data[runKey][key] = {}  # Init

# REMOVED since it's confusing - will always leave last filter mask set!
# 10/12/20 REINSTATED in order to allow multiple filtering more generally... NEED TO PROPAGATE back to base class.
                self.data[runKey][key]['mask'] = self.data[runKey]['mask'].copy()
                self.data[runKey][key]['filterSet'] = key
                self.data[runKey][key]['filter'] = self.filters[key]
                self.data[runKey][key]['shots'] = self.data[runKey][key]['mask'].sum()


#     # 1st go... running, but very slow.
#     # Would probably be faster to write this all for np.arrays, rather than using existing image2d.
#     def genVMI(self, bgSub=True, norm=True, keys=None, filterOptions={}, **kwargs):
#         """Generate VMI images from event data, very basic hv.Image version."""
# #         Quick test for run 89 - looks to be working, different images for each case.
#
# #         Need to improve:
#
# #         - Cmapping (maybe log10?)
# #         - Image processing, use gaussian kernel?
# #         - Move to Xarray dataset image stack for more advanced/careful processing...?
#
# # %%timeit
# # 24.3 s ± 370 ms per loop (mean ± std. dev. of 7 runs, 1 loop each) LW06, Runs 89 - 97 (good only)
#
#         # Default to all datasets
#         if keys is None:
#             keys = self.runs['proc']
#
#         # Use existing image2d to generate images.
#         # This is possibly a bit slow, may be faster to rewrite 2D hist code (see old CIS codes for fast methods)
#         self.image2d(dim=['xc','yc'], filterOptions=filterOptions)  # keys=keys,
#
#         # Restack
#         self.eVMI = {'full':{}, 'fullNorm':{}, 'bg':{}, 'bgNorm':{}, 'bgSub':{}}  # Define output dicts
#         for key in keys:
#             self.eVMI['full'][key] = self.data[key]['img']
#
#             # Norm by no. events with gas.
#             # May want to norm by other factors too?
#             self.eVMI['fullNorm'][key] = self.data[key]['img'].transform(z=hv.dim('z')/np.array(self.data[key]['raw']['gas']).sum())
#
#         # Background images - assumed to be same filter(s) but gas off
#         # NOTE: MAY NEED TO USE EXPLICT .copy() here? TBC
#         if bgSub:
#             filterOptions['gas'] = [False]  # Hard coded for now, should allow passing for more flexibility
#             self.image2d(dim=['xc','yc'], filterOptions=filterOptions)  # keys=keys,
#
#             for key in keys:
#                 self.eVMI['bg'][key] = self.data[key]['img']
#                 self.eVMI['bgNorm'][key] = self.data[key]['img'].transform(z=hv.dim('z')/(~np.array(vmi.data[key]['raw']['gas']).astype(bool)).sum())
#
#                 # Direct data manipulation OK, returns numpy.ndarray
#                 # Seems easiest way to go for now.
#                 # But might be better with hv.transform methods?
#                 # Didn't need dim in testing... but did here. Odd!
#                 self.eVMI['bgSub'][key] = hv.Image(self.eVMI['fullNorm'][key].data['z'] - self.eVMI['bgNorm'][key].data['z'])


    def genVMIXmulti(self, filterOptions={}, **kwargs):
        """Wrapper for genVMIX with multiple filter sets."""

        if filterOptions is not None:
            self.setFilter(filterOptions = filterOptions)

        # Run genVMIX for each filter set
        # Note bgSub = False to avoid recursive run in current form (v2), but should update this
        for item in self.filters.keys():
            if self.verbose['main']:
                print(f'Generating VMI images for filters: {item}')

            # Pass only single filter set here.
            # Should change to avoid repetition of filtering.
            # self.genVMIX(bgSub=False, name=item, filterOptions = self.filters[item], **kwargs)
            self.genVMIX(name=item, filterOptions = self.filters[item], **kwargs)



    # 2nd go, stack to Xarrays for processing
    def genVMIX(self, norm=True, normType = 'shots', keys=None, filterOptions={},
                bins = (np.arange(0, 1048.1, 1)-0.5,)*2, dim=['xc','yc'], name = 'imgStack',
                bootstrap = False, lambdaP = 1.0, weights = None, density = None, **kwargs):
        """Generate VMI images from event data, very basic Xarray version.

        v2: allow for multi-level filter via genVMIXmulti wrapper, changed to super() for filter.
            TODO: clean this up, currently using a nasty mix of new and old functionality.
                  Also issues with ordering of functions, and whether some dicts. are already set.
                  (Should filter all, then genVMIX.)

        v1: single filter set with hard-coded, recursive bg subtraction.

        Parameters
        ----------
        norm : bool, default = True
            Normalise images. Currently norm by shots only.
            TODO: add options here.

        normType : str, optional, default = 'shots'
            Type of normalisation to apply. Currently norm by shots only.
            This is only used if norm=True
            TODO: add options here, rationalise logic.

        bootstrap : bool, optional, default = False
            If true, generate weights from Poission dist for bootstrap sampling using lambdaP parameter.
            https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Poisson_bootstrap
            (Alternatively, pass array weights)

        weights, density : options parameters for binning with Numpy.histogram2d, implemented for bootstrapping routine.
        https://numpy.org/devdocs/reference/generated/numpy.histogram2d.html#numpy.histogram2d

        Notes
        -----
        This currently uses Numpy.histogram2d, should be able to implment faster methods here.

        """

        # %%timeit
        # 18.8 s ± 63 ms per loop (mean ± std. dev. of 7 runs, 1 loop each) LW06, Runs 89 - 97 (good only)

        # Default to all datasets
        if keys is None:
            keys = self.runs['proc']

        if filterOptions is not None:
#             print(filterOptions)
#             self.filterData(filterOptions = filterOptions)
            super().filterData(filterOptions = filterOptions)  # Use super() for case of single filter set.

        # For Poission bootstrap, weights will be generated for each dataset.
        pFlag = False
        if bootstrap and (weights is None):
            rng = np.random.default_rng()
            pFlag = True

        # Current method below (as per Elio's code). For LW06 run 89 tests (~70k shots, ~7M events)
        # Single shot: 1.88 s ± 15.7 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        # All shots: 9.2 s ± 62.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        # All shots, convert to int: 11.5 s ± 399 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

        # See also psana, opal.raw.image(evt)

        # Loop over all datasets
        imgArray = np.empty([bins[0].size-1, bins[1].size-1, len(keys)])  # Set empty array
        normVals = []
        metrics = {'filterOptions':filterOptions.copy()}  # Log stuff to Xarray attrs

        for n, key in enumerate(keys):

            if self.verbose['main']:
                print(f"Generating VMI images for dataset {key}")

            # Initially assume mask can be used directly, but set to all True if not passed
            # Will likely want more flexibility here later
#             if mask is None:
#             mask = np.ones_like(self.data[key]['raw'][dim[0]]).astype(bool)

            # Check mask exists, set if not
            if 'mask' not in self.data[key].keys():
#                 self.filterData(keys=[key])
                super().filterData(keys=[key])  # Use super() for case of single filter set.

            # Note flatten or np.concatenate here to set to 1D, not sure if function matters as long as ordering consistent?
            # Also, 1D mask selection will automatically flatten? This might be an issue for keeping track of channels?
            # Should use numpy masked array...? Slower in testing.
            d0 = np.array(self.data[key]['raw'][dim[0]])[self.data[key]['mask']].flatten()
            d1 = np.array(self.data[key]['raw'][dim[1]])[self.data[key]['mask']].flatten()

            # If bootstrapping, generate Poission weights of not passed
            if pFlag:
                weights = rng.poisson(lambdaP, d0.size)

            # Histogram and stack to np array
            imgArray[:,:,n] = np.histogram2d(d0,d1, bins = bins, weights = weights, density = density)[0]

            # Normalisation options
            if normType is 'shots':
                normVals.append(self.data[key]['mask'].sum()) # shots selected - only for norm to no gas?
            else:
                normVals.append(1) # Default to unity

            metrics[key] = {'shots':self.data[key]['raw'][dim[0]].shape,
                            'selected':self.data[key]['mask'].sum(),
                            # 'gas':np.array(self.data[key]['raw']['gas']).sum(),  # This doesn't exist in new datasets, just remove for now.
                            'events':d0.size,
                            'normType':normType,
                            'norm':normVals[-1]}


            if name not in self.data[key].keys():
                self.data[key][name] = {}

            self.data[key][name]['metrics'] =  metrics[key].copy() # For mult filter case, push metrics to filter dict.
            self.data[key][name]['mask'] = self.data[key]['mask'].copy()
            self.data[key][name]['imgDims'] = dim  # Set for tracking dim shifts later/in plotting.

#         return imgArray
        # Convert to Xarray
        imgStack = xr.DataArray(imgArray, dims=[dim[0],dim[1],'run'],
                                coords={dim[0]:bins[0][:-1], dim[1]:bins[1][:-1], 'run':keys},
                                name = name)
        # 2nd attempt, swap dim labels & reverse y-dir. This maintains orientation for image plots.
        # imgStack = xr.DataArray(imgArray, dims=[dim[0],dim[1],'run'],
        #                         coords={dim[0]:bins[0][:-1], dim[1]:bins[1][-2::-1], 'run':keys},
        #                         name = name)

        imgStack['norm'] = ('run', normVals)  # Store normalisation values

        if norm:
            imgStack = imgStack/imgStack['norm']
            imgStack.name = name  # Propagate name! Division kills it

        imgStack.attrs['metrics'] = metrics

        # Recursive call for bg calculation, but set bgSub=False
        # CURRENTLY NOT WORKING - always get idential (BG only) results for both cases???
        # As constructed ALWAYS assumes 1st call will have bgSub = True
#         if bgSub:
#             self.imgStack = imgStack.copy()  # May need .copy() here?
#             filterOptions['gas'] = [False]  # Set bg as gas off, all other filter options identical
#             self.genVMIX(bgSub=False, norm=norm, keys=keys, filterOptions=filterOptions, bins = bins, **kwargs)
#         else:
#             self.imgStackBG = imgStack.copy().rename('BG')

        # Try keeping multiple results sets in stack instead.
        # This is a little ugly, but working.
        if not hasattr(self,'imgStack'):
            # self.imgStack = []  # May need .copy() here?  # v1, set as list
            self.imgStack = xr.Dataset()  # v2, set as xr.Dataset and append Xarrays to this

#         self.imgStack.append(imgStack.copy())  # May need .copy() here?  # v1
        self.imgStack[name] = imgStack.copy()  # v2 using xr.Dataset - NOTE THIS KILLS METRICS!
                                        # TODO: push to main dict, or coord?

# DEPRECATED - now just loop over filtersets with genVMIXmulti()
#         if bgSub:
#             filterOptions['gas'] = [False]  # Set bg as gas off, all other filter options identical
#             self.genVMIX(bgSub=False, norm=norm, name=name+'BG', keys=keys, filterOptions=filterOptions, bins = bins, **kwargs)
# #             self.imgStack.append((self.imgStack[-2] - self.imgStack[-1]).rename(name + 'BGsub'))  # Use last 2 img sets for subtraction
#             self.imgStack[name + 'BGsub'] = ((self.imgStack[name] - self.imgStack[name+'BG']).rename(name + 'BGsub'))

        # Restack final output to NxNxm Xarray for easy manipulation/plotting.
#         self.imgStack = self.imgStack.to_array(dim = 'type').rename('stacked')

#     def imgStacksub(self)

    # TODO: want to chain this for image plotting, but first set which array to use!
    # TODO: options for which dataset to use, just hacked in duplicate code for now.
    def restackVMIdataset(self, reduce = True, step = [5,5]):
        # Restack image dataset to NxNxm Xarray for easy manipulation/plotting.
        # Return rather than set internally...?

        if reduce:
            # if not hasattr(self, 'imgReduce'):
            #     # Run default reduce if not set
            #     self.downsample(step = step)

            # For ease-of-use, just always run this! Prevents issues when new imgStack items are added.
            self.downsample(step = step)


            # Restack, note transpose to force new dim ('type') to end.
            # This currently matters for smoothing function with scipy gaussian_filter.
            imgReduce = self.imgReduce.to_array(dim = 'name').rename('stacked')
            #.transpose('yc','xc','run','type')

            # Send new dim to end
            dimStack = imgReduce.dims
        #         self.imgStack = self.imgStack.transpose(*dimStack[1:],dimStack[0])
            return imgReduce.transpose(*dimStack[1:],dimStack[0])

        else:
            # Restack, note transpose to force new dim ('type') to end.
            # This currently matters for smoothing function with scipy gaussian_filter.
            imgStack = self.imgStack.to_array(dim = 'name').rename('stacked')
            #.transpose('yc','xc','run','type')

            # Send new dim to end
            dimStack = imgStack.dims
        #         self.imgStack = self.imgStack.transpose(*dimStack[1:],dimStack[0])
            return imgStack.transpose(*dimStack[1:],dimStack[0])


    def downsample(self, step = [2,2], dims = None):
        """Wrapper for xr.coarsen to downsample images by step.

        Set to trim boundaries, and sum over points. Coord system will be maintained.

        This will work for Dataset or Dataarray forms.
        Currently set to use smoothed dataset if available, or imgStack if not.

        TODO: add some options here.
        TODO: auto dim setting.
        """

        # TODO
        # if dims is None:
        #     dims = self.data[key][name]['imgDims']
        if dims is None:
            # dims = self.data[run][name]['imgDims']  # Set dims
            dims = list(self.imgStack.dims)[-1:0:-1]  # Use dims from Xarray (note ordering, list(FrozenSortedDict) needs reversing!)

        # v1 with list
#         self.imgReduce = []

#         for n, item in enumerate(self.imgStack):
# #             print(n)
#             self.imgReduce.append(item.coarsen({dim[0]:step[0], dim[1]:step[1]}, boundary="trim").sum())
# #             self.imgReduce[n] = item.coarsen({dim[0]:step[0], dim[1]:step[1]}, boundary="trim", keep_attrs=True).sum()

        # v2 with DataArray
#         self.imgReduce = self.imgStack.coarsen({dims[0]:step[0], dims[1]:step[1]}, boundary="trim").sum()
#         for item in ['imgStack', 'imgSmoothed']:
        if hasattr(self, 'imgSmoothed'):
            self.imgReduce = self.imgSmoothed.coarsen({d:s for (d,s) in zip(dims,step)}, boundary="trim").sum()
        else:
            self.imgReduce = self.imgStack.coarsen({d:s for (d,s) in zip(dims,step)}, boundary="trim").sum()


    def smooth(self, sigma = [1,1]):
    # Try using scipy.ndimage for smoothing...
    # Can use xr.apply_ufunc for this.
    # NOTE: this applies to ALL DIMS, so set 0 for additional stacking dims!
    # NOTE: this currently assumes dim ordering (not checked by name)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
    #
    # TODO: generalise to xr.Dataset http://xarray.pydata.org/en/stable/computation.html#math-with-datasets

    # smoothed = xr.apply_ufunc(gaussian_filter, imgReduce, 1)

    # v1 for lists
#         self.imgSmoothed = []

#         # TODO: add options for which stack to smooth
#         for item in enumerate(self.imgStack):
#             self.imgSmoothed.append(xr.apply_ufunc(gaussian_filter, item, sigma))  # Final value is sigma [dim0,dim1...]

        # v2 with DataArray
        # Set any additional dims to zero
        if len(sigma) != self.imgStack.ndims:
            sigma = np.pad(sigma, [0, self.imgStack.ndims - len(sigma)])

        self.imgSmoothed = (xr.apply_ufunc(gaussian_filter, self.imgStack, sigma))  # Final value is sigma [dim0,dim1...]

#     def fastHist(self, dims = ['xc','yc'], bins = [1048,1048], nullEvent = -9999):
#         """Generate 2D histogram via indexing. (Fast for small batches, but needs work for scale-up.)

#         NOTE: currently set for electron images, 1048x1048 bins.
#         To explore: see notes below!
#         """

#         hist2D = np.zeros(bins+2)  # Sized for 1-indexed bins, plus final cell for invalid hits.

#         # Index method... WAY FASTER single shot (orders of magnitude on np.histogram2d), moderately faster (~30%) all shots (but should be able to improve with Numba?)
#         # Lots of options here, for now use .flatten() to allow for np.delete below. Masking may be faster?
#         d0 = np.array(self.data[key]['raw'][dims[0]]).astype(int).flatten()
#         d1 = np.array(self.data[key]['raw'][dims[1]]).astype(int).flatten()

#         #  easy way to drop -9999 "no hits"?
#         d0 = np.delete(xhits, np.where(xhits == -9999))  # This only works for 1D case! May want to set NaNs instead?
#         d1 = np.delete(yhits, np.where(yhits == -9999))


#************* Plotting

    def showImg(self, run = None, name = 'signal', clims = None, hist = True, dims = None, swapDims = None,
                log10 = False, returnImg = False, backend = 'hv'):
        """
        Crude wrapper for hv.Image (or native Xarray plotter)

        Set backend = 'hv' (default), or 'xr'.

        Note:
        - dims = ['yc','xc'] by default (now set from input Xarray) - changing will flip image!
        - For plotting non-dimensional dims, pass as dims = [plot dims] and swapDims = [old dims].
        - hv backend doesn't colourmap well with log10 setting at the moment.
        - hv backend always uses reduced resolution image stack (TODO: add options here).

        """

        # Default to first run if not set
        if run is None:
            run = self.imgStack['run'][0].data.item()  # Force to single item

        if dims is None:
            # dims = self.data[run][name]['imgDims']  # Set dims
            dims = list(self.imgStack.dims)[-1:0:-1]  # Use dims from Xarray (note ordering, list(FrozenSortedDict) needs reversing!)

        else:
            self._checkDims(dataType = 'imgStack', dimsCheck = dims, swapDims = swapDims)


        if backend == 'xr':
            if log10:
                self.imgStack[name].sel(run=run).pipe(np.log10).plot.imshow()
            else:
                self.imgStack[name].sel(run=run).plot.imshow()

        if backend == 'hv':
            if log10:
                # log10 option - currently OK for .plot.imshow(), but doesn't cmap properly for hv if Nan/inf - need to check options here.
                hvImg = hv.Image(self.restackVMIdataset().sel(run=run, name=name).pipe(np.log10), kdims = dims).opts(aspect='square')
            else:
                hvImg = hv.Image(self.restackVMIdataset().sel(run=run, name=name), kdims = dims).opts(aspect='square')

            if clims is not None:

                hvImg = hvImg.redim.range(z=tuple(clims))

            # Code from showPlot()
            if self.__notebook__ and (not returnImg):
                if hist:
                    display(hvImg.hist())  # If notebook, use display to push plot.
                else:
                    display(hvImg)  # If notebook, use display to push plot.

            # Is this necessary as an option?
            if returnImg:
                return hvImg  # Otherwise return hv object.


    def showImgSet(self, run = None, name = 'signal', clims = None, hist = True, dims = None,
                    swapDims = None, returnImg = False):
        """
        Crude wrapper for hv.HoloMap for images - basically as per showImg(), but full map.

        Note:
        - dims = ['yc','xc'] by default, changing will flip image!
        - For plotting non-dimensional dims, pass as dims = [plot dims] and swapDims = [old dims].
        - Should add a dim check here for consistency.
        - name is currently not used for plotting, but should add this for flexibility - in some cases the cmap can be blown with multiple image sets on different scales.

        """

        # Check dims - TODO
        # list(self.restackVMIdataset().coords.keys())

        # Check image dims for kdims to pass.
        # Default to first run if not set
        # if run is None:
        #     run = self.imgStack['run'][0].data.item()  # Force to single item

        if dims is None:
            # dims = self.data[run][name]['imgDims']  # Set dims
            dims = list(self.imgStack.dims)[-1:0:-1]  # Use dims from Xarray (note ordering, list(FrozenSortedDict) needs reversing!)

        else:
            self._checkDims(dataType = 'imgStack', dimsCheck = dims, swapDims = swapDims)

        # Firstly set to an hv.Dataset
        imgDS = hv.Dataset(self.restackVMIdataset())

        # Then a HoloMap of images
        hvImg = imgDS.to(hv.Image, kdims=dims).opts(colorbar=True)


        if clims is not None:

            hvImg = hvImg.redim.range(z=tuple(clims))

        # Code from showPlot()
        if self.__notebook__ and (not returnImg):
            if hist:
                display(hvImg.hist())  # If notebook, use display to push plot.
            else:
                display(hvImg)  # If notebook, use display to push plot.

        # Is this necessary as an option?
        if returnImg:
            return hvImg  # Otherwise return hv object.
