"""
Class and method dev for SLAC TMO data (Run 18) - stuff for correlations, in a class.

Preprocessed data (h5py IO) + further processing + Holoviews.

08/12/20 v0.0.1

Paul Hockett

https://github.com/phockett/tmo-dev

"""

# Dev code for new class
# Inherit from base class, just add evmi functionality here
import tmoDataBase as tb
from inversion import VMIproc  # VMI processing class - this has imaging methods + multilevel filtering.

import xarray as xr
import holoviews as hv

import numpy as np


# class corr(tb.tmoDataBase, VMIproc):   # TypeError: Cannot create a consistent method resolution order (MRO) for bases tmoDataBase, VMIproc
class corr(VMIproc):
    """
    Beginnings of correlation & convariance functions.

    TODO: consistent filtering for single/multiple filter case.
    Currently using paradigms from tmoDataBase, essentially single filter set.
    """

    def __init__(self, **kwargs):
        # Run __init__ from base class
        super().__init__(**kwargs)



    def corrRun(self, keys = None, dims = ['intensities', 'eShot'], filterOptions = None):
        """
        Compute covariance for specified dimensions, per run, with single filter set.

        Uses numpy.corrcoef (https://numpy.org/devdocs/reference/generated/numpy.corrcoef.html) to compute the matrices, which returns normalised covariance (correlation coefficients).



        TODO:

        - implement multiple filter sets here.

        """

         # Default to all datasets
        if keys is None:
            keys = self.runs['proc']

        if filterOptions is not None:
            self.filterData(filterOptions = filterOptions)

        for key in keys:
            # Check mask exists, set if not
            if 'mask' not in self.data[key].keys():
                self.filterData(keys=[key])

            # Get data and stack dims, assumes all dims are (shots x N), and stacks along N
            dimData = []
            # kdims = []  # Acutally not used - just set generic ['x','y'] for 2d plotting after dim stacking.
            labels = []
            for dim in dims:
                dimArray = self.getDataDict(dim, key, returnType = 'data')[self.data[key]['mask']]
                dimData.append(dimArray)

                # Set dim labels
                if dimArray.ndim == 1:
                    # kdims.append(dim)
                    labels.extend(dim)
                else:
                    # kdims.append(f'{dim} col')
                    # kdims.append(f'{dim} col')


                    # Crude dim mapping for some known types
                    if dim == 'intensities':
                        labels.extend([f'{n} ns' for n in np.array(self.data[key]['raw']['ts']).astype(str)])
                    else:
                        labels.extend([f'{dim}-{n}' for n in range(0, dimArray.shape[1])])  # Set arb labels for now - need to propagate from elsewhere!

            testData = np.concatenate(dimData, axis = 1)

    #         testCov = np.cov(testData, rowvar=False) # , bias=True)
            testCC = np.corrcoef(testData, rowvar=False)

            # self.data[key]['metrics']['cc'] = testCC
            # For now just label with n, since dims might be length (also would need conversion to tuple)
            # Should come up with a better method here!
            if 'cc' in self.data[key].keys():
                n = len(self.data[key]['cc']) + 1
            else:
                n = 0
                self.data[key]['cc'] = {}

            self.data[key]['cc'][n] = hv.HeatMap((labels, labels, testCC), ['x','y'], 'cc')
            # self.data[key]['metrics']['cc']

        # TODO: stack to dicts/HoloMap/other...?
        # Use existing histOverlay() method for this, using dummy var dim = n
        self.histOverlay(dim = n, pType='cc')

        if self.verbose['main']:
            print(f'Set self.ndoverlay, self.hmap and self.layout for dim={dims}.')



    def setSparse(self, keys = None, dims = ['xc', 'yc', 'shot'], fillValues = 1,  sparseObjFlag = True,
                    bounds = {'xc':[0,1018], 'yc':[0,1018], 'shot':lambda x: [0, x.shape[0]]}):  #, shots = None):
        """
        Set data to sparse array format. Suitable for per-shot imaging applications and correlations, or other ND sparse binning.

        Makes use of Sparse library, https://sparse.pydata.org/en/stable/

        This requires coordinate values for each point in the array which is not null, and (optionally) can set data values.

        Parameters
        ----------
        keys : list, default = None
            Keys (runs) to process.
            If None run for all keys.

        dims : list, default = ['xc', 'yc', 'shot']
            Dimensions to set in sparse array.
            Default values correspond to per-shot electron VMI images in 2020 datasets.

        fillValues : None, int or array, default = 1
            Fill values for sparse array.
            - If int, all points will be set to the same value, i.e. fillValues = 1 will count hits.
            - If array, this must match the size of the coords found.
            - If str, pass another data dimension to use here, e.g. set as 'pv' to use hit amplitudes.

        """

        # Default to all datasets
        if keys is None:
            keys = self.runs['proc']

        for key in keys:
            data = {}

            # Get data for each specified dim, sort and pass to Sparse
            for dim in dims:
                if dim is not 'shot':
                    data[dim] = self.getDataDict(dim, key, returnType = 'data')

                    # Set mask on out-of-range data. This will otherwise throw a coord error on conversion to sparse.
                    # This is general here, but usually will just be filtering out -9999 no hit value.
                    # Note mask is cumulative to enforce same data selection per dim
                    if 'idxMask' in data.keys():
                        data['idxMask'] *= (data[dim] >= bounds[dim][0]) & (data[dim] <= bounds[dim][1])
                    else:
                        data['idxMask'] = (data[dim] >= bounds[dim][0]) & (data[dim] <= bounds[dim][1])

            # Set shot coord, assumed to be all shots and match shape of dims[0] array
            data['shot'] = np.tile(np.arange(0, data[dims[0]].shape[0])[:, np.newaxis], (1, data[dims[0]].shape[1]))

            # Convert to sparse
            coords = [data[dim][data['idxMask']] for dim in dims]  # Set list of arrays with masking to pass as coords
            # TODO: this will fail for 1D vs. 2D data...? Need to check dims first?

            # Set fillValues from array if required
            if isinstance(fillValues, str):
                fillValues = self.getDataDict(fillValues, key, returnType = 'data')[data['idxMask']]

            # if 'shot' in bounds.keys():
                # bounds['shot'] = bounds['shot'](data[dims[0]])  # Set shots assuming lambda fn: doesn't make sense to do this unless more control is given?
            if 'shot' in dims:
                bounds['shot'] = [0, data[dims[0]].shape[0]]  # Assume # shots == dim[0] rows

            ndShape = tuple((bounds[k][1] - bounds[k][0] + 1) for k in bounds.keys())  # Set size as bounded interval? +1 to ensure coords span all values OK.

            # s = sparse.COO(coords, fillValues, shape = ndShape)
            if sparseObjFlag:
                s = sparse.COO(coords, fillValues, shape = ndShape)  # Return sparse array
            else:
                s = (coords, fillValues, ndShape)  # Return arrays rather than sparse object

            if self.verbose['main']:
                print(f'Set sparse array, dims = {dims}')

                # Print nice summary if possible
                if self.__notebook__:
                    display(s)
                else:
                    print(s)

            # Set to master structure
            # if 'sparse' in self.data[key]:

            self.data[key].update({'sparse': s})


            # Return sparse array
            # return s


    # Downsample array, from https://stackoverflow.com/a/55297292
    # This downsamples all axis by same int value (block)
    # Works by reshaping by block size, then summing over new axes
    # Modified here for different block sizes per dim
    # TODO: add cropping for dimensions to avoid block_size issues
    def downsampleSparse(sparray: sparse.COO, block_size: list):
        if any(x%block_size[n] != 0 for n,x in enumerate(sparray.shape)):
            return IndexError('One of the Array dimensions is not divisible by the block_size')

        axis = tuple(range(1, 2 * sparray.ndim + 1, 2))
    #     shape = tuple(index for s in sparray.shape for index in [(int(s/bsn), bsn) for bsn in block_size])
        shape = []
        for n, bsn in enumerate(block_size):
            shape.extend(index for index in (int(sparray.shape[n]/bsn), bsn))

        return sparray.reshape(shape).sum(axis=axis)


***********************************************************************
# TODO: finish this... just copied code over so far (22/12/20), need to implement
# - Sparse check/creation.
# - Set data for covariance, inc. filtering/multiple filters (for now wrap as per genVMIXmulti)
# - Downsample
# - Gen covar images & stack to Xarray (as per existing code)
# Protype code in https://pswww.slac.stanford.edu/jupyterhub/user/phockett/lab/tree/dev/VMI-correlations_sparse_tests_131220.ipynb

# Generate correlations with ion data - loop over pixels, may want to aggregate/downsample first?
# WITH DOWNSAMPLING, only set for image dims (otherwise need to ds on correlated shots too)

dsRatios = [4,4,1]  # Downsample dims [x,y,shot]

s2ds = downsample3(s2, [4,4,1])

# dsSize = [s2.shape[n]/dsRatios[n] for n in range(0,len(dsRatios))]
dsSize = s2ds.shape

corrImg = np.zeros([dsSize[0], dsSize[1]])

iData = np.array(data.data[key]['raw']['intensities'][shot, 0])

# For large (x,y) this is slow... should be able to parallelise.
# Parallel over one dim might be faster too, although will involve a lot of redundant calcs.
# Efficient way to reduce array first?
iRange = [0,dsSize[0]]
for x in np.arange(iRange[0],iRange[1]):
    for y in np.arange(iRange[0],iRange[1]):
        corrImg[x,y] += np.corrcoef(np.c_[s2ds[x,y,:].todense(),iData], rowvar=False)[0,-1]

**********************************************************************
    # Generate correlated VMI data.
    # Currently mirrors genVMIX for most boilerplate, so should clear up.
    def genCorrVMIX(self, norm=True, normType = 'shots', keys=None, filterOptions={},
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
