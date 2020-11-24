"""
Class and method dev for SLAC TMO data (Run 18) - VMI & image handling class.

Preprocessed data (h5py IO) + further processing + Holoviews.

20/11/20 v0.0.1

Paul Hockett

https://github.com/phockett/tmo-dev

"""


# Dev code for new class
# Inherit from base class, just add evmi functionality here
class VMI(tmoDataBase):

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
            super().filterData(self, filterOptions = self.filters[key], keys = keys, dim = dim)

            # Sort outputs to nested format
            # Note this leaves current settings in self.data[key]['mask']
            # These are STILL USED by histogram functions
            for runKey in keys:
                self.data[runKey][key]['mask'] = self.data[runKey]['mask'].copy()
                self.data[runKey][key]['filter'] = self.filters[key]

    # 1st go... running, but very slow.
    # Would probably be faster to write this all for np.arrays, rather than using existing image2d.
    def genVMI(self, bgSub=True, norm=True, keys=None, filterOptions={}, **kwargs):
        """Generate VMI images from event data, very basic hv.Image version."""
#         Quick test for run 89 - looks to be working, different images for each case.

#         Need to improve:

#         - Cmapping (maybe log10?)
#         - Image processing, use gaussian kernel?
#         - Move to Xarray dataset image stack for more advanced/careful processing...?

# %%timeit
# 24.3 s ± 370 ms per loop (mean ± std. dev. of 7 runs, 1 loop each) LW06, Runs 89 - 97 (good only)

        # Default to all datasets
        if keys is None:
            keys = self.runs['proc']

        # Use existing image2d to generate images.
        # This is possibly a bit slow, may be faster to rewrite 2D hist code (see old CIS codes for fast methods)
        self.image2d(dim=['xc','yc'], filterOptions=filterOptions)  # keys=keys,

        # Restack
        self.eVMI = {'full':{}, 'fullNorm':{}, 'bg':{}, 'bgNorm':{}, 'bgSub':{}}  # Define output dicts
        for key in keys:
            self.eVMI['full'][key] = self.data[key]['img']

            # Norm by no. events with gas.
            # May want to norm by other factors too?
            self.eVMI['fullNorm'][key] = self.data[key]['img'].transform(z=hv.dim('z')/np.array(self.data[key]['raw']['gas']).sum())

        # Background images - assumed to be same filter(s) but gas off
        # NOTE: MAY NEED TO USE EXPLICT .copy() here? TBC
        if bgSub:
            filterOptions['gas'] = [False]  # Hard coded for now, should allow passing for more flexibility
            self.image2d(dim=['xc','yc'], filterOptions=filterOptions)  # keys=keys,

            for key in keys:
                self.eVMI['bg'][key] = self.data[key]['img']
                self.eVMI['bgNorm'][key] = self.data[key]['img'].transform(z=hv.dim('z')/(~np.array(vmi.data[key]['raw']['gas']).astype(bool)).sum())

                # Direct data manipulation OK, returns numpy.ndarray
                # Seems easiest way to go for now.
                # But might be better with hv.transform methods?
                # Didn't need dim in testing... but did here. Odd!
                self.eVMI['bgSub'][key] = hv.Image(self.eVMI['fullNorm'][key].data['z'] - self.eVMI['bgNorm'][key].data['z'])


    # 2nd go, stack to Xarrays for processing
    def genVMIX(self, bgSub=True, norm=True, keys=None, filterOptions={}, bins = (np.arange(0, 1048.1, 1)-0.5,)*2, **kwargs):
        """Generate VMI images from event data, very basic Xarray version."""

        # %%timeit
        # 18.8 s ± 63 ms per loop (mean ± std. dev. of 7 runs, 1 loop each) LW06, Runs 89 - 97 (good only)

        # Default to all datasets
        if keys is None:
            keys = self.runs['proc']

        if filterOptions is not None:
            self.filterData(filterOptions = filterOptions)

        # Current method below (as per Elio's code). For LW06 run 89 tests (~70k shots, ~7M events)
        # Single shot: 1.88 s ± 15.7 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        # All shots: 9.2 s ± 62.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        # All shots, convert to int: 11.5 s ± 399 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

        # See also psana, opal.raw.image(evt)

        # Loop over all datasets
        imgArray = np.empty([bins[0].size-1, bins[1].size-1, len(keys)])  # Set empty array
        norm = []
        for n, key in enumerate(keys):
            # Initially assume mask can be used directly, but set to all True if not passed
            # Will likely want more flexibility here later
#             if mask is None:
#             mask = np.ones_like(self.data[key]['raw'][dim[0]]).astype(bool)

            # Check mask exists, set if not
            if 'mask' not in self.data[key].keys():
                self.filterData(keys=[key])

            # Note flatten or np.concatenate here to set to 1D, not sure if function matters as long as ordering consistent?
            # Also, 1D mask selection will automatically flatten? This might be an issue for keeping track of channels?
            # Should use numpy masked array...? Slower in testing.
            d0 = np.array(self.data[key]['raw'][dim[0]])[self.data[key]['mask']].flatten()
            d1 = np.array(self.data[key]['raw'][dim[1]])[self.data[key]['mask']].flatten()

            # Stack to np array
            imgArray[:,:,n] = np.histogram2d(d0,d1, bins = bins)[0]

            norm.append(np.array(self.data[key]['raw']['gas']).sum()) # shots gas on

#         return imgArray
        # Convert to Xarray
        imgStack = xr.DataArray(imgArray, dims=[dim[0],dim[1],'run'],
                                coords={dim[0]:bins[0][0:-1], dim[1]:bins[1][0:-1], 'run':keys},
                                name = 'imgStack')

        imgStack['norm'] = ('run', norm)  # Store normalisation values

        if norm:
            imgStack = imgStack/imgStack['norm']

        self.imgStack = imgStack  # May need .copy() here?

        # Recursive call for bg calculation, but set bgSub=False
        if bgSub:
            filterOptions['gas'] = [False]  # Set bg as gas off, all other filter options identical
            self.genVMIX(bgSub=False, norm=norm, keys=keys, filterOptions=filterOptions, bins = bins, **kwargs)
        else:
            self.imgStackBG = self.imgStack
