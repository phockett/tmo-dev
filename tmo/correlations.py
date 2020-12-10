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
