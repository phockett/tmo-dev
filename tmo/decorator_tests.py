"""
Class to test decorators for filter/method wrapper.
18/12/20

Just playing & testing mods, based on VMI class.

For tests: https://pswww.slac.stanford.edu/jupyterhub/user/phockett/lab/workspaces/tests/tree/dev/dectorator_tets_201220.ipynb

"""



from pathlib import Path
import sys
import inspect

import numpy as np
import xarray as xr
import holoviews as hv

import vmi as vmi

# Test decotrator with optional args using boilerplate from https://realpython.com/primer-on-python-decorators/#both-please-but-never-mind-the-bread
# Wrap key and filter logic?
# As is this WON'T WORK as desired, since runLoop() options will be set at decotrator declaration.
import functools
# def runLoop(_func=None, *, filterOptions={}, keys=None, filterSet=None):
def runFilterSetLoop(_func=None, *, demoArg = None):
    def decorator_runFilterSetLoop(func):
        @functools.wraps(func)

        # Wrapper with optional args, kwargs. Note specify self first for class method wrapping.
        def wrapper_runFilterSetLoop(self, *args, filterOptions={}, keys=None, filterSet=None, dim = 'energies', **kwargs):

            # Update filters if required
            filterUpdate = False
            if filterOptions:
                self.setFilter(filterOptions)
                filterUpdate = True

            # Default to all filters
            if filterSet is None:
                filterSet = self.filters.keys()

            # Default to all datasets
            if keys is None:
                keys = self.runs['proc']

            returnVals = {}  # Set optional return val dict.
            for key in keys:
                for filterKey in filterSet:

                    # *** With existing fns. (inc. filter)
                    # value = func(*args, keys = key, filterSet = filterKey, **kwargs)  # Not sure if this will work - self will be fist *arg, then rest keyword?

                    # *** Consolidate filter logic...
                    # Move filter logic here
                    if filterUpdate or (not 'mask' in self.data[key][filterKey].keys()):
                        # super().filterData(filterOptions = self.filters[filterKey], keys = key, dim = dim)
                        filterData(filterOptions = self.filters[filterKey], keys = key, dim = dim)

                    value = func(*args, keys = key, **kwargs)  # Not sure if this will work - self will be fist *arg, then rest keyword?

                    returnVals[key].update({filterSet:value})

            return returnVals

        return wrapper_runFilterSetLoop

    if _func is None:
        return decorator_runFilterSetLoop
    else:
        return decorator_runFilterSetLoop(_func)



# Class to test decorators for filter/method wrapper.
# Inherit from base VMI class.
class decTest(vmi.VMI):

    def __init__(self, method='cpbasex', **kwargs):
        # Run __init__ from base class
        super().__init__(**kwargs)

        self.method = method
