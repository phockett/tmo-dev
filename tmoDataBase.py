"""
Class and method dev for SLAC TMO data (Run 18) - base class (IO + basic analysis and plots).

Preprocessed data (h5py IO) + further processing + Holoviews.

18/11/20 v0.0.1

Paul Hockett

https://github.com/phockett/tmo-dev

"""


import numpy as np
from h5py import File
from pathlib import Path

# HV imports
import holoviews as hv
from holoviews import opts
hv.extension('bokeh', 'matplotlib')

# Set some default plot options
def setPlotDefaults(fSize = [800,400]):
    """Basic plot defaults"""
    opts.defaults(opts.Curve(width=fSize[0], height=fSize[1], tools=['hover'], show_grid=True),
                  opts.Image(width=fSize[0], aspect='square', tools=['hover'], colorbar=True),   # Force square format for images (suitable for VMI)
                  opts.HexTiles(width=fSize[0], height=fSize[1], tools=['hover'], colorbar=True))


class tmoDataBase():
    """
    Very crude data handling & plotting class for TMO run 18 preprocessed data.

    Reads h5py files & plots various properties & correlations using Holoviews.

    18/11/20 v0.0.1

    Paul Hockett

    https://github.com/phockett/tmo-dev


    #TODO: histND & pyvista
    Pull run info from elog
    Support for raw data


    """


    __version__ = '0.0.1'
    __notebook__ = self.isnotebook()

    def __init__(self, fileBase = None, ext = 'h5', runList = None, fileSchema=None, fileList=None, verbose = 1):
        # Set file properties
        self.runs = {'fileBase':Path(fileBase),
                     'ext':ext,
#                      'prefix':prefix,
                     'fileList':fileList,
                     'runList':runList,
                     'files': {N:Path(fileBase, f'run{N}{fileSchema}.{ext}') for N in runList}
                     }


        # Set for main functions and subfunctions
        self.verbose = {'main':verbose, 'sub':verbose-1}

        if self.verbose['sub'] < 0:
            self.verbose['sub'] = 0

#**** IO
    def readFiles(self):
        # Set data per run
        self.data = {}

        # Set h5 file object - should just be file pointers (?)
        # Yep, OK, returns items <HDF5 file "run93_preproc_elecv2.h5" (mode r)>
        # No mem usage (checked with %memit, https://timothymonteath.com/articles/monitoring_memory_usage/)
        for key in self.runs['files'].keys():
            self.data[key] = {'raw':File(self.runs['files'][key])}

        # Additionally pull some useful metrics
        # Does h5py object already report some of this directly? Not sure.
        self.runs['proc'] = []
        self.runs['invalid'] = []
        for key in self.data.keys():
            self.data[key]['items'] = self.data[key]['raw'].keys()
            self.data[key]['dims'] = {item:self.data[key]['raw'][item].shape for item in self.data[key]['raw'].keys()}

            # Very basic IO check, if energies is missing dataset may be problematic?
            # TODO: use unified dict or sets to check consistent dims over all datasets.
            if 'energies' not in self.data[key]['dims']:
                print(f'*** WARNING: key {key} missing energies data, will be skipped.')
                self.runs['invalid'].append(key)
            else:
                self.runs['proc'].append(key)

        if self.verbose['main']:
            print(f"Read {len(self.data)} files.")
            print(f"Good datasets: {self.runs['proc']}")
            print(f"Invalid datasets: {self.runs['invalid']}")

#**** ANALYSIS
    def setFilter(self, filterOptions = {}):
        """
         Master filter settings.

        Now updated to
        - maintain filter for multiple settings.
        - update or replace existing filter.
        - Set multiple filter types. (This is now set for image processing only in :py:class:`vmi`.)
        - Set filter functions for derived data (e.g. total counts)

        NOTE: Multilevel filter is now set for image processing only in :py:class:`vmi`.
        Only the full multilevel filter is set here, for histogram functions pass options to method independently (for now).
        This will set masks to self.data[key][filterName][mask], while old methods set self.data[key]['mask'].

        """

        # Loop over input filterOptions and set
        for key,val in filterOptions.items():

            # Add an item globally
            if type(val) is not dict:
                for masterKey in self.filters.keys():
                    self.filters[masterKey][key] = val

            # Add item to subset only
            else:
                for optKey, optVal in val.items():
                    if key in self.filters.keys():
                        self.filters[key][optKey] = optVal
                    else:
                        self.filters[key] = val  # Create new item from input dict.



    def filterData(self, filterOptions = {}, keys = None, dim = 'energies'):
        """
        Very basic filter/mask generation function.

        filterOptions : dict, containing {dim:values} to filter on.
            Singular values are matched.
            Pairs of values as used as ranges.
            For multidim parameter sets, specify which source column to use as 3rd parameter.

        keys : list, optional, default = None
            Datasets to process, defaults to self.runs['proc']

        dim : str, optional, default = 'energies'
            Data to use as template. Not usually required, unless multidim return and/or default data is missing.

        TODO:

        - More flexibility.
        - Filter functions, e.g. saturated electron detector shots? ('xc' > 0).sum(axis = 1) <=1000 in this case I think.

        """

        # Default to all datasets
        if keys is None:
            keys = self.runs['proc']

        for key in keys:

            # Set full mask = true, use passed dim ('energies' as default) to define datasize event dim
            mask = np.ones_like(self.data[key]['raw'][dim]).astype(bool)
            if len(mask.shape)>1:
                mask = mask[:,0]

            for item in filterOptions.keys():

                testData = np.array(self.data[key]['raw'][item])

                # Match single items
                if len(filterOptions[item])==1:
                    mask *= (testData == filterOptions[item])

                if len(filterOptions[item])==2:
                    mask *= (testData >= filterOptions[item][0]) & (testData <= filterOptions[item][1])

                # Case for multidim testData
                if len(filterOptions[item])==3:
                    mask *= (testData[:,filterOptions[item]] >= filterOptions[item][0]) & (testData[:,filterOptions[item]] <= filterOptions[item][1])

            self.data[key]['mask'] = mask  # For single filter this is OK, for multiples see vmi version.




    # def checkDims(testArray):
    #     """Check if array is 2D"""

#**** PLOTTING STUFF

    def isnotebook():
    """
    Check if code is running in Jupyter Notebook.

    Taken verbatim from https://exceptionshub.com/how-can-i-check-if-code-is-executed-in-the-ipython-notebook.html
    Might be a better/more robust way to do this?

    """

    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


    def showPlot(self, dim, run, pType='curve'):
        """Render (show) specified plot from pre-set plot dictionaries (very basic!)."""

        try:
            if self.__notebook__:
                display(self.data[run][pType][dim])  # If notebook, use display to push plot.
            else:
                return self.data[run][pType][dim]  # Otherwise return hv object.

        except:
            print(f'Plot [{run}][{pType}][{dim}] not set.')


    def curvePlot(self, dim, filterOptions = None, keys = None):
        """Basic wrapper for hv.Curve, currently assumes a 1D dataset, or will skip plot."""

        # Default to first dataset
        if keys is None:
            keys = self.runs['proc'][0]


        for key in keys:
            # Check mask exists, set if not
            if 'mask' not in self.data[key].keys():
                self.filterData(keys=[key])

            d0 = np.array(self.data[key]['raw'][dim[0]])[self.data[key]['mask']]

            try:
                if self.__notebook__:
                    display(hv.Curve(d0))  # If notebook, use display to push plot.
                else:
                    # return hv.Curve(d0)  # Otherwise return hv object.
                    pass
            except:
                pass


    def hist(self, dim, bins = 'auto', filterOptions = None, keys = None):
        """
        Construct 1D histrograms using np.histogram.

        """
        # Default to all datasets
        if keys is None:
            keys = self.runs['proc']

        # Loop over all datasets
        curveDict = {}
        for key in keys:
            # Check mask exists, set if not
            if 'mask' not in self.data[key].keys():
                self.filterData(keys=[key], dim = dim)  # Pass dim or use default here - issue with passing is additional dim checks required to avoid collapsing multidim data

            d0 = np.array(self.data[key]['raw'][dim])[self.data[key]['mask']]

            # Check cols, otherwise will be flatterned by np.histogram
            # TODO: move to separate function, and use .ndim (OK for h5 and np arrays)
            d0Range = 1
            if len(d0.shape)>1:
                d0Range = d0.shape[1]
#                 kdims.append(f'{dim[1]} col')
            else:
                d0 = d0[:,np.newaxis]

                # This doesn't work due to ordering of np.histogram returns! Doh!
#             curveDict[key] = {f'{key},{i}':hv.Curve((np.histogram(d0[:,i], bins)), dim, 'count') for i in np.arange(0,d0Range)}
            curveDict[key] = {}
            for i in np.arange(0,d0Range):
                freq, edges = np.histogram(d0[:,i], bins)
#                 curveDict[key][i] = hv.Curve((edges, freq), dim, 'count')
                # curveDict[key][f'{i} ({key})'] = hv.Curve((edges, freq), dim, 'count')  # Keep run label here for histOverlay, although might be neater way
                curveDict[key][(key, i)] = hv.Curve((edges, freq), kdims=dim, vdims='count') # Try as tuples, see http://holoviews.org/reference/containers/bokeh/NdOverlay.html

# TODO: consider best stacking here dict, holomap, ndoverlay...?
#             if 'curve' in self.data[key].keys():
#                 self.data[key]['curve'][dim] = hv.HoloMap(curveDict)
#             else:
#                 self.data[key]['curve'] = {dim:hv.HoloMap(curveDict)}

            if 'curve' in self.data[key].keys():
                self.data[key]['curve'][dim] = hv.NdOverlay(curveDict[key], kdims = ['Run', 'Channel'])
            else:
                self.data[key]['curve'] = {dim:hv.NdOverlay(curveDict[key], kdims = ['Run', 'Channel'])}

# Basic case for 1D vars
#             frequencies, edges = np.histogram(d0, bins)


#             if 'curve' in self.data[key].keys():
#                 self.data[key]['curve'][dim] = hv.Curve((edges, frequencies), dim, 'count')
#             else:
#                 self.data[key]['curve'] = {dim:hv.Curve((edges, frequencies), dim, 'count')}

        if self.verbose['main']:
            print(f"Set self.data[key]['curve'] for dim={dim}.")


    def histOverlay(self, dim = None, **kwargs):
        """
        Plot overlay of histograms over datasets (runs)

        See http://holoviews.org/user_guide/Building_Composite_Objects.html

        """

        # Loop over all datasets
        overlayDict = {}
        for key in self.runs['proc']:
            if 'curve' not in self.data[key].keys():
                self.hist(dim=dim, keys=[key], **kwargs)

            overlayDict[key] = self.data[key]['curve'][dim]

        # Set outputs - NdOverlay, holomap and holomap layout.
        self.ndoverlay = hv.NdOverlay(overlayDict, kdims='Run') # .relabel(group='Runs',label=dim, depth=1)
        self.hmap = hv.HoloMap(self.ndoverlay)  # This works in testing, but with data seems to keep multiple datasets in plot? Update issue?
                                                # See http://holoviews.org/reference/containers/bokeh/NdOverlay.html
                                                # TESTS: https://pswww.slac.stanford.edu/jupyterhub/user/phockett/notebooks/dev/classDemo_191120_dev_bk1.ipynb
        self.layout = hv.HoloMap(self.ndoverlay).layout().cols(1)  #.opts(height=300).layout().cols(1)  # opts here overrides defaults? Not sure why, just removed for now.

        if self.verbose['main']:
            print(f'Set self.ndoverlay, self.hmap and self.layout for dim={dim}.')


    def hist2d(self, dim = None, ref = None, filterOptions = None):
        """
        Basic wrapper for hv.HexTiles (http://holoviews.org/reference/elements/bokeh/HexTiles.html#elements-bokeh-gallery-hextiles) for 2D histograms.

        Pass dim = [dim0, dim1] for dims to map.

        Currently assumes that dim0 is 1D, and dim1 is 1D or 2D.

        Optionally set ref dimension and filterOptions for raw data.
        """

#         hexOpts = opts.HexTiles(width=600, height=400, tools=['hover'], colorbar=True)
        hexDict = {}

        if filterOptions is not None:
            self.filterData(filterOptions = filterOptions)

        # Loop over all datasets
        for key in self.runs['proc']:
            # Initially assume mask can be used directly, but set to all True if not passed
            # Will likely want more flexibility here later
#             if mask is None:
#             mask = np.ones_like(self.data[key]['raw'][dim[0]]).astype(bool)

            # Check mask exists, set if not
            if 'mask' not in self.data[key].keys():
                self.filterData(keys=[key])

            d0 = np.array(self.data[key]['raw'][dim[0]])[self.data[key]['mask']]
            d1 = np.array(self.data[key]['raw'][dim[1]])[self.data[key]['mask']]

            # Check cols
            d1Range = 1
            kdims = [dim[1]]
            if len(d1.shape)>1:
                d1Range = d1.shape[1]
                kdims.append(f'{dim[1]} col')

#             hexList[key] = {i:hv.HexTiles((d0, d1), dim).opts(hexOpts) for i in np.arange(0,d1Range-1)}
            # hexDict[key] = {i:hv.HexTiles((d0, d1[:,i]), dim) for i in np.arange(0,d1Range)}
            hexDict[key] = {(key, i):hv.HexTiles((d0, d1[:,i]), dim) for i in np.arange(0,d1Range)}

            self.data[key]['hist2d'] = hexDict[key]

            # Set outputs - NdOverlay, holomap and holomap layout.
            # self.ndoverlay = hv.NdOverlay(hexDict, kdims = ['Run', 'Channel']) # .relabel(group='Runs',label=dim, depth=1)
            self.data[key]['hist2d']['hmap'] = hv.HoloMap(hexDict[key], kdims = ['Run', 'Channel'])  # This works in testing, but with data seems to keep multiple datasets in plot? Update issue?
                                                    # See http://holoviews.org/reference/containers/bokeh/NdOverlay.html
                                                    # TESTS: https://pswww.slac.stanford.edu/jupyterhub/user/phockett/notebooks/dev/classDemo_191120_dev_bk1.ipynb
            # This is currently throwing an error... no idea why!
            # self.data[key]['hist2d']['layout'] = hv.HoloMap(hexDict[key], kdims = ['Run', 'Channel']).layout().cols(1)  #.opts(height=300).layout().cols(1)  # opts here overrides defaults? Not sure why, just removed for now.

        if self.verbose['main']:
            print(f"Set self.data[key]['hist2d'] for dim={dim}.")


#         return hv.HoloMap(hexList, kdims = kdims)  # This doesn't work for nested case, but should be a work-around...?

    def image2d(self, dim = None, ref = None, filterOptions = None, bins = (np.arange(0, 1048.1, 1)-0.5,)*2):
        """
        Basic wrapper for hv.Image for 2D histograms as images (see also hist2d).

        Pass dim = [dim0, dim1] for dims to map.

        Currently assumes that dim0 and dim1 are 1D.

        Optionally set ref dimension and filterOptions for raw data.

        Based on Elio's routine in "elec_pbasex.ipynb".

        """

#         hexOpts = opts.HexTiles(width=600, height=400, tools=['hover'], colorbar=True)
        imDict = {}

        if filterOptions is not None:
            self.filterData(filterOptions = filterOptions)

        # Loop over all datasets
        for key in self.runs['proc']:
            # Initially assume mask can be used directly, but set to all True if not passed
            # Will likely want more flexibility here later
#             if mask is None:
#             mask = np.ones_like(self.data[key]['raw'][dim[0]]).astype(bool)

            # Check mask exists, set if not
            if 'mask' not in self.data[key].keys():
                self.filterData(keys=[key])

            # Note flatten or np.concatenate here to set to 1D, not sure if function matters as long as ordering consistent?
            # Also, 1D mask selection will automatically flatten? This might be an issue for keeping track of channels?
            # Should use numpy masked array...?
            d0 = np.array(self.data[key]['raw'][dim[0]])[self.data[key]['mask']].flatten()
            d1 = np.array(self.data[key]['raw'][dim[1]])[self.data[key]['mask']].flatten()


            # Quick test with single image - should convert to multiple here?
            imDict[key] = hv.Image((np.histogram2d(d0,d1, bins = bins)[0]), dim) # for i in np.arange(0,d1Range)}

            self.data[key]['img'] = imDict[key]

#         return hv.HoloMap(hexList, kdims = kdims)  # This doesn't work for nested case, but should be a work-around...?
