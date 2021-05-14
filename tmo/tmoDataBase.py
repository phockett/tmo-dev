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

# Basic timing decorator - IN PROGRESS
# See, e.g., https://realpython.com/primer-on-python-decorators/#timing-functions
# NOTE: currently assumes self.verbose exists, and no return values.
# import functools, time
# def timer(func):
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         start = time.perf_counter()
#         func(*args, **kwargs)
#         duration = time.perf_counter() - start
#
#         if self.verbose['main']:
#             print(f'Ran {func.__name__!r} in {duration:.4f} secs')
#
#


# Set some default plot options
def setPlotDefaults(fSize = [800,400], imgSize = 500):
    """Basic plot defaults"""
    opts.defaults(opts.Curve(width=fSize[0], height=fSize[1], tools=['hover'], show_grid=True),
                  opts.Image(width=imgSize, frame_width=imgSize, aspect='square', tools=['hover'], colorbar=True),   # Force square format for images (suitable for VMI)
                  opts.HeatMap(width=imgSize, frame_width=imgSize, aspect='square', tools=['hover'], colorbar=True),
                  opts.HexTiles(width=fSize[0], height=fSize[1], tools=['hover'], colorbar=True))


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



class tmoDataBase():
    """
    Very crude data handling & plotting class for TMO run 18 preprocessed data.

    Reads h5py files & plots various properties & correlations using Holoviews.

    13/05/21 v0.0.1-sacla, updates + mods for SACLA data.

    18/11/20 v0.0.1

    Paul Hockett

    https://github.com/phockett/tmo-dev


    #TODO: histND & pyvista
    Pull run info from elog
    Support for raw data


    """

    # self.__version__ = __version__
    __notebook__ = isnotebook()

    # Setup optional local imports
    # from .sacla import sacla  # This method DOESN"T work for self.fun(), should write as full function calls, or proper subclass, instead?
                                # Could also monkeypatch at __init__, but probably not a good idea.
    from .sacla.sacla import setup, calibration  # Bind directly at class instantiation. This works.


    def __init__(self, fileBase = None, ext = 'h5', runList = None, fileSchema='aq{N:03.0f}',
                 fileList=None, verbose = 1, accelerator = 'sacla'):
        """
        Files to read defined by:

        - fileBase: base dir.
        - ext: file type/extension, e.g. 'h5'
        - runList: list of runs to read, e.g. [96,97,105]
        - fileSchema: format string to generate file names using runList items
            - For SLAC TMO, e.g. 'run{N}_preproc_elecv2', where the tail may change.
            - For SACLA, 'aq{N:03.0f}'
            - NOTE this has changed since original version. Dir scan + pattern matching should also be implemented here.

        """


        if verbose:
            print(f'Running for file pattern: {fileSchema}')

        # Set file properties
        self.runs = {'fileBase':Path(fileBase),
                     'ext':ext,
                     # 'prefix':prefix,
                     'fileList':fileList,
                     'runList':runList,
                     # 'files': {N:Path(fileBase, f'run{N}{fileSchema}.{ext}') for N in runList}  # For SLAC TMO runs
                     # 'files': {N:Path(fileBase, f'aq{N:03.0f}{fileSchema}.{ext}') for N in runList} # For SACLA runs
                     'files': {N:Path(fileBase, fileSchema.format(N=N) + f'.{ext}') for N in runList}  # With generic format string set
                     # 'files': self.getFiles(ext=ext, runList=runList, fileSchema=fileSchema, fileList=fileList)
                     }

        # Set default data dicts
        self.dTypes = ['raw','metrics']

        # Set for main functions and subfunctions
        self.verbose = {'main':verbose, 'sub':verbose-1}

        if self.verbose['sub'] < 0:
            self.verbose['sub'] = 0

        # Additional setup for SACLA data
        # Assume from filenames, or via additional flag.
        # TODO: should just generalise for all cases, force by accelerator name.
        if fileSchema.startswith('aq') or (accelerator=='sacla'):
            self.accelerator = 'sacla'
            print(f"Setting additional params for {self.accelerator}")

            # self.setup = self.sacla.setup   # This doesn't work for direct self.method(self) binds - but could monkeypatch here.
            # self.calibration = self.sacla.calibration

            self.setup()

#**** IO
    # TO DO: write proper file IO here with dir scan!
    # def getFiles(self, ext='h5', runList = None, fileSchema=None, fileList=None):
    #
    #     if
    #         {N:Path(fileBase, f'run{N}{fileSchema}.{ext}') for N in runList}
    #
    #  Get files by type from dir
    #  fileList = [os.path.join(fileBase, f) for f in os.listdir(fileBase) if f.endswith(fType)]
    #
    #  Subselect from list by number - rather ugly, should use sets or re?
    #  runList = [95, 100, 46]
    #
    #  for N in runList:
    #     [print(item) for item in fileList if item.endswith('aq{0:03.0f}.{1}'.format(N,ext))]

    def readFiles(self, keyDim = None, runMetrics = True):
        """
        Read (preprocessed) data from h5 files.

        Parameters
        ----------

        keyDim : str, optional, default = None
            If set, use this dim to check if data is valid.
            Defaults to 'energies' or 'gmd_energy'.
            Probably want to implement a more careful/thorough method here in future!

        NOTE: this currently leaves hdf5 filestreams open for reading later, which might be bad practice. Not sure if there is a better way for handling multiple large files and data subselection?

        """

        # Set data per run
        self.data = {}

        # Set h5 file object - should just be file pointers (?)
        # Yep, OK, returns items <HDF5 file "run93_preproc_elecv2.h5" (mode r)>
        # No mem usage (checked with %memit, https://timothymonteath.com/articles/monitoring_memory_usage/)
        # 02/12/20: added basic try/except here to skip missing files.
        for key in self.runs['files'].keys():
            try:
                # print(f"* Trying file {self.runs['files'][key]}")
                # For SACLA case raw data has some different dims, so push it elsewhere as a quick fix
                # Should remove hardcoded 'raw' dType refs for a proper fix.
                # if self.accelerator is 'scala':
                #     self.data[key] = {'scRaw':File(self.runs['files'][key],'r')}
                # else:
                self.data[key] = {'raw':File(self.runs['files'][key],'r')}
                # print('OK')
            except OSError:
                self.data[key] = None


        # Additionally pull some useful metrics
        # Does h5py object already report some of this directly? Not sure.
        self.runs['proc'] = []
        self.runs['invalid'] = []
        for key in self.data.keys():
            print(f"* Trying file {self.runs['files'][key]}")

            if self.data[key] is not None:
                self.data[key]['items'] = self.data[key]['raw'].keys()
                self.data[key]['dims'] = {item:self.data[key]['raw'][item].shape for item in self.data[key]['raw'].keys()}

                # Very basic IO check, if energies is missing dataset may be problematic?
                # TODO: use unified dict or sets to check consistent dims over all datasets.
                # 25/11/20: hacked in additional passed dim check, UGLY.
                if keyDim is None:
                    if ('energies' not in self.data[key]['dims']) and ('gmd_energy' not in self.data[key]['dims']) and ('fel_status' not in self.data[key]['dims']):
                        print(f'*** WARNING: key {key} missing default keyDims, will be skipped.')
                        self.runs['invalid'].append(key)
                    else:
                        self.runs['proc'].append(key)
                        print('OK')
                else:
                    if keyDim not in self.data[key]['dims']:
                        print(f'*** WARNING: key {key} missing {keyDim} data, will be skipped.')
                        self.runs['invalid'].append(key)
                    else:
                        self.runs['proc'].append(key)
                        print('OK')
            else:
                print(f'*** WARNING: key {key} file missing, will be skipped.')
                self.runs['invalid'].append(key)

        # Run metrics
        if runMetrics:
            try:
                self.runMetrics()
            except KeyError as err:
                print(f'*** WARNING: {err}')
                print('Skipped runMetrics() for dataset.')


        if self.verbose['main']:
            print(f"Read {len(self.data)} files.")
            print(f"Good datasets: {self.runs['proc']}")
            print(f"Invalid datasets: {self.runs['invalid']}")

        # For SACLA case raw data has some different dims, so push it elsewhere as a quick fix
        # Should remove hardcoded 'raw' dType refs for a proper fix.
        if self.accelerator is 'sacla':
            for key in self.runs['proc']:
                self.data[key]['scRaw'] = self.data[key].pop('raw')

            #     # In testing this seems OK, but may need explicit copy here?
            #     self.data[key]['scRaw'] = self.data[key]['raw']
            #     del self.data[key]['raw']



#**** ANALYSIS
    # Set some additional dataset parameters
    def runMetrics(self, keys = None, eROI = [20, 1010]):
        """
        Pull metrics from data.

        Currently sets:
        Ions:
            - iShot: ion data/events per shot from self.data[key]['raw']['ktofIpk']
            - iTot: total ion counts

        Electrons:
            - eShot: electron data/events per shot for [x,y] coords, from self.data[key]['raw']['xc'] and ['yc']
            - eROI: as eShot but filter on ROI, default = [20, 1010] pixels (covers detector area in LW06, and omits edge counts)
            - eTot: total electron counts

        Note: "no hit" value is assumed to be -9999.

        Parameters
        ----------
        eROI : optional, default = [20,1010]
            Define upper and lower limits (pixels) used for assessing electron hit data counts. Currently set for square ROI.
            Note default ROI set for detector area (as of tmolw0618 data).

        Notes
        -----
        - 28/02/21  Added basic dim checks, and force (xc,yc) to int16 to save memory. May also want to implement for ion data.
                    TODO: check use of np.asarray typing vs. hdf5.astype('int') - latter should be preferred here?
        """

        # Default to all datasets
        if keys is None:
            keys = self.runs['proc']

        #     if not hasattr(self, 'metrics'):
        #         self.metrics = {}

        for key in keys:
        #         self.metrics[key] = {}

            metrics = {}

            if 'ktofIpk' in self.data[key]['dims']:
                metrics['iShot'] = (~(np.asarray(self.data[key]['raw']['ktofIpk']) == -9999)).sum(1)
                metrics['iTot'] = metrics['iShot'].sum()
            else:
                metrics['iShot'] = None
                metrics['iTot'] = None

            # Electron data checks - may want to set processed data to var?
            # metrics['eCount'] = np.c_[(~(np.asarray(self.data[key]['raw']['xc']) == -9999)).sum(1), (~(np.asarray(self.data[key]['raw']['yc']) == -9999)).sum(1)]

            dtype = 'int16'
            if 'xc' in self.data[key]['dims']:
                xc = np.array(self.data[key]['raw']['xc'], dtype = dtype)
                xInd = (xc>-9999) #eROI[0]) & (xc<eROI[1])

                yc = np.array(self.data[key]['raw']['yc'], dtype = dtype)
                yInd = (yc>-9999)  #(yc>eROI[0]) & (yc<eROI[1])

        #         metrics['eShot'] = np.c_[(~(xc == -9999)).sum(1), (~(yc == -9999)).sum(1)]
        #         metrics['eTot'] = np.c_[xc[xc>-9999].size, yc[yc>-9999].size]  # Totals

                metrics['eShotXY'] = np.c_[xInd.sum(1), yInd.sum(1)]
                metrics['eShot'] = xInd.sum(1)  # Set as single channel
                # metrics['eShot'] = xInd.sum(1) - yInd.sum(1)  # Set 1D total hits case as difference?
                # metrics['eShot'] = (xInd.sum(1) + yInd.sum(1)) - (xInd.sum(1) - yInd.sum(1))  # Sum minus difference.
                metrics['eMean'] = np.nanmean(xc)
                metrics['eROI'] = np.c_[((xc>eROI[0]) & (xc<eROI[1])).sum(1), ((yc>eROI[0]) & (yc<eROI[1])).sum(1)]

                metrics['eTot'] = np.c_[metrics['eShotXY'].sum(0), metrics['eROI'].sum(0)]

            else:
                metrics['eShotXY'] = None
                metrics['eShot'] = None
                metrics['eROI'] = None
                metrics['eTot'] = None

                #         self.metrics[key] = metrics  # Set as new dict
            self.data[key]['metrics'] = metrics  # Add to existing data dict

        # def setFilterRange(self, filterOptions = {}):
        #     """
        #     Set filters for a range of values/bins for a given data type.
        #     """
        #
        #

    def setFilter(self, filterOptions = {}, reset = False):
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

        TODO:
        - Ranges:

        e.g. Set for time scan...
        tRange = np.arange(test.min(), test.max(), 2e-3)
        laserFilter = {n:{'epics_las_fs14_target_time':[tRange[n], tRange[n+1]]} for n in range(0,tRange.size-1)}

        """
        # Reset filter?
        if reset:
            self.filters = {'Default':{}}  # Set an empty value here, otherwise filtersetting will fail later for global settings!

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



    def filterData(self, filterOptions = {}, keys = None, dim = 'energies', dTypes = ['raw','metrics']):
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

        dTypes : list, optional, default = ['raw','metrics']
            Data dicts to use for filtering.
            TODO: move this elsewhere!

        TODO:

        - More flexibility.
        - Filter functions, e.g. saturated electron detector shots? ('xc' > 0).sum(axis = 1) <=1000 in this case I think.

        07/12/20: added support for "metrics" data.

        """

        # Default to all datasets
        if keys is None:
            keys = self.runs['proc']

        for key in keys:

            # Check key/template dim exists
            # TODO: make this better.
            if dim not in self.data[key]['dims']:
                # Missing dim, try defaults.
                if 'gmd_energy' in self.data[key]['dims']:
                    dim = 'gmd_energy'
                elif 'energies' in self.data[key]['dims']:
                    dim = 'energies'
                else:
                    print(f"Missing dims for filtering: {dim}, and defaults not present.")

                    # if self.verbose['main']:
                    #     print(f"Reset )

            # Set full mask = true, use passed dim ('energies' as default) to define datasize event dim
            mask = np.ones_like(self.data[key]['raw'][dim]).astype(bool)
            if len(mask.shape)>1:
                mask = mask[:,0]

            for item in filterOptions.keys():

                # For 'raw' data types
                # if item in self.data[key]['raw'].keys():
                #     testData = np.array(self.data[key]['raw'][item])
                #
                # # For metrics (derived data)
                # elif item in self.metrics[key].keys():
                #     testData = self.metrics[key][item]

                # Version with dict testing
                # for dType in dTypes:
                #     if item in self.data[key][dType].keys():
                #         dataDict = dType
                #         testData = np.array(self.data[key][dataDict][item])  # np.array wrapper for 'raw' case
                #     else:
                #         dataDict = None
                #         testData = None

                testData = self.getDataDict(item, key, returnType = 'data')  # Method version of the above

                # if dataDict is not None:
                #     testData = np.array(self.data[key][dataDict][item])  # np.array wrapper for 'raw' case
                #
                # else:
                #     testData = None

                    # 27/11/20 = quick hack for multidim col selection in v4 preprocessed data
                    # Pass as dict with 'col', 'value' parameters
                    # ACTUALY, multidim case below was OK, just had a bug!
                    # if type(filterOptions[item]) is dict:
                    #     col = filterOptions[item]['col']
                    #     val = filterOptions[item]['val']
                    #
                    #     testData = testData[:,col]
                    #

                if testData is not None:
                    # Match single items
                    if type(filterOptions[item]) is not list:
                        filterOptions[item] = [filterOptions[item]]  # UGLY - wrap to list.

                    if len(filterOptions[item])==1:
                        mask *= (testData == filterOptions[item])

                    if len(filterOptions[item])==2:
                        mask *= (testData >= filterOptions[item][0]) & (testData <= filterOptions[item][1])

                    # Case for multidim testData
                    if len(filterOptions[item])==3:
                        mask *= (testData[:,filterOptions[item][2]] >= filterOptions[item][0]) & (testData[:,filterOptions[item][2]] <= filterOptions[item][1])

                else:
                    if self.verbose['main'] and (item is not 'desc'):   # Ignore 'desc' entries.
                        print(f"Can't filter on data type {item}, not found in dataset {key}")

                # TODO: add other filter types here.

            self.data[key]['mask'] = mask  # For single filter this is OK, for multiples see vmi version.


    def getDataDict(self, dim, key = None, dTypes = None, returnType = 'dType'):
        """
        Return specific dataset from various dictionaries by dimension name.

        dim : string
            Dimension (data) to find/check.

        key : string, int, optional, default = None
            Run key into main data structure.
            If None, use the first run in self.runs['proc'].

        dTypes : str, list, optional, default = self.dTypes
            Data dicts to check, defaults to global settings.

        returnType : str, optional, default = 'dType'
            - 'dType' return data type to use as index.
            - 'data' return data array.
            - 'lims' return min & max values only.
            - 'unique' return list of unique values.

        08/12/20: first attempt, to replace repeated code in various base functions, and allow for multiple types (e.g. 'raw', 'metrics' etc.)

        TODO: may also want to add datatype to array conversion routine, since this will otherwise default to float64 and can be memory hungy.
        May also want to add chunking here too.

        TO FIX: dTypes checking buggy, for multiple matched dTypes only returns last matching item.

        """

        # Default to first dataset
        if key is None:
            key = self.runs['proc'][0]

        if dTypes is None:
            dTypes = self.dTypes

        dataDict = None # Set default
        for dType in dTypes:
            if (dType in self.data[key].keys()):  # && (dim in self.data[key][dType].keys()):  # Short circuit here in case dType doesn't exist in dict.
                if (dim in self.data[key][dType].keys()):
                    dataDict = dType
                # else:
                #     dataDict = None

        if returnType is 'dType':
            return dataDict

        elif returnType is 'data':
            if dataDict is not None:
                if dataDict is 'raw':
                    return np.array(self.data[key][dataDict][dim])  # Explicit conversion to np.array - may just wrap everything this way?
                else:
                    return self.data[key][dataDict][dim]
            else:
                return dataDict  # Currently set to None if dim not found, may change later.

        # def checkDims(testArray):
        #     """Check if array is 2D"""

#**** PLOTTING STUFF

    def showPlot(self, dim, run, pType='curve'):
        """Render (show) specified plot from pre-set plot dictionaries (very basic!)."""

        try:
            if self.__notebook__:
                display(self.data[run][pType][dim])  # If notebook, use display to push plot.
            else:
                return self.data[run][pType][dim]  # Otherwise return hv object.

        except:
            print(f'Plot [{run}][{pType}][{dim}] not set.')


    def curvePlot(self, dim, filterOptions = None, keys = None): # , dTypes = ['raw','metrics']):
        """Basic wrapper for hv.Curve, currently assumes a 1D dataset, or will skip plot.

        07/12/20: quick mod to support metrics datatype.
        """

        # Default to first dataset
        if keys is None:
            keys = self.runs['proc'][0]

        # dTypes = ['raw','metrics']  # Quick hack for multiple data dics - should move to function!

        for key in keys:
            # Check mask exists, set if not
            if 'mask' not in self.data[key].keys():
                self.filterData(keys=[key])

            # for dType in dTypes:
            #     if dim in self.data[key][dType].keys():
            #         dataDict = dType
            #     else:
            #         dataDict = None
            #     # elif dim in self.metrics[key].keys():
            #     #     d0 = np.array(self.metrics[key][dim[0]])[self.data[key]['mask']]
            #
            # if dataDict is not None:
            #     d0 = np.array(self.data[key][dataDict][dim[0]])[self.data[key]['mask']]
            # else:
            #     pass  # Just skip error cases for now

            d0 = self.getDataDict(dim, key, returnType = 'data')[self.data[key]['mask']]

            try:
                if self.__notebook__:
                    display(hv.Curve(d0))  # If notebook, use display to push plot.
                else:
                    # return hv.Curve(d0)  # Otherwise return hv object.
                    pass
            except:
                pass


    def hist(self, dim, bins = 'auto', filterOptions = None, keys = None,
                weights = None, normBin = False):
        """
        Construct 1D histrograms using np.histogram.

        28/02/21 Added option and handling for weights, pass as array or dim.
        NOTE: this currently assumes 1D weights dim. For auto bins, these will be determined from dim values.

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

            # d0 = np.array(self.data[key]['raw'][dim])[self.data[key]['mask']]
            d0 = self.getDataDict(dim, key, returnType = 'data')[self.data[key]['mask']]

            # Set weights if passed as dim
            if isinstance(weights, str):
                weightVals = self.getDataDict(weights, key, returnType = 'data')[self.data[key]['mask']]
            else:
                weightVals = weights

            # Check cols, otherwise will be flatterned by np.histogram
            # TODO: move to separate function, and use .ndim (OK for h5 and np arrays)
            d0Range = 1
            if len(d0.shape)>1:
                d0Range = d0.shape[1]
                # kdims.append(f'{dim[1]} col')
            else:
                d0 = d0[:,np.newaxis]

            # Numpy hist bins options - use these for str checks vs. dims
            npBinOpts = ['auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', 'sqrt']

            # For weighted case, 'auto' bin is not supported, so define bins first
            if (weights is not None) and (bins == 'auto'):
                freqBins, binsW = np.histogram(d0[:,0], bins)  # TODO: fix dims here! Assuming 1st dim only.
            elif isinstance(bins, str) and not (bins in npBinOpts):
                binsW = self.getDataDict(bins, key, returnType = 'data')
            else:
                binsW = bins
                freqBins = np.ones(d0[:,0].size)

                # This doesn't work due to ordering of np.histogram returns! Doh!
            # curveDict[key] = {f'{key},{i}':hv.Curve((np.histogram(d0[:,i], bins)), dim, 'count') for i in np.arange(0,d0Range)}
            curveDict[key] = {}
            for i in np.arange(0,d0Range):
                freq, edges = np.histogram(d0[:,i], bins = binsW, weights = weightVals)

                # Normalise by bin counts (shots)
                if normBin:
                    freq = freq/freqBins

                # freq, edges = np.histogram(d0[:,i], bins)
                # curveDict[key][i] = hv.Curve((edges, freq), dim, 'count')
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


    def histOverlay(self, dim = None, pType='curve', **kwargs):
        """
        Plot overlay of histograms over datasets (runs)

        See http://holoviews.org/user_guide/Building_Composite_Objects.html

        """

        # Loop over all datasets
        overlayDict = {}
        for key in self.runs['proc']:
            if pType not in self.data[key].keys():
                if pType == 'curve':
                    self.hist(dim=dim, keys=[key], **kwargs)
                else:
                    print(f'*** Missing hv plot {pType} for key={key}')
                # if pType == 'cc':
                #     self.corrRun()


            overlayDict[key] = self.data[key][pType][dim]

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

        # hexOpts = opts.HexTiles(width=600, height=400, tools=['hover'], colorbar=True)
        hexDict = {}

        if filterOptions is not None:
            self.filterData(filterOptions = filterOptions)

        # Loop over all datasets
        for key in self.runs['proc']:
            # Initially assume mask can be used directly, but set to all True if not passed
            # Will likely want more flexibility here later
            # if mask is None:
            # mask = np.ones_like(self.data[key]['raw'][dim[0]]).astype(bool)

            # Check mask exists, set if not
            # NOW set to always run, otherwise self.filter settings may be missed on update
            # ACTUALLY removed again, but should set an update flag on filters?
            if 'mask' not in self.data[key].keys():
                self.filterData(keys=[key])

            # d0 = np.array(self.data[key]['raw'][dim[0]])[self.data[key]['mask']]
            # d1 = np.array(self.data[key]['raw'][dim[1]])[self.data[key]['mask']]

            d0 = self.getDataDict(dim[0], key, returnType = 'data')[self.data[key]['mask']]
            d1 = self.getDataDict(dim[1], key, returnType = 'data')[self.data[key]['mask']]

            # Check cols
            d1Range = 1
            kdims = [dim[1]]
            if len(d1.shape)>1:
                d1Range = d1.shape[1]
                kdims.append(f'{dim[1]} col')

                hexDict[key] = {(key, i):hv.HexTiles((d0, d1[:,i]), dim) for i in np.arange(0,d1Range)}

            else:
                # For 1D data case
                hexDict[key] = {(key, 0):hv.HexTiles((d0, d1), dim)}

            # hexList[key] = {i:hv.HexTiles((d0, d1), dim).opts(hexOpts) for i in np.arange(0,d1Range-1)}
            # hexDict[key] = {i:hv.HexTiles((d0, d1[:,i]), dim) for i in np.arange(0,d1Range)}


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


        # return hv.HoloMap(hexList, kdims = kdims)  # This doesn't work for nested case, but should be a work-around...?

    def image2d(self, dim = None, ref = None, filterOptions = None, bins = (np.arange(0, 1048.1, 1)-0.5,)*2):
        """
        Basic wrapper for hv.Image for 2D histograms as images (see also hist2d).

        Pass dim = [dim0, dim1] for dims to map.

        Currently assumes that dim0 and dim1 are 1D.

        Optionally set ref dimension and filterOptions for raw data.

        Based on Elio's routine in "elec_pbasex.ipynb".

        """

        # hexOpts = opts.HexTiles(width=600, height=400, tools=['hover'], colorbar=True)
        imDict = {}

        if filterOptions is not None:
            self.filterData(filterOptions = filterOptions)

        # Loop over all datasets
        for key in self.runs['proc']:
            # Initially assume mask can be used directly, but set to all True if not passed
            # Will likely want more flexibility here later
            # if mask is None:
            # mask = np.ones_like(self.data[key]['raw'][dim[0]]).astype(bool)

            # Check mask exists, set if not
            if 'mask' not in self.data[key].keys():
                self.filterData(keys=[key])

            # Note flatten or np.concatenate here to set to 1D, not sure if function matters as long as ordering consistent?
            # Also, 1D mask selection will automatically flatten? This might be an issue for keeping track of channels?
            # Should use numpy masked array...?
            # d0 = np.array(self.data[key]['raw'][dim[0]])[self.data[key]['mask']].flatten()
            # d1 = np.array(self.data[key]['raw'][dim[1]])[self.data[key]['mask']].flatten()
            d0 = self.getDataDict(dim[0], key, returnType = 'data')[self.data[key]['mask']].flatten()
            d1 = self.getDataDict(dim[1], key, returnType = 'data')[self.data[key]['mask']].flatten()


            # Quick test with single image - should convert to multiple here?
            imDict[key] = hv.Image((np.histogram2d(d0,d1, bins = bins)[0]), dim) # for i in np.arange(0,d1Range)}

            self.data[key]['img'] = imDict[key]

        # return hv.HoloMap(hexList, kdims = kdims)  # This doesn't work for nested case, but should be a work-around...?
