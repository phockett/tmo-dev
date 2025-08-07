"""
Class and method dev for peak finding and fitting.

23/07/25 v1 Implements basic peak find routine (using Scipy), and fitting with lmfit (wrapped by xrlmfit).

Paul Hockett

https://github.com/phockett/tmo-dev

"""

from scipy.signal import find_peaks
import scipy.signal

import numpy as np
import xarray as xr

import holoviews as hv
from holoviews import opts
import hvplot.xarray

import lmfit

# Logging
from loguru import logger

# xrlmfit - optional as may require xr update
# If flag is false basic fitting with wrapper implemented
try:
    import xarray_lmfit
    xrlmfitFlag = True
    logger.info("Using `xarray_lmfit` wrapper.")

except ImportError: 
    xrlmfitFlag = False
    logger.info("Missing `xarray_lmfit`, using bare lmfit routines.")
    
# xrlmfitFlag = False   # Set False to test bare functions.
    
    
class peakFit:
    """
    Class for peak finding & fitting, aims to automate as much as possible.
    
    TODO: more options, logging.
    
    """
    
    def __init__(self, data, peaks=None, quickFit=True, **kwargs):
        
        
        self.data = data
        self.peaks = peaks
        
        if not self.data.name:
            self.data.name = 'Intensity'
            
        if peaks is None:            
            self.findPeaks(**kwargs)
            
        if quickFit:
            self.quickFit(**kwargs)
            
            
    def quickFit(self, **kwargs):
        """
        Run methods for peak find and fit with minimal user input.
        
        **kwargs are passed to self.createModel if control required.
        
        """
        
        logger.info("Running quick fit...")
        
        self.createModel(**kwargs)
        self.fit()
        
    
    def findPeaks(self, data = None, thres=None, thresPC=0.5, prominence = 0.1, width = 5, 
                  plotPeaks = True, **kwargs):
        """
        Use scipy.signal.find_peaks to get peak positions.
        
        thres, thresPC: set threshold value for peaks, or set as %age of maximum (default 50%)
        
        prominence, width: params passed to find_peaks.
        
        TODO: pass optional kwargs, currently just set defaults.
        
        """
        
        if data is None:
            data = self.data
        
        if thres is None:
            thres = thresPC*self.data.max().data
        
        logger.info(f"Peak finding for thres={thres}...")
                    
        # peaks, props = find_peaks(testData, height=thres, prominence=prominence*thres)  # For NIST Gauss2 thres=80, 10% prominence seems good. 
                                                                                        # Way to logically set defaults here? Something like 70% of max value works OK.

        peaks, props = find_peaks(data, height=thres, prominence=prominence*thres, width=5)  # Even more robust with a reasonable min width set, gets main features even at 0.1*max

        self.peaks = peaks
        self.peakProps = props
        
        logger.info(f"Found peaks at {peaks}, see `self.peakProps` for more details.")
        
        # Quick plot return - should use checkNotebook and display as per older codes.
        if plotPeaks:
            try:
                plotOut = self.data.hvplot(title='Peak finder') * hv.VLines(peaks).opts(opts.VLines(line_dash='dashed', alpha=0.5))   # VLines if present!
            except:
                plotOut = self.data.hvplot(title='Peak finder') * hv.Points((peaks, thres*np.ones(len(peaks)))).opts(marker = 'x', size=10, color='r')
            
            self.plot = plotOut
            
            display(plotOut)
            
            
            
    def createModel(self, peaks = None, model = 'GaussianModel', baseline = None,
                       dim = 'x', guessParams = True, **kwargs):
        """
        Create model function using lmfit models.
        
        Create:
        - One peak of type "model" per feature.
        - Add baseline function if set.
        
        Models: see https://lmfit.github.io/lmfit-py/builtin_models.html
        Most likely to need "GaussianModel", "LorentzianModel" or "VoigtModel" for peaks.
        For background "ExponentialModel", "ConstantModel" or "LinearModel".
        
        Alternatively, pass an lmfit model directly to use instead of auto-setting.
        
        NOTE: currently written assuming xr-lmfit, which handles dims...  
        BUT: "NotImplementedError: guess() not implemented for CompositeModel", so need to add guess per component and handle data.
        UPDATE: Setup just uses base lmfit class, only use wrapper for fitting? Hmmm.
        UPDATE 2: Can set `guess=True` for xlm.modelfit param guess, inc. composite case - runs basically as configured here (per component).
        UPDATE 3: Added **kwargs for passing specific params to self.modelEval if desired.
        
        """
        
        if peaks is None:
            peaks = self.peaks
        
        if isinstance(model, str):
            logger.info(f"Creating peaks with function {model}...")
            
            modelDict = {}
            
            for n, x in enumerate(peaks):
                modelDict[n] = eval(f"lmfit.models.{model}(prefix='p{n}_')")
                
                # Try skipping params to start with, should guess automatically...?
                # BUT doesn't seem to work well in test data?
                if guessParams:
                    if n==0:
                        pars = modelDict[n].guess(self.data.values, x=self.data['x'].values)  # TODO: set dims here.
                    else:
                        pars.update(modelDict[n].guess(self.data.values, x=self.data['x'].values))

                # Create composite model object
                if n==0:
                    modelSum = modelDict[n]
                else:
                    modelSum += modelDict[n]
                    
            if baseline is not None:
                modelDict['base'] = eval(f"lmfit.models.{baseline}(prefix='base_')")
                modelSum += modelDict['base']
                
                if guessParams:
                    pars.update(modelDict['base'].guess(self.data.values, x=self.data['x']))
        
#         self.model = sum(modelDict.values())
        self.modelDict = modelDict
        self.model = modelSum
            
        if guessParams:
            self.params = pars
        else:
            pars = None
            self.params = None
        
#         # need to reformat for XR...
#         initialFit = modelSum.eval(pars, x=self.data[dim])
# #         compDictReformat = {k:(dim,v) for k,v in compDict.items()}
#         initialFit = xr.DataArray(initialFit, coords={'x':self.data[dim]})
        
        self.modelEval(dim=dim, **kwargs)
        
        self.initialFit = self.current.copy()
        self.initialFit.name = 'init'
    
        logger.info(f"Created model, self.model={modelSum}.")
        
        if guessParams:
            logger.info(f"Guessed params, set to self.params.")
            
#         self.init = 

#             model = lmfit.models.GaussianModel() + lmfit.models.LinearModel()


    def modelEval(self, dim='x', **kwargs):
        """
        Evaluate lmfit model function using currently set params.
        
        Or, if kwargs is passed, use these instead.
        
        NOTE: some inconsistencies here? self.params always works, but custom params sometimes fails for custom functions? Maybe to do with wrappers... TBC...
        UPDATE: now set to use passed params OR kwargs, but only valid if all set. Can also just pass ones to change, per https://lmfit.github.io/lmfit-py/model.html#model-class-methods.
        In current case just modify self.params for changing single values.
        UPDATE 2: NOW BROKEN for custom kwarg passing, although same code working in independent tests...????
        UPDATE 3: NOW working again, but not for combined case (i.e. use kwargs OR self.params, not mix).
        """
        
        # print(type(kwargs))
        # print(kwargs)
        if kwargs:
            # Try getting params object if passed
            params = kwargs.get('params', None )
            
            # Otherwise set new params object
            if params is None:
                logger.info(f"Setting params from passed kwargs.")
                # Set to params object here, although can also use **kwargs directly in self.model.eval()
                params = self.model.make_params(**kwargs)
                
            else:
                logger.info(f"Using params from passed parameters object.")
                
        else:
            params = self.params
            logger.info(f"Using params from self.params.")
            
        # params = self.params
        
        currentModel = self.model.eval(params=params, x=self.data[dim])  #, **kwargs)
        # currentModel = self.model.eval(self.params, x=self.data[dim])
#         compDictReformat = {k:(dim,v) for k,v in compDict.items()}
        currentModel = xr.DataArray(currentModel, coords={dim:self.data[dim]})
    
        currentModel.name = 'current'
        currentModel.attrs['params'] = params
        
        self.current = currentModel
        
        # Add components?
        # Note this also needs params set, otherwise fails for custom models in testing.
        # (Although seems to work for built-in models without additional passing.)
        compDict = self.model.eval_components(params = params, x=self.data[dim])  #.values)

        # need to reformat for XR...
        compDictReformat = {k:(dim,v) for k,v in compDict.items()}
        compXR = xr.Dataset(compDictReformat, coords={'x':self.data[dim]})
        
#         compXR.name = 'components'
        compXR.attrs['params'] = params
        
        self.currentComponents = compXR
        


    def fit(self, dim='x', plotFit = True, displayDetails = False):
        """
        Fitting with xarray_lmfit wrapper.
        
        NOTE: consider passing **kwargs here, e.g. can set `guess=True` for xlm.modelfit param guess.
        
        UPDATE 23/07/25: added case for no xarray_lmfit wrapper, but to make both methods compatible is more work than just not using wrapper full stop! Currently have pushed data to XRs in either case.
        
        """
        
        
        if xrlmfitFlag:
            logger.info("Running lmfit routine with XRlmfit wrapper...")
            fitDS = self.data.xlm.modelfit(dim, model=self.model, params=self.params)
                        # NOTE: can also set guess=True to set model params inline here
                        # Will leave as-is currently, since already coded up params part in model routine.
            self.fitDS = fitDS
            self.fitParams = fitDS['modelfit_coefficients']
            self.bestFit = fitDS['modelfit_best_fit']

    #         data.xlm.modelfit('x', model=peaks2.model, guess=True)

            # Test components - don't seem to be output automatically...
            # NOTE that Fit Results includes eval_components with fitted params.
            # Not sure how to cleanly pull this...?
            lmfitResult = fitDS['modelfit_results'].data.tolist()
        
        
        # Basic lmfit run without XR wrapper -- may need some work.
        else:
            logger.info("Running lmfit routine...")
            lmfitResult = self.model.fit(self.data, x=self.data[dim], params=self.params)
            
            self.fitDS = "Not available without xrlmfit wrapper."
            self.fitParams = lmfitResult.params
            self.bestFit = xr.DataArray(lmfitResult.best_fit, coords={dim:self.data[dim]}, name='best fit')
    
    
        self.fitResults = lmfitResult
        
        compDict = lmfitResult.eval_components(x=self.data[dim])  #.values)

        # need to reformat for XR...
        compDictReformat = {k:(dim,v) for k,v in compDict.items()}
        compXR = xr.Dataset(compDictReformat, coords={'x':self.data[dim]})
    #         compXR.name = 'Fit components'

        self.fitComponents = compXR
    
    
        if plotFit:
            self.plotFit()
            
        if displayDetails:
            self.fitDetails()
        

#     def fitDetails(self):
#         """
#         Display lmfit results details.
#         """
        
#         # Not sure how to cleanly pull this...?
#         return self.fitResults['modelfit_results'].data.tolist()
    
    
    
    #**** Plotters - should consolidate!
    def plotCurrent(self, plotComponents = True):
        """
        Plot model results in self.current
        """
        
        plotOut = self.current.hvplot(label='current')
        
        if plotComponents:
            plotOut *= self.currentComponents.hvplot(line_dash='dashed', group_label='components') 
            
        display(plotOut)
    
    
    def plotFit(self, plotComponents = True, plotInitial = False):
        """
        Use HVplot to plot results from Xarrays.
        """
        
        # plotOut = self.fitResults['modelfit_best_fit'].hvplot(label='fit', title='Fit results') \
        #             * self.fitResults['modelfit_data'].hvplot(label='data', alpha=0.7)
        
        plotOut = self.bestFit.hvplot(label='fit', title='Fit results') \
            * self.data.hvplot(label='data', alpha=0.7)
        
        if plotComponents:
            plotOut *= self.fitComponents.hvplot(line_dash='dashed', group_label='components') 
        
        if plotInitial:
            plotOut *= self.initialFit.hvplot(line_dash='dotted', label='initial guess')
            
        display(plotOut)