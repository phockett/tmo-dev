"""
Class and method dev for SLAC TMO data (Run 18) - raw data handling class.

Read raw data (XTC files + psana IO) + further processing + Holoviews.

Note: this is currently very low-level, but imports methods already developed for preprocessed data.

09/12/20 v0.0.1 Devoping from _perShot version for more general use. Code dev from https://pswww.slac.stanford.edu/jupyterhub/user/phockett/lab/workspaces/scratch-Feb21/tree/dev/results_LW08/LW08_raw_VMI_94-101_091220.ipynb and https://pswww.slac.stanford.edu/jupyterhub/user/phockett/lab/workspaces/scratch-Feb21/tree/dev/results_LW08/read_raw_LW08mod_061220.ipynb

Paul Hockett

https://github.com/phockett/tmo-dev

"""

# Dev code for new class
# Inherit from base class, just add evmi functionality here
import tmoDataBase as tb
import inversion as inv

# psana for grabbing raw data.
import psana as ps

class getRaw(tb.tmoDataBase, inv.VMIproc):

    def __init__(self, **kwargs):
        # Run __init__ from base class
        super().__init__(**kwargs)

    def getRun(self, run_number):
        # Set data
        expt = self.runs['expt']
        ds = ps.DataSource(exp=expt, run=run_number)
        run = next(ds.runs())

        return run

    # Minimal function for pulling timing details & VMI data
    # Basically follows example `read_raw.ipynb` example notebook
    def getRawDataImgs(self, run_number, Nfind = 300, nskip = 1, shotImgs = True, filterOptions = {}):

        # # Set data
        # expt = self.runs['expt']
        # ds = ps.DataSource(exp=expt, run=run_number)
        # run = next(ds.runs())


        # Get detectors
        if self.verbose['main']:
            print(f"*** Reading raw data for expt={expt}, run={run}")
            print(f"Found detectors: {run.detnames}")

        data = {}
        for det in run.detnames:
            data[det] = det
            data[det]['raw'] = run.Detector(det)
            data[det]['items'] = [item for item in dir(data[det]) if not item.startswith('_')]

        # tt_camera = run.Detector('tmoopal2')
        # opal = run.Detector('tmoopal')
        # ims = None

        # Set data
        # items = [item for item in dir(tt_camera.ttfex) if not item.startswith('_')]

        Nfound = 0

    #     ttFltpos = np.empty(Nfind)
    #     ttPS = np.empty(Nfind)
        # ttArray = np.empty((Nfind,len(items)))


        for nevent, event in enumerate(run.events()):

            if nevent%nskip!=0: continue

            # Loop over values and set to master array
            for n, item in enumerate(items):
    #             print(f"{item}: {type(getattr(tt_camera.ttfex, item)(event))}")

                data = getattr(tt_camera.ttfex, item)(event)

                if type(data) is not np.ndarray:
                    ttArray[Nfound,n] = data

                # For now skip the array data in proj_ref and proj_sig, although this should probably be looked at!
                else:
                    pass

                # Opal images
                im = opal.raw.image(event)
                if im is None:
                    print("Didn't find Opal")
                    continue

                if ims is None:
                    ims = np.empty((Nfind,)+im.shape)

                ims[Nfound] = im

    #         ttFltpos[Nfound] = tt_camera.ttfex.fltpos(event)
    #         ttPS[Nfound] = tt_camera.ttfex.fltpos_ps(event)
    #         ttImg = tt_camera.raw.image(event)

            Nfound += 1
            if Nfound==Nfind: break

        # Set Xarray for images
        # May want to use a more meaningful metric for 'shot' here
        print(f'Run {run_number}, N={Nfind}')
        dim = ['xc','yc']
        imgStack = xr.DataArray(ims, dims=['shot',dim[0],dim[1]],
                                coords={'shot':np.arange(0,ims.shape[0]), dim[0]:np.arange(0,ims.shape[1]), dim[1]:np.arange(0,ims.shape[1])},
                                name = run_number)
    #                             name = f'Run {run_number}, N={Nfind}') #.expand_dims({'run':[run_number]})

    #     imgStack['ttps'] = ('shot',ttArray)
        imgStack.attrs['ttps'] = ('shot',ttArray)
        imgStack.attrs['Nfind'] = Nfind

        return imgStack

    #     return ttArray, ims, imgStack

    # Loop over runs & set data
    def getRawRuns(self, shotImgs = False, filterOptions = {}, **kwargs):
        for run in self.runs['runList']:

            imgStack = self.getRawDataImgs(run, shotImgs=shotImgs, filterOptions=filterOptions, **kwargs)

            if not hasattr(self,'imgStackShots'):
                # self.imgStack = []  # May need .copy() here?  # v1, set as list
                self.imgStackShots = xr.Dataset()  # v2, set as xr.Dataset and append Xarrays to this

    #         self.imgStack.append(imgStack.copy())  # May need .copy() here?  # v1
    #         self.imgStack[imgStack.name] = imgStack.copy()
            self.imgStackShots[imgStack.name] = imgStack.copy()

    def shotImgRestack(self):
        # Sum over shots and restack
        restack = self.imgStackShots.sum('shot').to_array(dim = 'name').rename('shotImgs').rename({'name':'run'})
        self.imgStack = restack.to_dataset()  # Push back to dataset to match existing convention
