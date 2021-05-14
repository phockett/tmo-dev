import numpy as np
import pandas as pd


# TODO: param setting, IO from JSON file?
# def setParams(self, params = None):
#     params = {'jetvx_SI':0,'jetvy_SI':0,'jetx0_SI':0,'jety0_SI':0,
#                   'T0':0,'C':1.0,
#     'DelayA':6.66666,'DelayB':-6.66666,'DelayC':-5.688,'ManualDelayShift':244360,
#              'RadialVelocityScaleCoefficient':1.18}

def setup(self):
    """Setup for SACLA data """

    # Calibration params.
    # Quick hack with defaults hard-coding here for now!
    self.saclaCalibParams = {'jetvx_SI':0,'jetvy_SI':0,'jetx0_SI':0,'jety0_SI':0,
                             'T0':0,'C':1.0,
                             'DelayA':6.66666,'DelayB':-6.66666,'DelayC':-5.688,'ManualDelayShift':244360,
                             'RadialVelocityScaleCoefficient':1.18}

    # self.dTypes.extend(['scRaw', 'df'])
    self.dTypes.extend(['df'])



def calibration(self, params = None, keys = None, dTypeIn = 'scRaw', dTypeOut = 'raw'):
    """
    Convert raw SACLA data from h5 file to per-shot formatting, return as Pandas DataFrame.

    Defaults to 'scRaw' input data and 'raw' output for interoperability with existing codebase.
    
    Also resets global indexers self.data[key]['items'] and self.data[key]['dims'] for consistency with filters etc.

    Parameters
    ----------
    params : optional, default = None
        Pass dictionary of calibration parameters.
        If not set, use `self.saclaCalibParams`

    keys : list, optional, default = None
        Datasets to process, defaults to self.runs['proc']



    Notes
    ------

    Code adapted from "SACLA2021 Analysis Example - IR-XUV Thiophene Scan.ipynb"

    Notebook from Felix Allum, likely with some prior credit to others too!

    13/05/21 v1

    """
    # Calibration params.
    # Quick hack with defaults hard-coding here for now!
    if params is None:
        params = self.saclaCalibParams

    # Default to all datasets
    if keys is None:
        keys = self.runs['proc']

    #     if not hasattr(self, 'metrics'):
    #         self.metrics = {}

    for key in keys:

        hdf = self.data[key][dTypeIn]
        ### open the file
        # with h5py.File(r'C:\Users\mb\Documents\FelixDPhil\SACLA 2021\BromoThiophene\aq092_merged.h5', 'r') as hdf:
        # with h5py.File(path + h5Filename, 'r') as hdf:
        data = {}
        #### iterate over each dataset in the file, and load them all
        for dimKey in list(hdf.keys()):
            print(dimKey)
            data[dimKey] = np.array(hdf[dimKey])


        #### now restructure data into the sort of format we want
        data_per_shot=[]
        for ind,shot in enumerate(data['tagevent']):
            delay_calibrated=(params['DelayA']*data['delay_motor'][ind]+params['DelayB']*data['delay_offset'][ind]+params['DelayC']*data['delay_jitter'][ind]+params['ManualDelayShift'])
            tagevent = data['tagevent'][ind]
            FEL_int = data['fel_shutter'][ind]*data['fel_intensity'][ind]
            LAS_int        = float(data['laser_shutter'][ind])
            #### include laser shots without ions
            if data['nions'][ind] == 0:
                data_per_shot.append([tagevent,delay_calibrated,np.nan,np.nan,np.nan,FEL_int,LAS_int,np.nan,np.nan])
            else:
                for ind2 in range(data['nions'][ind]):
                    start         = data['nlistpos'][ind]
                    end           = data['nlistpos'][ind] + data['nions'][ind]
                    tof           = data['tof'][start:end][ind2]
                    realtof_SI    = (tof-params['T0'])*1e-9
                    xpos          = data['xpos'][start:end][ind2]
                    ypos          = data['ypos'][start:end][ind2]


                    dataunit=np.array([tagevent, delay_calibrated,tof,
                    params['RadialVelocityScaleCoefficient']*((xpos*0.001-params['jetx0_SI'])/realtof_SI- params['jetvx_SI']),
                    params['RadialVelocityScaleCoefficient']*((ypos*0.001-params['jety0_SI'])/realtof_SI- params['jetvy_SI']),
                    FEL_int,LAS_int,
                    xpos*0.001-params['jetx0_SI']-realtof_SI*params['jetvx_SI'],
                    ypos*0.001-params['jety0_SI']-realtof_SI*params['jetvy_SI']])

                    data_per_shot.append(dataunit)

        #### now combine the data from each laser shot into a large dataframe
        self.data[key][dTypeOut] = pd.DataFrame(np.vstack(data_per_shot), columns=['tagID','delay','tof','vx','vy','FEL_int','LAS_int','x','y'])
        
        # Also reset dim indexes, to avoid inconsistencies later.
        # UGLY!!!
        self.data[key]['items'] = self.data[key][dTypeOut].keys()
        self.data[key]['dims'] = {item:self.data[key][dTypeOut][item].shape for item in self.data[key]['raw'].keys()}
