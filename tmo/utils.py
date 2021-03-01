"""
Class and method dev for SLAC TMO data (Run 18) - base class (IO + basic analysis and plots).

Preprocessed data (h5py IO) + further processing + Holoviews.

18/11/20 v0.0.1

Paul Hockett

https://github.com/phockett/tmo-dev

"""


def _checkDims(self, dataType = 'imgStack', dimsCheck = None, swapDims = None):
    """
    Check if passed dims exist in Xarray.
    Also test if dims are dimensional (key) or non-dimensional.
    Pass swapDims to provide (old) dims to swap with non-dimensional dims.

    """

    # Set dataset
    testData = getattr(self, dataType)

    #     dimsND = []
    #     dimsMissing = []
    #     dimsOK = []

    #     for dim in dims:
    #         # Check if dim is dimensional/key dim
    #         if dim in testData.dims:
    # #             dims.pop()
    #             dimsOK.append(dim)

    #         # Check if dim exists at all
    #         elif dim in list(testData.coords.keys()):
    #             dimsND.append(dim)

    #         else:
    #             dimsMissing.append(dim)

    # Check with sets
    dimsD = {*testData.dims} & {*dimsCheck}
    dimsND = {*testData.coords.keys()} & ({*dimsCheck} - dimsD)
    dimsMissing = {*dimsCheck} ^ dimsD ^ dimsND

    if self.verbose['sub']:
        print(f"Checked dims, found dims {dimsD}, ND dims {dimsND}, missing {dimsMissing}")

    if dimsND and (swapDims is not None):
        if len(swapDims) == len(dimsND):
            setattr(self, dataType, testData.swap_dims(dict(zip(swapDims, dimsND))))

            if self.verbose['sub']:
                print(f"Swapped dims: {dict(zip(swapDims, dimsND))} and set {dataType}")

        else:
            print(f"*** WARNING: couldn't swap dim sets {swapDims} for {dimsND}, length doesn't match.")
