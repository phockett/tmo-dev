"""
Wrappers & methods for vmi image inversion.

"""


from pathlib import Path
import sys
import inspect

import vmi as vmi

# Set cpbasex imports
# Imports adapted from Elio's code from tmolw0618/scratch/results/tmo_ana/elio/elec_pbasex.ipynb
# Code adapted from similar method in ePSproc.orbPlot.importChemLabQC
# This will import pbasex + helper functions if found

def importCPBASEX(pbasexPath = None, basisPath = None, imgFlag = True):
    """
    Import cpBasex & additional functions.

    This is set as optional, and a local path can be defiened, since it may already be installed in local env., or in data processing location.

    A different path can also be specified for the basis sets required.

    Code:
    - CPBASEX: https://github.com/e-champenois/CPBASEX
    - Quadrant: https://github.com/e-champenois/quadrant
    - Polar-Rebinning: https://github.com/e-champenois/Polar-Rebinning

    For local install instructions, see https://github.com/e-champenois/CPBASEX#installation

    To do: add more error parsing and logging, see, e.g., https://airbrake.io/blog/python-exception-handling/importerror-and-modulenotfounderror

    """

    cpImport = False

    # If pbasex is locally installed, this should just work...
    if pbasexPath is None:
        try:
            import pbasex
            import quadrant

            cpImport = True
            pbasexPath = Path(inspect.getfile(pbasex)).parent

        except ImportError:
            pass

    if pbasexPath is not None:
        try:
            # Load dev scripts - sidesteps issues with chemlab installation
            sys.path.append(pbasexPath)

            # Import from local dir, assuming both modules present for the moment
            import pbasex
            import quadrant

            cpImport = True

        except ImportError:
            pass

    if not cpImport:
        print('cpBasex not found.')
        return None
    else:
        print(f'cpBasex import OK, checking for basis functions...')

    try:
        # if basisPath is None:
        #     basisPath = pbasexPath  # This could work if files are parsed.

        if basisPath is not None:
            gBasis = pbasex.loadG(Path(basisPath), make_images = imgFlag)
            print(f'Found basis at {basisPath}.')
        else:
            gBasis = None
            print('Basis file not set, please run importer again with basisPath=<gBasis file location>')

    except OSError:
        print(f'*** Failed to load basis from {basisPath}.')

    return (pbasex, quadrant, gBasis)


# Class to hold inversion methods.
# Inherit from base VMI class.
class VMIproc(vmi.VMI):

    def __init__(self, method='cpbasex', **kwargs):
        # Run __init__ from base class
        super().__init__(**kwargs)

        self.method = method

    # Import specific method
    def setMethod(**kwargs):
        if self.method == 'cpbasex':
            method = importCPBASEX(**kwargs)

            # Unpack tuple if returned
            if method is not None:
                self.cp, self.qu, self.gBasis = method

        else:
            print(f'No method {self.method}.')
