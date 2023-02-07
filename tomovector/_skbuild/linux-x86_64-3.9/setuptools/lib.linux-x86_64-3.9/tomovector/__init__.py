from pkg_resources import get_distribution, DistributionNotFound

from tomovector.radonusfft import *
from tomovector.solver_tomo import *
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass