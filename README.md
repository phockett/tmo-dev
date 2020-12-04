# tmo-dev
Dev and scratch space for tmo analysis routines dev.

## Overview

Aims: develop some python classes for high(ish) level data analysis, including interactive plotting.

This mainly wraps or otherwise uses similar base functionality to that developed and demonstrated by the TMO team (thanks all!), plus adds some simple to use filtering, plotting and analysis functionality (using [Xarray](http://xarray.pydata.org/en/stable/) & [Holoviews](http://holoviews.org/) for the most part). It currently accepts preprocessed data only.

Demos are [available in HTML format](https://phockett.github.io/tmo-dev/demos/index.html), with [notebooks via docs/demos in this repo](https://github.com/phockett/tmo-dev/tree/main/docs/demos).

To many use of any of this, you'll need both the tmo-dev source code from Github, and also to set up your own Anaconda environment to install the additional packages required. General instructions are on the [base class demo page](https://phockett.github.io/tmo-dev/demos/classDemo_191120.html)  and there's an [Anaconda + package guide from SLAC here](https://confluence.slac.stanford.edu/display/PSDM/Installing+Your+Own+Python+Package).


## Releases

- 04/12/20 v0.0.1. Basic working version for low-level data analysis + VMI image processing.
