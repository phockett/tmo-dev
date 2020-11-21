
Preprocessed data formats.
=========================

v3 updates (EC, 20/11/20)
-------------------------

Besides a bunch of extra data being saved compared to v2, the main differences that might affect existing analysis code:
- No longer saving just the E-L trigger "gas" boolean array explicitly. We are now saving all 256 event codes that are shared per shot. The E-L trigger code can be accessed as the 70th column of the 2d "evrs" array (rows represent shot number).
- "enery" and "xenergy" are now saved as "gmd_energy" and "xgmd_energy", trying to follow the notation of how they are called from psana (gmd.raw.energy and xgmd.raw.energy). This way, we have automated saving this data from all detectors and all of their underlying methods.



v2 dataset notes (from EC, 17/11/20)
------------------------------------

Some datasets saved in the most recent preprocessing codes:
- timestamp: timestamp of the event, used to reorder the data in lab-time
- energies: shot to shot pulse energies measured by a gas detector before the gas attenuator
- xenergies: same as above, but after the gas attenuator
- epics_strs: array of descriptors for various epics variables that get stored, including undulator settings, High voltage settings, etc.
- epics: values for the above, on a shot by shot basis (though most are only updated at ~1Hz or so as far as I am aware)
- gas: E-L valve triggered or not, per shot (doesn't affect KToF)
- ts: array of times that were sampled in the ion ToF
- intensities: integrated intensity in a small ROI of the ion ToF waveforms at the times above, on a shot by shot basis
- xc, yc, pv: x and y centroids, and pixel values, for the electron hits found in the electron VMI images using hit finding (this is what we generally use to look at eVMI data)
- photonEs, l3s, bc2s: photon energy, ebeam energy, and ebeam charge, though these values should probably not be trusted for now to get a real photon energy measurement
soon to be added as soon as we get back up and running today on the servers: short region of the raw ktof waveform where we expect the pump and probe photolines, as well as hit finding (software CFD) output
Of course, anyone can run their own preprocessing code as well, and if there are things that you think should be added to this one that we try to run for every run, let us know


Example from lw06 run 97

test.data[97]['dims']

{'bc2s': (36198,),
 'energies': (36198,),
 'epics': (36198, 158),
 'epics_strs': (158,),
 'gas': (36198,),
 'intensities': (36198, 5),
 'ktofIpk': (36198, 1000),
 'ktofslice': (36198, 1784),
 'ktofslice_ts': (1784,),
 'ktoftpk': (36198, 1000),
 'l3s': (36198,),
 'photonEs': (36198,),
 'pv': (36198, 1000),
 'timestamp': (36198,),
 'ts': (5,),
 'xc': (36198, 1000),
 'xenergies': (36198,),
 'yc': (36198, 1000)}
