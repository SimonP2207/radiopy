# radiopy
**radiopy** is a python library for correctly handling radio data with scope to perform common analyses of that data in a variety of contexts.

### Dependencies
**radiopy** requires the following versions/libraries/modules:
+ Python 3+
+ numpy (tested with version 1.15.1)
+ scipy (tested with version 1.1.0)
+ matplotlib (tested with version 2.2.3)
+ uncertainties (tested with version 3.0.2)

## Component class for radio observations
The component class is designed to hold information about the observation of a radio component, for which the essential properties of the Component class are:
+ RA, Dec, error in RA, error in Dec (i.e. instance of Coordinate class) <-- Deconvolved
+ Peak flux, rms flux, imfit flux, imfit flux err, frequency (i.e. instance of Flux class)
+ Beam (instance of Beam class), robustness, cell size 