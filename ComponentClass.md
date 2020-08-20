# Component class for radio observations
The component class is designed to hold information about the observation of a radio component, for which the essential properties of the Component class are:
+ RA, Dec, error in RA, error in Dec (i.e. instance of Coordinate class) <-- Deconvolved
+ Peak flux, rms flux, imfit flux, imfit flux err, frequency (i.e. instance of Flux class)
+ Beam (instance of Beam class), robustness, cell size 