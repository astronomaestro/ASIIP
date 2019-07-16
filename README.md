# Intensity Interferometry Target Planner
* What this target planner does

-A general purpose Intensity Interferometry target planner that can be used for a given telescope array. It was initally designed for Veritas
but can work with any Intensity Interferometry system. 

-Queries 5 different catalogs with custom constraints defined by the user and constructs a master catalog from them. 
It then cross checks found targets with the Bright Star Catalog and SIMBAD to obtain B magnitudes, radial velocities, 
spectral types and the star id.

-It then analyzes the identified targets and determines to first order how measurable they are with the current telescope 
array. It does this by repeatedly fitting analytical models to empirical models and then ranking each target by how well
the analytical fits converge to the true angular diameter fit value. 


* Running the Script

This Python script was written using Python 3.7, so a Python 2 compiler might not work as expected. To run the target 
planner, open a terminal (or command prompt) and change your directory to where interferometer.py is located and then 
type this

    - python interferometer.py


There are some package dependencies that you will need in order to get the script running.

    - astropy
    - astroquery
    - numpy
    - matplotlib 
    - json

All of these should be easily installed using 'pip install (your package)'. If there happens to be any more packages you
need use pip install should easily be able to help you with those as well.


* The Parameter input file

    
    
This is a json file which is used to modify the many differing parameters the script uses to calculate which targets to
analyze. Description of the parameters are given below. If you want to see an example input values, look at the file
IIparameters.json included with this package.
 
    - time: Specifies the desired date of observation. The target planner expects a time starting at midnight
    - raRange: Specifies the desired RA range query constraints
    - decRange: Specifies the DEC range query constraints 
    - magrange: Specifies the Magnitude range query constraints
    - wavelength: Specifies the wavelength of the telescope filter being used
    - telLocs: Releative telescope locations of the telescopes you will be using to preform your observation
    - integrationTIme: The lenth, in hours, of your integration interval
    - telLat: the central latitude of the telescope array
    - telLon: the central longitude of the telescope array
    - telElv: the central elevation of the telescope array
    - observationStart: The desired start time of the observation. An input of 0 starts the observation at midnight. An input of null stars the observation as early as possible.
    - observationEnd: The desired end time of the observation. An input of 0 ends the observation at midnight. An input of null ends the observation as late as possible.
    - sigmaTel: The predetermined error of your array.
    - sigmaMag: The magnitude of the star used when determining sigmaTel.
    - sigmaTime: The integration time used when determining sigmaTel.
    - bootStrapRuns: How many times you desire to fit an analytical model to an empirical one.
    
   
In the event you want multiple parameter files or you want to change the name of your parameter file you can by running 
the script like this

     - python interferometer.py (name of your parameter file).json


* How to interpret the results of the analysis

Once the analysis has run it's course, it will print out a table with data on the input targets. A definition of the
columns is given below

    -Index: this specfies a number which is tied to the row. Use this to choose which star you want to make graphs of
    -SIMID: The ID given by SIMBAD to star located at the associated RA and DEC
    -RA: The stars J2000 Right Acension 
    -DEC: The stars J2000 Declination
    -ANGD: The stars angular diameter in milli arc seconds (mas)
    -FILT: The filter that was used when querying the catalog
    -MAG: The magnitude associated with the FILT entry
    -CAT: The catalog the entry comes from
    -BS_BMAG: The Bright Star catalog B magnitude
    -BS_VMAG: The Bright Star catalog V magnitude
    -BSSkyD: The difference between the input RA and DEC vs the Bright Star catalog RA and DEC in mas. If this is not 0
    it means that the RA and DEC don't match precicely and this is the closest matching position.
    -BSSpT: the spectral type as given by the Bright Star Catalog
    -BSRV: THe radial velocity as given by the Bright Star Catalog in km/s
    -SimBMAG: The SIMBAD catalog B magnitude
    -SIMSpT: the spectral type as given by SIMBAD
    -SIMSkyD: The difference between the input RA and DEC vs the SIMBAD RA and DEC in mas. If this is not 0
    it means that the RA and DEC don't match precicely and this is the closest matching position.
    -SIMRV: The radial velocity as given by SIMBAD in km/s
    -ErrAmp: The calculated error for the given integration time and magnitude
    -TotObsTime: The total available observation time
    -ObsTime: The integration time used in the analysis
    -MeanDiamFit: The mean analytical fit of the empirical models
    -PerDiamErr: the percentage error of the mean diameter fit to the true value
    -PerErrDiamFitErr: The percentage deviation of the fit from trial to trial. If this is higher then .2 then caution 
    should be taken when fitting this data as the fit may simply be converging to a value close to the guess value
    -PerFailFit: The percentage of how many analytical fits failed. If this is high, then it's very likely a poor target
    to observe
