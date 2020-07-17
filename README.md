# separation of overlapping particles on low contrast images.

## Project description
Why does this project exist? 
In my company we produce Polymer beads. Their size distribution has to be determinded via random sampling; possibly quick and reliable. One apprach to rapidly count circular objects on an image is represented in my Pearl Chain project. However the pearl chain methode fails when the objects touch eachother. Infact the micrographs that are available to us are very poor in contrast and conventional methods to separate touching particles such as 'watershed' fail in this case.

In this repository you find two codes that use distance transform to make watershed more effective, even in low constrast images. The code is slower than the pearl cain methode but still is able to  process 10-20 micrographs within less than 5 min and meassure the size distribution of those beads. 

## How did it solve a problem?

### The heart of the code
Dark regions below a certain local threshold are identified as background from which follows that brighter regions are objects of interest. Not all of these objects are single beads as some of them touch eachother and would be detected as one object. Now the heart of both codes,a distance transform, comes into play. Regions where particles touch are getting darker, whereas the center of the beads shine brightest. By determining local maxima and the local contour groups the beads are identified. 
### Correction 
One code (DRIVE_MONO) is ideal to measure the size distribution of relaively monodispersed (within the same oder of magnitide)

## Installation

You need Python >= 3.6 installed on your sytem. To install the required packages, run 
```
pip install -r requirements.txt
```

## Run particle detection

To start particle detection for monodisperse particles (using images from the `Monodisperse` folder) , run
```
python DRIVE_MONO.py
```
For polydisperse particles (using images from the `Polydisperse` folder) , run
```
python DRIVE_POLY.py
```

[Where to find result? What are the results (histogram etc)]

## Example

### Input image
![Input Image](Input_image.JPG?raw=true "Input Image")
### Output image
![Output Image](Output_image.JPG?raw=true "Output Image")
### Histogram
![Input Image](GLOBAL_Histogramm.JPG?raw=true "Histogram")
