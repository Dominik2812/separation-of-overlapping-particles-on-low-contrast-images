# separation of overlapping particles on low contrast images.

## Project description
Why does this project exist? 
In my company we produce Polymer beads. Their size distribution has to be determinded via random sampling; possibly quick and reliable. One apprach to rapidly count circular objects on an image is represented in my Pearl Chain project. However the pearl chain methode fails when the objects touch eachother. Infact the micrographs that are available to us are very poor in contrast and conventional methods to separate touching particles such as 'watershed' fail in this case as well.

In this repository you find two codes that use distance transform to make watershed more effective, even in low constrast images. The code is slower than the pearl cain methode but still is able to  process 10-20 micrographs within less than 5 min, which provides sufficient daata to meassure the a size distribution of those beads. 

## How does it work?

### The heart of the code
Dark regions below a certain local threshold are identified as background from which follows that brighter regions are objects of interest. Not all of these objects are single beads as some of them touch eachother and would be detected as one object. Now the heart of both codes,a distance transform, comes into play. Regions where particles touch are getting darker, whereas the center of the beads shine brightest. By determining local maxima and the local contours single beads are identified. 

### troubleshooting:, removing  overlapping
As the image are not the best quality, also the mechanism described above will detect circles do not correspond to any bead in the image. It turned out that those errors occure in certain regions of the image with the consequence that several errors accumulate there. Thus those falsely detected circles are usually characterized by a significant overlap. Both codes remove circles from the statistics that exceed an overlap of a certain limit. You can adjust this limit suitable to your specific problem. It is reasonable to choose a value between 0.6 and 0.8, meaning that if the distance between the centers of the circles is smaller than 60% or 80% of the sum of the radia, the circles probably do not correspond to real beads. 

### Why two versions? Poly and Mono....
Even circles that correspond to real beads often overestimate the radius by 10 to 20% and overlap with their neighbors. If the size distribution of the beads is is narrow (DRIVE_MONO.py), it helps to shrink those circles until they don't overlap anymore (see example image ![Output Image](Output_image.JPG?raw=true "Output Image")). The circle contours then precisely describe the contours of the beads. This correction does however not apply to braod distributions (range of radia exceeds order of magnitude, e.g. 50-500µm,DRIVE_POLY.py). Small beads will be partially covered by big beads which leads to a strong overlap. Shrinking the radius willthen lead to a significant error in size ditribution. In this case one has to live with the overestimation.

### (Mass)Histogramms 
'DRIVE_MONO.py' and 'DRIVE_POLY.py' also differ regarding their output histogramms. This has no specific reason, I just wanted to experiment a bit with the histograms as I am just learning Python. I distinguish between local and global histograms; the first displaying the mass distribution of each single image, the latter sums up all the data from the previous images as well. While in 'DRIVE_MONO.py' there is only a global histogram, 'DRIVE_POLY.py' also considers local distributions. 

### Circles colorcode
In 'DRIVE_POLY.py' correctly detected cicles are colorcoded, to visualzed whether the bead size is below, within or above a desired size range. Circles that overlap strongly are not shown anymore, in contrast to 'DRIVE_MONO.py'.

## Installation
You need Python >= 3.6 installed on your sytem. To install the required packages, run 
```
pip install -r requirements.txt
```

## Run particle detection

To start particle detection for monodisperse particles (using images from the `Monodisperse_Input_images` folder) , run
```
python DRIVE_MONO.py
```
For polydisperse particles (using images from the `Polydisperse_Input_images` folder) , run
```
python DRIVE_POLY.py
```

[Where to find result? What are the results (histogram etc)]

## Example DRIVE_MONO

### Input image
![Input Image](MONO_Input.JPG?raw=true "Input Image Monodisperse")
### Output image
![Output Image](Output_MONO.JPG?raw=true "Output Image Monodisperse")
### GlobalHistogram
![histogramm](MONO_GLOBAL_Histogramm.JPG?raw=true "Global Histogram Monodisperse")

## Example DRIVE_POLY

### Input image
![Input Image](POLY_Input.JPG?raw=true "Input Image Monodisperse")
### Output image
![Output Image](Output_POLY.JPG?raw=true "Output Image Monodisperse")
### GlobalHistogram
![histogramm](POLY__GLOBAL_Histogramm_POLY.JPG?raw=true "Global Histogram Monodisperse")
### localHistogram
![histogramm](POLY__local_Histogramm.JPG?raw=true "local Histogram Monodisperse")



