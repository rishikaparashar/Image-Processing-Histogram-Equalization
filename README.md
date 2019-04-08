# Image-Processing-Histogram-Equalization
Applying histogram equalization on an image using opencv and python
The program gets as input a color image, performs histogram equalization in the Luv domain, and
writes the scaled image as output. 
Histogram equalization in Luv is applied to the luminance values, as computed in the specified window. It requires a discretization step, where the real-valued L is discretized into 101 values.
In the program, pixel values outside the window are not changed. Only pixels within the window are changed.
