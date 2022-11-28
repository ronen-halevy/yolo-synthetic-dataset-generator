#!/bin/bash

# creates a video from a sequence of images:

ffmpeg -framerate 2 -i %06d.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4