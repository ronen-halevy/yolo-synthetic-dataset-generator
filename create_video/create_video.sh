#!/bin/bash

# creates a video from a sequence of images:
#ffmpeg -framerate 2 -i dataset/train/img_%06d.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4
ffmpeg -framerate 2 -i dataset/red_circle/img_%06d.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p red_circle.mp4
ffmpeg -framerate 2 -i dataset/red_circle/img_00006%01d.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p red_circle_short.mp4
