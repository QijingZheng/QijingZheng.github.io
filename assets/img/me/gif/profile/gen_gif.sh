#!/bin/bash

convert -loop 0 -delay '1x12' -resize 30% \
        7.jpg 7.jpg 7.jpg 7.jpg \
        {6..1}.jpg \
        1.jpg 1.jpg 1.jpg 1.jpg \
        1.jpg 1.jpg 1.jpg 1.jpg \
        1.jpg 1.jpg 1.jpg 1.jpg \
        {1..6}.jpg \
        7.jpg 7.jpg 7.jpg \
        profile_redeye.gif


