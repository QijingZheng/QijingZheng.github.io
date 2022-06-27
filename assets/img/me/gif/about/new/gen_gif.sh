#!/bin/bash

convert -loop 0 -delay '1x12' -resize 40% \
        zqj.png zqj.png zqj.png zqj.png \
        zqj.png zqj.png zqj.png zqj.png \
        {7..1}.png \
        1.png 1.png \
        {1..7}.png \
        zqj.png zqj.png zqj.png \
        zqj_redeye.gif


