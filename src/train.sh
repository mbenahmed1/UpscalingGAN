#!/bin/bash

for((i = 0; i < 100; i++))
do
	python train.py ../seq/gen ../seq/dis
	mv ../seq/upscaled.pdf ../seq/${i}_up.pdf
    mv ../seq/full_res_image.pdf ../seq/${i}_full.pdf
    mv ../seq/low_res_image.pdf ../seq/${i}_low.pdf
done    