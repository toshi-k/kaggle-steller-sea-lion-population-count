#!/bin/sh

for seed in `seq 0 3`
do
	th main.lua -seed ${seed}
done
