#!/bin/sh

files=`find $1/*`
model=$2
output=$3

for file in $files
do
	echo
	echo "***************************************"
	echo "* Generating z vector for file: $file *"
	echo "***************************************"
	echo
	bfile=`basename $file`
	python getz.py --model=$model --input=$file --output=$output/$bfile
done