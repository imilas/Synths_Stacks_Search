#!/bin/bash
mkdir -p packs
mkdir -p zips
mkdir -p extracted
mkdir -p temext #temp extractions

while read p; do
	filename=$(basename "$p")
	ls ./zips/ | grep -q $filename
	ret=$?
	if [ "$ret" -eq "1" ] 
	then
		wget $p -P ./zips/ 
		echo "let us do things to " $filename
		unzip "./zips/$filename" -d ./temext
		mv ./temext/* ./extracted

	else
		#echo not downloading $p. either link broken or already downloaded
		echo no to $(basename $p)
	fi
	done <links.txt

rm -r temext
