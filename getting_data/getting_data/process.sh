#!/bin/bash
extr="./processed"

sh download.sh

find ./extracted -depth -name "* *" -execdir rename 's/ /_/g' {} \;
 	
echo "named replaced by strings"

comp() {
   awk -v n1="$1" -v n2="$2" 'BEGIN { print (n1 >= n2) ? "1" : "0" }'
}

types="10s,2s,noise,other,kick,snare,hat,rim,tom,shake,clap,other,pad,voc"

IFS=',' read -r -a types <<< "$types";

for e in "${types[@]}";
do
    mkdir -p $extr"/"$e
done

for i in $(find ./extracted -type f -name "*.wav" );
do
        len=$(soxi -D $i)
	bs=$(basename $i)
	pa=$(basename $(dirname $i))
	bspa="$pa"_"$bs"
	bspa=${bspa,,}
	echo $bspa $len

	if [ $(echo $(comp $len 10) -eq 1) ]
        then 
                mv $i $extr"/10s"
	elif [ $(echo $(comp $len 2) -eq 1) ]
        then 
                mv $i $extr"/2s"
	else
		for e in "${types[@]}"
		do 
			if [[ $bspa == *${e}* ]]
		       	then 
				echo $e
		                mv $i $extr"/"$e --backup=numbered
			fi	
		done
	fi
	[ -f $i ] && mv $i ./other/$bspa 
done


