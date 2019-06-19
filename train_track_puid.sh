#!/bin/bash

scratch=/scratch/spigazzi/MTD/TDR/TrackPUID/
train_name=ttbar_PMV13_MTDquality_v1

ls $scratch

mkdir -p $scratch/results/$train_name/

export PATH=/t3home/spigazzi/anaconda3/bin/:$PATH
export CONDA_PATH=/t3home/spigazzi/anaconda3/

source activate python3

if [ "$#" -eq 0 ]; then
    python /t3home/spigazzi/MTD/TrackPUID/train_track_puid.py --inp-dir $scratch/samples/ttbar_PMV13_v1/ --inp-file input_tracks_train_3D.hd5 --out-dir $scratch/results/$train_name/
elif [ "$#" -eq 1 ]; then
    python /t3home/spigazzi/MTD/TrackPUID/train_track_puid_MTDquality.py --inp-dir $scratch/samples/ttbar_PMV13_v1/ --inp-file input_tracks_train_4D.hd5 --out-dir $scratch/results/$train_name/ --wmtd    
fi

