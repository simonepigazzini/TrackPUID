#!/bin/bash
#SBATCH --job-name=xgbo_mtd_trackpuid
#SBATCH --account=gpu_gres  # to access gpu resources
#SBATCH --nodes=1       # request to run job on single node                                       
#SBATCH --ntasks=5     # request 10 CPU's (t3gpu01: balance between CPU and GPU : 5CPU/1GPU)      
#SBATCH --gres=gpu:1    # request 2 GPU's on machine                                         
#SBATCH -o /t3home/spigazzi/MTD/TrackPUID/logs/xgbo_train_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e /t3home/spigazzi/MTD/TrackPUID/logs/xgbo_train_%j.err  # File to which STDERR will be written, %j inserts jobid

scratch=/scratch/spigazzi/MTD/TDR/TrackPUID/
train_name=ttbar_PMV13_v1_bo

ls $scratch

mkdir -p $scratch/results/$train_name/

export PATH=/t3home/spigazzi/anaconda3/bin/:$PATH
export CONDA_PATH=/t3home/spigazzi/anaconda3/

source activate python3

if [ "$#" -eq 0 ]; then
    python /t3home/spigazzi/MTD/TrackPUID/train_track_puid.py --inp-dir $scratch/samples/ttbar_PMV13_v1/ --inp-file input_tracks_train_3D.hd5 --out-dir $scratch/results/$train_name/ --bo
elif [ "$#" -eq 1 ]; then
    python /t3home/spigazzi/MTD/TrackPUID/train_track_puid.py --inp-dir $scratch/samples/ttbar_PMV13_v1/ --inp-file input_tracks_train_4D.hd5 --out-dir $scratch/results/$train_name/ --wmtd --bo
fi
