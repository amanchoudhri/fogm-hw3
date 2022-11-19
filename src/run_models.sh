#!/bin/bash
#SBATCH -p ccn
#SBATCH -N 1
#SBATCH --array=4,5,8,10,12,15,18,20
#SBATCH --time=12:00:00
#SBATCH -o logs/run_models_%j.log

MODEL=$1

case $MODEL in
    diagonal_mixture|mixture|diagonal_mixed_membership|mixed_membership)
        echo "Selected model ${MODEL}"
        ;;
    *)
        echo "Invalid model type ${MODEL}!"
        exit
        ;;
esac

OUTDIR=$2

if [[ -z $OUTDIR ]]; then
    OUTDIR=results
fi

DATA=$3

if [[ -z $DATA ]]; then
    DATA=data/concatenated.npz
fi

python fit.py $OUTDIR $MODEL --K $SLURM_ARRAY_TASK_ID --data $DATA
