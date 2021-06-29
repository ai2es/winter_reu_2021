#!/bin/bash
#
#SBATCH --partition=normal
###SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=10
# memory in MB
#SBATCH --mem=15000
#SBATCH --output=results/hw4_deep_B_%04a_stdout.txt
#SBATCH --error=results/hw4_deep_B_%04a_stderr.txt
#SBATCH --time=08:00:00
#SBATCH --job-name=hw4
#SBATCH --mail-user=andrewhfagg@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/fagg/aml_2021/code/hw/hw4ahf
#SBATCH --array=0-4
#
#################################################
source ~fagg/pythonenv/tensorflow/bin/activate

#python hw4_base.py @net_shallow.txt -vv -rotation $SLURM_ARRAY_TASK_ID
#python hw4_base.py @net_shallow.txt -l2 .001 -rotation $SLURM_ARRAY_TASK_ID
#python hw4_base.py @net_deep.txt -l2 .00001 -rotation $SLURM_ARRAY_TASK_ID
#python hw4_base.py @net_deep.txt -l2 .0001 -patience 1000 -rotation $SLURM_ARRAY_TASK_ID
#python hw4_base.py @oscer.txt @net_deep2.txt -rotation $SLURM_ARRAY_TASK_ID
#python hw4_base.py @oscer.txt @exp.txt @net_shallow3.txt -rotation $SLURM_ARRAY_TASK_ID
#python hw4_base.py @oscer.txt @exp.txt @net_shallow3.txt -l2 .001 -rotation $SLURM_ARRAY_TASK_ID
#python hw4_base.py @oscer.txt @exp.txt @net_shallow3.txt -l2 .001 -conv_nfilters 10 20 40 -rotation $SLURM_ARRAY_TASK_ID
#python hw4_base.py @oscer.txt @exp.txt @net_shallow3.txt -l2 .001 -hidden 100 30 -rotation $SLURM_ARRAY_TASK_ID
#python hw4_base.py @oscer.txt @exp.txt @net_shallow3.txt -l2 .001 -hidden 100 30 -rotation $SLURM_ARRAY_TASK_ID -label rep2
#python hw4_base.py @oscer.txt @exp.txt @net_deep3.txt -l2 .001 -rotation $SLURM_ARRAY_TASK_ID -v
python hw4_base.py @oscer.txt @exp.txt @net_deep3.txt -l2 .001 -rotation $SLURM_ARRAY_TASK_ID -v -batch 400 -label deep_B
