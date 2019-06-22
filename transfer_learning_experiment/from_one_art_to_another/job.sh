#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=50GB

KERAS_BACKEND=tensorflow

declare -a tl_modes=("fine_tuning" "off_the_shelf") #The two TL approaches investigated in the paper

dataset_name="Art_Style_Macys" #The name of your dataset

metadata_path="/media/dev_hdd2/ramya/art_style_exps/approaches/metadata/data.csv" #The path to your metadata file in .csv extension

results_path="/media/dev_hdd2/ramya/art_style_exps/approaches/results" #The path where wou would like to store your results
datasets_path="/media/dev_hdd2/ramya/art_style_exps/approaches/hdf5_files" #The path where the hdf5 files will be stored for the experiments

tl_mode="fine_tuning" #Choose a pre-training mode from tl_modes

python transfer_learning_experiment.py --dataset_name $dataset_name --ANN
"RijksVGG19" --metadata_path $metadata_path --results_path $results_path --datasets_path $datasets_path --tl_mode $tl_mode
