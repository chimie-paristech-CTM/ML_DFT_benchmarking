#!/bin/bash

# ITERATION 1
python final_model.py --iteration 1 --beta 1.8 --seed 10 --cutoff 0.75 --final_dir ../data/autode_input_1 --new_data 10 --pool_file ../data/hypothetical_chemical_space.csv --train_file ../data/data_smiles_curated.csv
python analyze_data.py --iter 1 --raw_data ../data/raw_data_round_1  --pool_file ../data/hypothetical_chemical_space.csv
python baseline_models.py --final_cv --input_file ../data/final_overview_data.csv --csv_file ../data/data_smiles_curated.csv

# ITERATION 2
python final_model.py --iteration 2 --beta 1.8 --seed 7 --cutoff 0.8 --final_dir ../data/autode_input_2 --new_data 15 --pool_file ../data/hypothetical_chemical_space_iter_1.csv --train_file ../data/data_smiles_curated.csv --selective_sampling '[3+2]cycloaddition' --selective_sampling_data 5 --new_data_file ../data/data_augmentation.csv
python analyze_data.py --iter 2 --raw_data ../data/raw_data_round_2  --pool_file ../data/hypothetical_chemical_space.csv
python baseline_models.py --final_cv --input_file ../data/final_overview_data.csv --csv_file ../data/data_smiles_curated_2.csv

# ITERATION 3
python final_model.py --iteration 3 --beta 1.8 --seed 2 --cutoff 0.8 --final_dir ../data/autode_input_3 --new_data 15 --pool_file ../data/hypothetical_chemical_space_iter_2.csv --train_file ../data/data_smiles_curated.csv --new_data_file ../data/data_augmentation.csv
python analyze_data.py --iter 3 --raw_data ../data/raw_data_round_3  --pool_file ../data/hypothetical_chemical_space.csv
python baseline_models.py --final_cv --input_file ../data/final_overview_data.csv --csv_file ../data/data_smiles_curated_3.csv

# ITERATION 4
python final_model.py --iteration 4 --beta 1.8 --seed 9 --cutoff 0.75 --final_dir ../data/autode_input_4 --new_data 15 --pool_file ../data/hypothetical_chemical_space_iter_3.csv --train_file ../data/data_smiles_curated.csv --new_data_file ../data/data_augmentation.csv
python analyze_data.py --iter 4 --raw_data ../data/raw_data_round_4  --pool_file ../data/hypothetical_chemical_space.csv
python baseline_models.py --final_cv --input_file ../data/final_overview_data.csv --csv_file ../data/data_smiles_curated_4.csv

# ITERATION 5
python final_model.py --iteration 5 --beta 1.8 --seed 20 --cutoff 0.7 --final_dir ../data/autode_input_5 --new_data 15 --pool_file ../data/hypothetical_chemical_space_iter_4.csv --train_file ../data/data_smiles_curated.csv --new_data_file ../data/data_augmentation.csv
python analyze_data.py --iter 5 --raw_data ../data/raw_data_round_5  --pool_file ../data/hypothetical_chemical_space.csv
python baseline_models.py --final_cv --input_file ../data/final_overview_data.csv --csv_file ../data/data_smiles_curated_5.csv

# ITERATION 6
python final_model.py --iteration 6 --beta 1 --seed 18 --cutoff 0.7 --final_dir ../data/autode_input_6 --new_data 50 --pool_file ../data/hypothetical_chemical_space_iter_5.csv --train_file ../data/data_smiles_curated.csv --new_data_file ../data/data_augmentation.csv
python analyze_data.py --iter 6 --raw_data ../data/raw_data_round_6  --pool_file ../data/hypothetical_chemical_space.csv
python baseline_models.py --final_cv --input_file ../data/final_overview_data.csv --csv_file ../data/data_smiles_curated_6.csv
