from pathlib import Path
import csv

from dictionary import mechanism_labels, drug_labels, species_labels

HOME = Path.home()
CWD = Path.cwd()

FILEPATH = CWD / f"ARG-BERT_DNN/outputs/Prediction results/HMDARG-DB/fold_5.test.csv"

def parse_csv(file_path):
    result = []
    
    with open(file_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            # Quantify drug
            drug = row['target']
            drug_int = drug_labels.get(drug, 0)
            
            # Quantify mechanism
            mechanism = row['mechanism']
            mechanism_int = mechanism_labels.get(mechanism, 0)
            
            # Quantify species
            species = row['species']
            species_parts = species.split(';')
            species_nums = []
            for part in species_parts:
                species_nums.append(species_labels.get(part, 0))
            species_nums += [0] * (7 - len(species_nums))  # Pad with zeros to ensure 7 numbers
            species_nums = species_nums[:7]  # Truncate if there are more than 7 numbers
            
            # Get proteinBERT output
            input_nums = []
            for i in range(512):
                try:
                    input_nums.append(float(row[str(i)]))
                except ValueError:
                    input_nums.append(0)
            
            result.append((drug_int, mechanism_int, species_nums, input_nums))

    return result

# result = parse_csv(FILEPATH)

# print(f"result:{result}")
# print(f"result[0][0]:{result[0][0]}")
# print(f"result[0][1]:{result[0][1]}")