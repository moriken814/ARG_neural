import torch
from torch.utils.data import Dataset
from pathlib import Path

from parse_csv import parse_csv

HOME = Path.home()
CWD = Path.cwd()

FILEPATH_TRAIN = CWD / f"ARG-BERT_DNN/outputs/Prediction results/HMDARG-DB/fold_5.train.csv"
FILEPATH_TEST = CWD / f"ARG-BERT_DNN/outputs/Prediction results/HMDARG-DB/fold_5.test.csv"


class FastaDataset(Dataset):
    def __init__(self, metadata):
        super().__init__()
        # metadata: [drug_class(int), mechanism(int), species(list), input_nums(list)]
        self.drug_class = [meta[0] for meta in metadata]
        self.mechanisms = [meta[1] for meta in metadata]
        self.species = [meta[2] for meta in metadata]
        self.input_nums = [meta[3] for meta in metadata]  
        self.max_length = self.get_max_length()

    def get_max_length(self):
        return max(len(seq) for seq in self.input_nums)
    
    def __len__(self):
        return len(self.input_nums)

    def __getitem__(self, idx):
        input_sequence = self.input_nums[idx]
        # print(f"sequence:{sequence}")
        drug_class = self.drug_class[idx]
        # print(f"drug_class:{drug_class}")
        mechanism = self.mechanisms[idx]
        # print(f"mechanisms:{mechanisms}")
        species = self.species[idx]
        
        return {
            "input": torch.tensor(input_sequence, dtype=torch.float),
            "mechanism": torch.tensor(mechanism, dtype=torch.long),
            "drug_class": torch.tensor(drug_class, dtype=torch.long),
            "species": torch.tensor(species, dtype=torch.long)
        }


if __name__ == "__main__":
    train_result = parse_csv(FILEPATH_TRAIN)
    test_result = parse_csv(FILEPATH_TEST)
    
    print(f"\nTotal train entries: {len(train_result)}")
    print(f"\nTotal test entries: {len(test_result)}")

    # dataset = FastaDataset(result)
    # torch.set_printoptions(threshold=float('inf'))
    # print(f"dataset[0]:{dataset[0]}")
    # print(f"dataset[1]:{dataset[1]}")
    # print(f"dataset[2]:{dataset[2]}")
    