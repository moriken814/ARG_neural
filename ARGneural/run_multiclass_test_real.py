import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.transformer")

HOME = Path.home()
CWD = Path.cwd()

from parse_csv import parse_csv
from dictionary import mechanism_labels, drug_labels, species_labels
from class_dataset_amino import FastaDataset

# please choose from 4 models
from neuralnetwork import Encoder_for_classification
# from neuralnetwork_onlyseq import Encoder_for_classification
# from neuralnetwork_nodrug import Encoder_for_classification
# from neuralnetwork_nospecies import Encoder_for_classification

# 1. In case of random data
# print("data:HMD-ARG")
# FILEPATH_TEST = CWD / f"ARG-BERT_DNN/outputs/Prediction results/HMDARG-DB/fold_5.test.csv"
# OUTPUT_FILEPATH = CWD / f"ARGneural/drug_and_mechanism_real.png"

# 2. In case of low homology data
THRESHOLD = 0.4
print(f"data:{THRESHOLD}")
FILEPATH_TEST = CWD / f"ARG-BERT_DNN/outputs/Prediction results/LHD/c{THRESHOLD}/fold_5_{THRESHOLD}.test.csv"
OUTPUT_FILEPATH = CWD / f"ARGneural/drug_and_mechanism_real_{THRESHOLD}.png"

job_id = os.environ.get("JOB_ID")
MODELPATH = CWD / "model"
RESULTS_DIR = CWD / f"results/{job_id}"

# Calculate metrics
def calculate_metrix(predicted_labels, true_labels, num_classes, train):
    if train == True:
        Train_or_valid = 'train'
    else:
        Train_or_valid = 'valid'
    
    # Convert to NumPy array
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average=None, zero_division=0, labels=range(num_classes))
    recall = recall_score(true_labels, predicted_labels, average=None, zero_division=0, labels=range(num_classes))
    f1 = f1_score(true_labels, predicted_labels, average=None, zero_division=0, labels=range(num_classes))
    macro_f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)
    cm = confusion_matrix(true_labels, predicted_labels, labels=range(num_classes))

    # Output results
    keep_correct = 0
    keep_total = 0
    keep_f1 = 0
    keep_recall = 0
    keep_precision = 0
    TP = 0
    FN = 0
    FP = 0 
    TN = 0
    print(f'{Train_or_valid} Acc: {accuracy:.4f}, {Train_or_valid} MacroF1: {macro_f1:.4f}')
    for i in range(num_classes):
        class_total = (true_labels == i).sum()
        class_correct = ((true_labels == i) & (predicted_labels == i)).sum()
        print(f'---Class {i}:')
        print(f'   {class_correct}/{class_total} ({class_correct/class_total*100:.2f}%), '
            f'precision: {precision[i]:.4f}, recall: {recall[i]:.4f}, F1: {f1[i]:.4f}')
        
        if i != 6:   # Keep statistics for classes other than 'negative' (class 6)
            keep_correct += class_correct
            keep_total += class_total
            keep_f1 += f1[i]
            keep_recall += recall[i]
            keep_precision += precision[i]
            class_predicted_negative = ((true_labels == i) & (predicted_labels == 6)).sum()
            FN += class_predicted_negative
            TP += class_total - class_predicted_negative
        
        elif i == 6:   # If class 6: negative
            TN += class_correct
            FP += class_total - class_correct
            
    print(f"\nTP:{TP}, FN:{FN}, TN:{TN}, FP:{FP}")
    two_class_recall = TP / (FN + TP)
    two_class_precision = TP / (TP + FP)
    two_class_f1 = 2 * two_class_recall * two_class_precision / (two_class_recall + two_class_precision)
    six_class_f1 = keep_f1 / 6
    six_class_recall = keep_recall / 6
    six_class_precision = keep_precision / 6

    print(f'---{Train_or_valid} 2 Classes (positive or negative):')
    print(f'   {TP + TN}/{TP + FN + FP + TN} ({(TP + TN)/(TP + FN + FP + TN)*100:.2f}%), F1: {two_class_f1:.4f}, recall: {two_class_recall:.4f}, precision: {two_class_precision:.4f}, ')

    print(f'\n---{Train_or_valid} 6 Classes (mechanism classification (without negative)):')
    print(f'   {keep_correct}/{keep_total} ({keep_correct/keep_total*100:.2f}%), F1: {six_class_f1:.4f}, recall: {six_class_recall:.4f}, precision: {six_class_precision:.4f}, ')

    print(f"\n{Train_or_valid} Confusion Matrix:")
    print(cm)
    print("\n")
    
    return accuracy, two_class_f1, six_class_f1


def main(args):
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    
    print(batch_size, 'is batch size')
    print(num_epochs, 'is num epochs')
    
    # read data
    test_result = parse_csv(FILEPATH_TEST) 
    print(f"\ntest entries: {len(test_result)}")
    # result:[drug_class(int), mechanism(int), species(list), input_nums(list)]
    
    num_output_classes = len(mechanism_labels)
    print(f"\nnum_output_classes:{num_output_classes}")   # num of mechanism_labels :7 (including negative)
    
    test_dataset = FastaDataset(test_result)

    validation_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8, drop_last=False, shuffle=False)


    ### Training ###
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available(): 
        print("[Warning] CUDA is Not Available!!!!!")
    
    # model settings
    model = Encoder_for_classification()
    model.to(device)

    # Set loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00002)
    
    # Load the saved parameter dictionary
    PARAMETER_PATH = CWD / f"model/model_XXXX.pth"
    checkpoint = torch.load(PARAMETER_PATH)
    
    try:   # Load the model state
        model.load_state_dict(checkpoint['model'])
    except RuntimeError as e:
        print(f"RuntimeError during model loading: {e}")
        # Load with initialization for missing model weights
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
    
    try:
        optimizer.load_state_dict(checkpoint['optimizer'])
    except ValueError as e:
        print(f"ValueError during optimizer loading: {e}")
        
        # Get current optimizer's state_dict
        optimizer_state_dict = optimizer.state_dict()   # current
        loaded_state_dict = checkpoint['optimizer']   # loaded

        # List keys that exist in the loaded state_dict but not in the current optimizer
        missing_keys = [k for k in loaded_state_dict['state'] if k not in optimizer_state_dict['state']]
        unexpected_keys = [k for k in optimizer_state_dict['state'] if k not in loaded_state_dict['state']]

        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
        
        # Add missing parameters to the current optimizer from the loaded state_dict
        optimizer_state_dict['state'].update({k: v for k, v in loaded_state_dict['state'].items() if k in optimizer_state_dict['state']})
        # optimizer_state_dict['state'].update(loaded_state_dict['state'])
        
        # Reload the updated state_dict into the optimizer
        optimizer.load_state_dict(optimizer_state_dict)


    # Training loop
    for epoch in range(num_epochs):
        
        # validation loop
        model.eval()  # evaluation mode (dropout = 0)
        val_loss = 0
        val_metrics = defaultdict(float)
        valid_class_correct = [0] * num_output_classes
        valid_class_total = [0] * num_output_classes
        
        valid_all_predictions = []
        valid_all_labels = []
                
        with torch.no_grad():
            for batch in validation_loader:
                sequences = batch["input"].to(device)
                labels = batch["mechanism"].to(device)
                drug_classes = batch["drug_class"].to(device)
                species = batch["species"].to(device)
                # print("validation input:")
                # print(sequences[0])
                # print("validation label:")
                # print(labels[0])
                
                outputs = model(sequences, drug_classes, species)
                # print("validation output:")
                # print(outputs[0])
                # _, preds = torch.max(outputs, dim=1)
                # print(preds[0])
                
                # Convert output like [0.1, 0.1, 0.5, ...] -> class index [2]
                _, predicted = torch.max(outputs, 1)
                valid_all_predictions.extend(predicted.cpu().numpy())
                valid_all_labels.extend(labels.cpu().numpy())


        # Calculate metrics at the end of (test) epoch
        valid_accuracy, valid_two_class_f1, valid_six_class_f1 = calculate_metrix(valid_all_predictions, valid_all_labels, num_output_classes, train=False)

        # Count the number of occurrences for each drug class
        drug_counts = defaultdict(int)
        for drug in validation_loader.dataset.drug_class:
            drug_counts[drug] += 1

        # Sort by occurrence and group all but top 14 as "others"
        sorted_drugs = sorted(drug_counts.items(), key=lambda x: x[1], reverse=True)
        top_14_drugs = set(drug for drug, _ in sorted_drugs[:14])

        # Record mapping between drug classes and resistance mechanisms (including grouping as "others")
        drug_mechanism_counts = defaultdict(lambda: [0] * len(mechanism_labels))

        for pred, drug in zip(validation_loader.dataset.mechanisms, validation_loader.dataset.drug_class):
            if pred != 6:  # Count predictions excluding those predicted as "negative"
                if drug in top_14_drugs:
                    drug_mechanism_counts[drug][pred] += 1
                else:
                    drug_mechanism_counts['others'][pred] += 1

        # Color mapping for mechanism labels: brown, blue, orange, green, red, purple, gray
        colors = ['brown', '#1D6AB5', '#FF871B', '#1DBD23', '#DF3C25', 'purple', 'gray']
        mechanism_color_map = {mech: color for mech, color in zip(mechanism_labels.values(), colors)}

        # Create the bar graph
        fig, ax = plt.subplots(figsize=(16, 8))

        # Create display labels for drug classes
        display_drug_labels = {k: v for k, v in drug_labels.items() if v in top_14_drugs}
        # Replace 'macrolide-lincosamide-streptogramin' with 'MLS'
        display_drug_labels = {'MLS' if k == 'macrolide-lincosamide-streptogramin' else k: v 
                            for k, v in display_drug_labels.items()}
        display_drug_labels = dict(sorted(display_drug_labels.items(), key=lambda x: x[0].lower()))  # Sort alphabetically
        display_drug_labels['others'] = len(display_drug_labels)

        y_pos = np.arange(len(display_drug_labels))
        left = np.zeros(len(display_drug_labels))

        for mech in mechanism_labels.values():
            counts = []
            for drug_name, drug_id in display_drug_labels.items():
                if drug_name == 'others':
                    counts.append(drug_mechanism_counts['others'][mech])
                else:
                    counts.append(drug_mechanism_counts[drug_id][mech])
                    
            ax.barh(y_pos, counts, left=left, color=mechanism_color_map[mech], 
                    label=list(mechanism_labels.keys())[mech])
            left += counts

        ax.set_yticks(y_pos)
        ax.set_yticklabels(list(display_drug_labels.keys()), fontsize=24)  
        ax.invert_yaxis()
        # ax.set_xlabel('Number of predictions')
        ax.tick_params(axis='x', labelsize=24)
        # ax.set_title('Predicted Resistance Mechanisms by Drug Class')
        ax.legend(loc='lower right')

        plt.tight_layout()
        plt.savefig(OUTPUT_FILEPATH)
        plt.close()
        print("plot is saved.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num_epochs', default=1, type=int)
    parser.add_argument('-b', '--batch_size', default=1, type=int)
    parser.add_argument('--tag', default=None)
    
    args = parser.parse_args()
    
    print(args)
    
    main(args)