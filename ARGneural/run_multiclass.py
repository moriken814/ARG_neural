import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time
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
print("data:HMD-ARG")
FOLD = 5
print(f"fold:{FOLD}")
FILEPATH_TRAIN = CWD / f"ARG-BERT_DNN/outputs/Prediction results/HMDARG-DB/fold_{FOLD}.train.csv"
FILEPATH_TEST = CWD / f"ARG-BERT_DNN/outputs/Prediction results/HMDARG-DB/fold_{FOLD}.test.csv"

# 2. In case of low homology data
# THRESHOLD = 0.4
# print(f"data:{THRESHOLD}")
# FOLD = 5
# print(f"fold:{FOLD}")
# FILEPATH_TRAIN = CWD / f"ARG-BERT_DNN/outputs/Prediction results/LHD/c{THRESHOLD}/fold_{FOLD}_{THRESHOLD}.train.csv"
# FILEPATH_TEST = CWD / f"ARG-BERT_DNN/outputs/Prediction results/LHD/c{THRESHOLD}/fold_{FOLD}_{THRESHOLD}.test.csv"

job_id = os.environ.get("JOB_ID")
MODELPATH = CWD / "model"
RESULTS_DIR = CWD / f"results/{job_id}"

# Visualize loss
def show_loss(train_losses, validation_losses, filepath):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(validation_losses, label='valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(filepath)
    
# Visualize accuracy
def show_accuracy(train_accuracies, test_accuracies, filepath):
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='valid Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(filepath)
    
# Visualize F1 score
def show_f1(train_f1s, val_f1s, filepath):
    plt.figure(figsize=(10, 5))
    plt.plot(train_f1s, label='Train F1 score')
    plt.plot(val_f1s, label='valid F1 score')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()
    plt.savefig(filepath)

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
    train_result = parse_csv(FILEPATH_TRAIN) 
    test_result = parse_csv(FILEPATH_TEST) 
    print(f"\ntrain entries: {len(train_result)}")
    print(f"\ntest entries: {len(test_result)}")
    # result:[drug_class(int), mechanism(int), species(list), input_nums(list)]
    
    num_output_classes = len(mechanism_labels)
    print(f"\nnum_output_classes:{num_output_classes}")   # num of mechanism_labels :7 (including negative)
    
    train_dataset = FastaDataset(train_result)
    test_dataset = FastaDataset(test_result)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, drop_last=False, shuffle=True)
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
    class_weights = torch.tensor([10.0, 1.0, 1.0, 1.0, 5.0, 5.0, 1.0]).to(device)  # Set higher weight for minority classes
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00002)

    # Lists to record loss and accuracy
    train_losses = []
    validation_losses = []
    train_accuracies = []
    val_accuracies = []
    train_2_f1s = []
    train_6_f1s = []
    val_2_f1s = []
    val_6_f1s = []

    # Training loop
    for epoch in range(num_epochs):
        
        if (epoch) % 100 == 0:
            if torch.cuda.is_available(): 
                torch.cuda.synchronize()
            start = time.time()
            print("Calculation Time...")
            
        model.train()
        train_loss = 0
        train_all_predictions = []
        train_all_labels = []
        
        for batch in train_loader:
            sequences = batch["input"].to(device)
            labels = batch["mechanism"].to(device)
            drug_classes = batch["drug_class"].to(device)
            species = batch["species"].to(device)
            # print("train input:")
            # print(f"size:{len(sequences[0])}")
            # print(sequences[0])
            # print("train label:")
            # print(labels[0])
            
            optimizer.zero_grad()
            outputs = model(sequences, drug_classes, species)  # [batch, num of mechanism_labels]
            # print("train output:")
            # print(f"size:{outputs[0].size()}")
            # print(outputs[0])
            # _, preds = torch.max(outputs, dim=1)
            # print(preds[0])
            
            # calculate loss
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()  # Convert from Tensor to Python float
            
            # Convert output like [0.1, 0.1, 0.5, ...] -> class index [2]
            _, predicted = torch.max(outputs, 1)
            train_all_predictions.extend(predicted.cpu().numpy())
            train_all_labels.extend(labels.cpu().numpy())

        
        # validation loop
        model.eval()  # evaluation mode (dropout = 0)
        val_loss = 0
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

                # calculate loss
                loss = criterion(outputs, labels.long())
                val_loss += loss.item()
                
                # Convert output like [0.1, 0.1, 0.5, ...] -> class index [2]
                _, predicted = torch.max(outputs, 1)
                valid_all_predictions.extend(predicted.cpu().numpy())
                valid_all_labels.extend(labels.cpu().numpy())
        
        
        # Calculate average loss
        train_loss /= len(train_loader)
        val_loss /= len(validation_loader)
        
        # Output results
        print(f'\nEpoch [{epoch+1}/{num_epochs}], '
            f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        
        # Calculate metrics after epoch (train)
        train_accuracy, train_two_class_f1, train_six_class_f1 = calculate_metrix(train_all_predictions, train_all_labels, num_output_classes, train=True)
        
        # Calculate metrics after epoch (test)
        valid_accuracy, valid_two_class_f1, valid_six_class_f1 = calculate_metrix(valid_all_predictions, valid_all_labels, num_output_classes, train=False)

        # Record epoch results
        train_losses.append(train_loss)
        validation_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(valid_accuracy)
        train_2_f1s.append(train_two_class_f1)
        val_2_f1s.append(valid_two_class_f1)
        train_6_f1s.append(train_six_class_f1)
        val_6_f1s.append(valid_six_class_f1) 

        
        if (epoch) % 100 == 0:
            if torch.cuda.is_available():  
                torch.cuda.synchronize()
            elapsed_time = time.time() - start
            print('cur_batch:', elapsed_time, 'sec.', flush=True)
            print('ETA', elapsed_time * (epoch+1) // 3600, "h/", elapsed_time * num_epochs// 3600, 'h.')

        if (epoch+1) % 100 == 0:
            # Save model
            state_dict = {
                "model": model.cpu().state_dict(),
                "optimizer": optimizer.state_dict()
            }
            
            MODELPATH.mkdir(exist_ok=True)            
            torch.save(state_dict, MODELPATH / f"model_{job_id}.pth")
            print("save the model.")
            
            # from cpu to gpu
            model.to(device)

    # Visualize loss
    RESULTS_DIR.mkdir(exist_ok=True)
    show_loss(train_losses, validation_losses, filepath=RESULTS_DIR / f"loss_{job_id}")
    show_accuracy(train_accuracies, val_accuracies, filepath=RESULTS_DIR / f"acc_{job_id}")
    show_f1(train_2_f1s, val_2_f1s, filepath=RESULTS_DIR / f"f1_2class_{job_id}")
    show_f1(train_6_f1s, val_6_f1s, filepath=RESULTS_DIR / f"f1_6class_{job_id}")

   
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num_epochs', default=1, type=int)
    parser.add_argument('-b', '--batch_size', default=1, type=int)
    parser.add_argument('--tag', default=None)
    
    args = parser.parse_args()
    
    print(args)
    
    main(args)