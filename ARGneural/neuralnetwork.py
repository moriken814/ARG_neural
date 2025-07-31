import torch
import torch.nn as nn

from dictionary import mechanism_labels, drug_labels, species_labels

class Encoder_for_classification(nn.Module):
    def __init__(self, d_model=128, dropout_rate=0.2):
        super().__init__()
        self.d_model = d_model
        
        self.num_output_classes = len(mechanism_labels)     # num of mechanism_labels :7 (including negative)
        self.num_drug_classes = len(drug_labels)            # num of drug_labels
        self.num_species_classes = len(species_labels) + 1  # num of species_labels
        
        self.drug_embedding = nn.Embedding(self.num_drug_classes, d_model)  # drug_class embedding layer
        self.species_embedding = nn.Embedding(self.num_species_classes, d_model)  # species_class embedding layer
        
        # Normalize
        self.layer_norm_src = nn.LayerNorm(512)
        self.layer_norm_drug = nn.LayerNorm(d_model)
        self.layer_norm_species = nn.LayerNorm(d_model)
        
        self.fc1 = nn.Linear(768, 512)
        # self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 128)
        # self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, 32)
        # self.dropout3 = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(32, self.num_output_classes)
        
        self.relu = nn.ReLU()
        # self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, src, drug_class, species):
        device = src.device
        
        # print(f"self.num_output_classes:{self.num_output_classes}")
        # print(f"self.num_drug_classes:{self.num_drug_classes}")
        # print(f"self.num_species_classes:{self.num_species_classes}")
        
        # Normalize src
        src_normalized = self.layer_norm_src(src)  # [Batch, 512]

        # Embed and normalize drug class
        drug_tensor = self.drug_embedding(drug_class)        # [Batch, d_model]
        drug_normalized = self.layer_norm_drug(drug_tensor)  # [Batch, d_model]
        
        # Embed and sum species information -> normalize
        species_tensor = self.species_embedding(species)           # [Batch, 7, d_model]
        species_sum = species_tensor.sum(dim=1)                    # [Batch, d_model]
        species_normalized = self.layer_norm_species(species_sum)
        
        # Concatenate normalized vectors
        combined_input = torch.cat((src_normalized, drug_normalized, species_normalized), dim=1)  # [Batch, 768(512 + 2*d_model)]

        # Reduce dimensions through fully connected layers
        out = self.fc1(combined_input)
        out = self.relu(out)
        # out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.relu(out)
        # out = self.dropout2(out)

        out = self.fc3(out)
        out = self.relu(out)
        # out = self.dropout3(out)

        out = self.fc_out(out)  # [Batch, num_output_classes]
        out = self.softmax(out)
        
        return out
