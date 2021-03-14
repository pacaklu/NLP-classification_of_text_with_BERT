import os 
import time
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import transformers
from transformers import BertModel, BertTokenizer, AdamW

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

import seaborn as sns

class quora_dataset(Dataset):
    """
    pytorch DS
    """
    def __init__(self, data, trained_tokenizer, MAX_LEN, n_classes):
        self.data = data #pd dataframe
        self.max_len = MAX_LEN
        self.tokenizer = trained_tokenizer
        self.n_classes = n_classes

    
    def __len__(self):
        return self.data.shape[0]
        
    def __getitem__(self, index):
        """
        Bert need as input list of token's ids
        and attention mask.
        """
        current_row = self.data.iloc[index]
        
        encoding = self.tokenizer.encode_plus(
                        current_row['question_text'],
                        max_length = self.max_len,
                        add_special_tokens = True,
                        pad_to_max_length = True,
                        return_attention_mask = True,
                        return_token_type_ids = False,
                        return_tensors = 'pt')
            
        ids = encoding['input_ids']
        mask = encoding['attention_mask']

 
        target = np.zeros(self.n_classes)
        target[current_row['target']] = 1

        return ids, mask, target





class network(nn.Module):
    """
    Design of our network
    """
    
    def __init__(self, n_classes, PRE_TRAINED_MODEL_NAME):
        super(network, self).__init__()
        self.bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.linear = nn.Linear(self.bert_model.config.hidden_size, n_classes)

            
    def forward(self, input_ids, attention_mask):
        input_ids = torch.squeeze(input_ids)
        attention_mask = torch.squeeze(attention_mask)
        _, output = self.bert_model(input_ids = input_ids, attention_mask = attention_mask, return_dict=False)
        #in the first output, it returns the output of encoders, that we dont nee
        #in the second output, it returns the output of final layer (usually 768 size, that we need) - that are features


        output = self.linear(output)

        return output     




def number_of_tokens(tokenizer, column):
    """
    Plots distribution of tokens
    """
    
    token_lens = []

    for txt in column:
        tokens = tokenizer.encode(txt, max_length=512, truncation = True)
        token_lens.append(len(tokens))

    sns.histplot(token_lens)
    plt.xlim([0, np.max(token_lens)]);
    plt.xlabel('Token count');

    print(f'99 pct quantile: {np.quantile(token_lens, 0.99)}')
    print(f'Mean: {np.mean(token_lens)}')
    print(f'Median: {np.median(token_lens)}')
    print(f'Max: {np.max(token_lens)}')


def train_model(model, criterion, optimizer, num_epochs, n_classes,
               train_loader, valid_loader, train_data, valid_data, device):
    """
    Function that defines what should be done in training and evaluation phases
    """

    train_loss = []
    valid_loss = []
    train_acc = []
    valid_acc = []
    
    for epoch in range(num_epochs):
        print('-----------------------------------')
        print(f'Epoch {epoch}/{num_epochs}')

        
        model.train() 
        #Tells the model to use train mode
        #Dropout layer behaves differently for train/eval phases
        
        actual_loss = 0
        num_corrects = 0  
        
        true_positive = 0
        false_positive = 0
        false_negative = 0
        
        for x in train_loader:
            
            optimizer.zero_grad()
            # otherwise by calling loss.backward() gradient of parameters would be summed
            
            inputs = x[0].to(device, dtype=torch.long)
            mask = x[1].to(device, dtype=torch.long)
            labels = x[2].to(device, dtype=torch.float)
                
            outputs = model(input_ids = inputs, attention_mask = mask)


              
            
            loss = criterion(outputs, labels.argmax(dim=1))     
            # creates graph of parameters, is connected to model throught outputs

            loss.backward()  #computes gradient of loss with respect to the parameters
            
                    
            optimizer.step() #updates models parameters
            
            # it is possible to put optimizer.step and optimizer.zero(grad) out of batch for with slower conv
            
            actual_loss += loss.item() * inputs.size(0) #sum of losses for given batch
            num_corrects += torch.sum(outputs.argmax(dim=1) == labels.argmax(dim=1)).item() 
            
            true_positive += torch.sum(((outputs.argmax(dim=1) == 1) &  (labels.argmax(dim=1) == 1))).item() 
            false_positive += torch.sum(((outputs.argmax(dim=1) == 1) &  (labels.argmax(dim=1) == 0))).item() 
            false_negative += torch.sum(((outputs.argmax(dim=1) == 0) &  (labels.argmax(dim=1) == 1))).item() 
            
            
        train_loss.append(actual_loss / len(train_data)) 
        train_acc.append(num_corrects / len(train_data))
        
        precision = true_positive/(true_positive + false_positive)
        recall = true_positive/(true_positive + false_negative)
        F1 = 2*(precision*recall)/(precision + recall)
        
        print('Train stats:')
        print(f'Loss: {round(train_loss[epoch],6)}') 
        print(f'Acc: {round(train_acc[epoch],6)}')
        print(f'Num_corrects: {round(num_corrects,6)}/ {len(train_data)} ')
        print(f'Precision: {round(precision,6)}') 
        print(f'Recall: {round(recall,6)}')
        print(f'F1: {round(F1,6)}')
        
        
        #VALIDATION_PHASE
        model.eval()
        with torch.no_grad():  # is it necessary?

            valid_actual_loss = 0
            valid_num_corrects = 0
            
            true_positive = 0
            false_positive = 0
            false_negative = 0
            
            for x in valid_loader:

                inputs = x[0].to(device, dtype=torch.long)
                mask = x[1].to(device, dtype=torch.long)
                labels = x[2].to(device, dtype=torch.float)
                
                outputs = model(input_ids = inputs, attention_mask = mask)
             
                loss = criterion(outputs, labels.argmax(dim=1))
                          
                valid_actual_loss += loss.item() * inputs.size(0)
                valid_num_corrects += torch.sum(outputs.argmax(dim=1) == labels.argmax(dim=1)).item()
                
                true_positive += torch.sum(((outputs.argmax(dim=1) == 1) &  (labels.argmax(dim=1) == 1))).item() 
                false_positive += torch.sum(((outputs.argmax(dim=1) == 1) &  (labels.argmax(dim=1) == 0))).item() 
                false_negative += torch.sum(((outputs.argmax(dim=1) == 0) &  (labels.argmax(dim=1) == 1))).item()  
            
                
            
            valid_loss.append(valid_actual_loss / len(valid_data))
            actual_acc = valid_num_corrects / len(valid_data)
            valid_acc.append(valid_num_corrects / len(valid_data))
            
            precision = true_positive/(true_positive + false_positive)
            recall = true_positive/(true_positive + false_negative)
            F1 = 2*(precision*recall)/(precision + recall)
            
            print('')
            print('Valid stats:')
            print(f'Loss: {round(valid_loss[epoch],6)}') 
            print(f'Acc: {round(valid_acc[epoch],6)}')
            print(f'Num_corrects: {round(valid_num_corrects,6)}/ {len(valid_data)} ')
            print(f'Precision: {round(precision,6)}') 
            print(f'Recall: {round(recall,6)}')
            print(f'F1: {round(F1,6)}')
            
    
    
    fig, axs = plt.subplots(2, figsize=(14,8))
    axs[0].plot(train_loss)
    axs[0].plot(valid_loss)
    axs[0].set_title('blue = Train Loss, orange = Valid Loss')

    axs[1].plot(train_acc)
    axs[1].plot(valid_acc)
    axs[1].set_title('Blue = Train Accuracy, orange = Valid Accuracy')
    plt.show()