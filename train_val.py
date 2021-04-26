from torch.utils.data import random_split
import torch
import numpy as np
from DataloaderClass import OCTDataset
import Network
import torch.nn as nn



#need to call dtaloader class nd split into train, val and test

def train_val_test_split(h5_file, train_per = 70, seed = 42):
    
    octdataset = OCTDataset(h5_file,train= True)
    
    length = len(octdataset)
    train_len = int(length*train_per/100)
    val_len = length-train_len
     
    octdata = octdataset[:][0]    
    train_data,val_data = random_split(octdata,
                                       [train_len, val_len],
                                       torch.Generator().manual_seed(seed))
    
    return train_data,val_data
    

    


def train_val(model,trainloader,validloader,criterion, optimizer, epochs = 5):
    
    #criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
    min_valid_loss = np.inf

    for e in range(epochs):
        train_loss = 0.0
        model.train()     # Optional when not using Model Specific layer
        for data in trainloader:
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()
            
            optimizer.zero_grad()
            target = model(data)
            loss = criterion(target,labels)
            loss.backward()
            optimizer.step()
            train_loss = loss.item() * data.size(0)
        
        valid_loss = 0.0
        model.eval()     # Optional when not using Model Specific layer
        for data, labels in validloader:
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()
            
            target = model(data)
            loss = criterion(target,labels)
            valid_loss = loss.item() * data.size(0)

        print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(trainloader)} \t\t Validation Loss: {valid_loss / len(validloader)}')
        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            # Saving State Dict
            torch.save(model.state_dict(), 'saved_model.pth')



file = "/home/Mukherjee/Data/Cross_ext.h5"

train,valid = train_val_test_split(h5_file = file,
                                   train_per = 70,                                        
                                   seed = 42)
train_loader = torch.utils.data.DataLoader(dataset = train, batch_size= 1)
valid_loader = torch.utils.data.DataLoader(dataset = valid, batch_size= 1)


model = Network.generate_model() 

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

train_val(model = model, 
          trainloader = train_loader, 
          validloader = valid_loader, 
          criterion= criterion, 
          optimizer= optimizer,
          epochs= 1)


        
      