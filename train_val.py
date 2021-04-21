from torch.utils.data import random_split
import torch
import numpy as np
from DataloaderClass import OCTDataset

#need to call dtaloader class nd split into train, val and test

def train_val_test_split(h5_file, train_per = 7, val_per = 2, test_per = 1,seed = 42):
    
    octdataset = OCTDataset(h5_file)
    trainloader,validloader,testloader = random_split(octdataset,
                                                      [train_per, val_per, test_per],
                                                      torch.Generator().manual_seed(seed))
    
    return trainloader,validloader,testloader
    



def train_val(model,trainloader,validloader,criterion, optimizer, epochs = 5):
    
    #criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
    min_valid_loss = np.inf

    for e in range(epochs):
        train_loss = 0.0
        model.train()     # Optional when not using Model Specific layer
        for data, labels in trainloader:
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



print(torch.cuda.is_available()) 
print(torch.cuda.current_device())           