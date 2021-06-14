from numpy.core.fromnumeric import shape
from torch.utils.data import random_split
from tqdm import tqdm
import torch
from torchvision import transforms
import numpy as np
import torch.nn as nn
import sys
import pandas as pd
import time
from torch.utils.tensorboard import SummaryWriter
from DataloaderClassV2 import OCTdatasetV2
import Network



#need to call dtaloader class nd split into train, val and test

def train_val(model,train_loader,valid_loader,criterion, optimizer, epochs = 1,plot = False):
    if plot is True:
        writer_train = SummaryWriter('runs/Siamese_net_experiment_train_1')
        writer_val = SummaryWriter('runs/Siamese_net_experiment_val_1')
    #criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
    min_valid_loss = np.inf
    i = 0
    running_t_loss = 0.0
    running_v_loss = 0.0
    for e in range(epochs):
        train_loss = 0.0
        model.train()     # Optional when not using Model Specific layer

        for init_data,datum,groundtruth in tqdm(train_loader):



            #init_data = init_data.unsqueeze(0)
            #print(index)

            #data = data.unsqueeze(0)
            if torch.cuda.is_available():
                init_data ,datum, groundtruth = init_data.cuda(), datum.cuda(), groundtruth.cuda()
            print(shape(init_data),shape(datum))
            optimizer.zero_grad()
            #print(type(init_data),type(data))
            target = model(init_data,datum)
            #print(shape(target),shape(groundtruth[0]))
            groundtruth = groundtruth.double()

            #print(traininit.shape,data.shape)
            #print(target.dtype,groundtruth.dtype)
            t_loss = criterion(target,groundtruth)
            t_loss.backward()
            optimizer.step()
            if plot == True:
                running_t_loss += t_loss.item()
                if i % 100 == 99:    # every 1000 mini-batches...
                    # ...log the running loss
                    writer_train.add_scalar('training loss',
                            running_t_loss / 100,
                            epochs * len(train_loader) + i)
                    running_t_loss = 0.0

            train_loss = t_loss.item() * datum.size(0)

        valid_loss = 0.0
        model.eval()     # Optional when not using Model Specific layer
        for init_data,datum,groundtruth in tqdm(valid_loader):


            #init_data = init_data.unsqueeze(0)
            #data = oct_data[index]
            #data = data.unsqueeze(0)
            if torch.cuda.is_available():
                init_data ,datum, groundtruth = init_data.cuda(), datum.cuda(), groundtruth.cuda()

            target = model(init_data,datum)
            groundtruth = groundtruth.double()
            v_loss = criterion(target,groundtruth)
            if plot == True:
                running_v_loss += v_loss.item()
                if i % 100 == 99:    # every 1000 mini-batches...
                    # ...log the running loss
                    writer_val.add_scalar('Validation loss',
                            running_v_loss / 100,
                            epochs * len(valid_loader) + i)
                    running_v_loss = 0.0

            valid_loss = v_loss.item() * datum.size(0)

        print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(train_loader)} \t\t Validation Loss: {valid_loss / len(valid_loader)}')
        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            # Saving State Dict
            timestr = time.strftime("%Y%m%d-%H%M%S")
            torch.save(model.state_dict(), '/models/saved_model_'+timestr+'.pth')









#tensorboard --logdir=runs
def main():


    train_data = OCTdatasetV2(train= True)
    valid_data = OCTdatasetV2(valid= True)
    train_loader = torch.utils.data.DataLoader(dataset= train_data, batch_size= 1,shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset = valid_data, batch_size= 1)


    model = Network.generate_model()
    model = model.double()
    #if path != 'None':
    #   model.load_state_dict(torch.load(path))
    if torch.cuda.is_available():
        model = model.cuda()

    #epochs = int(sys.argv[4])
    epochs = 1
    #lr = float(sys.argv[5])
    lr = 0.0001
    #print(lr.dtype)
    #criterion = nn.L1Loss()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    #optimizer = torch.optim.Adam(model.parameters(),lr = lr)




    train_val(model = model,

            train_loader = train_loader,
            valid_loader = valid_loader,
            criterion= criterion,
            optimizer= optimizer,
            epochs= epochs,plot = False)


if __name__ == "__main__":
    main()





