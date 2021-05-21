from torch.utils.data import random_split
from tqdm import tqdm
import torch
from torchvision import transforms
import numpy as np
import torch.nn as nn
import sys
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from DataloaderClass import OCTDataset
import Network






#need to call dtaloader class nd split into train, val and test

def train_val_test_split(h5_loc, list_loc, train_per = 0.7, seed = 42,csv_present = True):

    if csv_present is False:
        data_list = pd.read_csv(list_loc,header=None)
        octdataset = OCTDataset(h5_loc,
                                transform = transforms.Compose([transforms.ToTensor()]),
                                train= True,
                                index_list = data_list)

        length = len(octdataset)
        train_len = int(length*train_per)
        val_len = length-train_len

        oct_data = octdataset[:]
    else:
        data_path = h5_loc[0:len(h5_loc)-2]+"csv"
        df_oct_data = pd.read_csv(data_path,header=None,memory_map=True)
        #oct_data = df_oct_data.to_numpy()
        oct_data = df_oct_data.values.tolist()
        length = len(oct_data)
        train_len = int(length*train_per)
        val_len = length-train_len




    #print(octdata[0].shape)
    train_data,val_data = random_split(oct_data,
                                      [train_len, val_len],
                                       torch.Generator().manual_seed(seed))



    return train_data,val_data





def train_val(model,trainloader,validloader,criterion, optimizer, epochs = 1,plot = False):
    if plot is True:
        writer = SummaryWriter('runs/Siamese_net_experiment_1')
    #criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
    min_valid_loss = np.inf
    i = 0
    running_loss = 0.0
    for e in range(epochs):
        train_loss = 0.0
        model.train()     # Optional when not using Model Specific layer

        for init_data, data, labels in tqdm(trainloader):
            i = i+1
            if torch.cuda.is_available():
                init_data ,data, labels = init_data.cuda(), data.cuda(), labels.cuda()

            optimizer.zero_grad()
            print(type(init_data),type(data))
            target = model(init_data,data)
            labels = labels.double()
            #print(traininit.shape,data.shape)
            #print(target.dtype,labels.dtype)
            t_loss = criterion(target,labels)
            t_loss.backward()
            optimizer.step()
            if plot == True:
                running_t_loss += t_loss.item()
                if i % 100 == 99:    # every 1000 mini-batches...
                    # ...log the running loss
                    writer.add_scalar('training loss',
                            running_t_loss / 100,
                            epochs * len(trainloader) + i)
                    running_t_loss = 0.0

            train_loss = t_loss.item() * data.size(0)


        valid_loss = 0.0
        model.eval()     # Optional when not using Model Specific layer
        for init_data, data, labels in tqdm(validloader):
            if torch.cuda.is_available():
                init_data ,data, labels = init_data.cuda(), data.cuda(), labels.cuda()

            target = model(init_data,data)
            labels = labels.double()
            v_loss = criterion(target,labels)
            if plot == True:
                running_v_loss += v_loss.item()
                if i % 100 == 99:    # every 1000 mini-batches...
                    # ...log the running loss
                    writer.add_scalar('Validation loss',
                            running_v_loss / 100,
                            epochs * len(trainloader) + i)
                    running_v_loss = 0.0

            valid_loss = v_loss.item() * data.size(0)

        print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(trainloader)} \t\t Validation Loss: {valid_loss / len(validloader)}')
        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            # Saving State Dict
            torch.save(model.state_dict(), 'saved_model.pth')






#tensorboard --logdir=runs
def main():
    file = sys.argv[1]
    data_list = sys.argv[2]
    train,valid = train_val_test_split(h5_loc = file,
                                       list_loc = data_list,
                                       train_per = 0.7,
                                       seed = 42,csv_present=False)

    train_loader = torch.utils.data.DataLoader(dataset= train, batch_size= 1,shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset = valid, batch_size= 1)


    model = Network.generate_model()
    model = model.double()
    if torch.cuda.is_available():
        model = model.cuda()
    epochs = int(sys.argv[3])
    lr = float(sys.argv[4])
    #print(lr.dtype)
    criterion = nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    #optimizer = torch.optim.Adam(model.parameters(),lr = lr)




    train_val(model = model,
            trainloader = train_loader,
            validloader = valid_loader,
            criterion= criterion,
            optimizer= optimizer,
            epochs= epochs,plot = True)


if __name__ == "__main__":
    main()





