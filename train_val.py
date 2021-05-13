from torch.utils.data import random_split
from tqdm import tqdm
import torch
from torchvision import transforms
import numpy as np
import torch.nn as nn
import sys
from torch.utils.tensorboard import SummaryWriter
from DataloaderClass import OCTDataset
import Network



writer = SummaryWriter('runs/Siamese_net_experiment_1')


#need to call dtaloader class nd split into train, val and test

def train_val_test_split(h5_file, train_per = 0.7, seed = 42):

    octdataset = OCTDataset(h5_file,transform = transforms.Compose([transforms.ToTensor()]),train= True)

    length = len(octdataset)
    train_len = int(length*train_per)
    val_len = length-train_len

    oct_data = octdataset[:]
    oct_data_init = octdataset[:][0][0].reshape(1,1,64,64,64)



    #print(octdata[0].shape)
    train_data,val_data = random_split(oct_data,
                                      [train_len, val_len],
                                       torch.Generator().manual_seed(seed))

    all_data = oct_data

    return train_data,val_data,oct_data_init,all_data





def train_val(model,init_data,all_dataloader,trainloader,validloader,criterion, optimizer, epochs = 5,plot = False):

    #criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
    min_valid_loss = np.inf
    i = 0
    running_loss = 0.0
    for e in range(epochs):
        train_loss = 0.0
        model.train()     # Optional when not using Model Specific layer
        for init_data,init_labels in tqdm(all_dataloader):
            for data, labels in trainloader:
                i = i+1
                if torch.cuda.is_available():
                    init_data = init_data.cuda()
                    data, labels = data.cuda(), labels.cuda()

                optimizer.zero_grad()
                target = model(init_data,data)
                labels = labels.double()
                #print(traininit.shape,data.shape)
                #print(target.dtype,labels.dtype)
                loss = criterion(target,labels)
                loss.backward()
                optimizer.step()
                if plot == True:
                    running_loss += loss.item()
                    if i % 100 == 99:    # every 10 mini-batches...

                        # ...log the running loss
                        writer.add_scalar('training loss',
                                running_loss / 1000,
                                epochs * len(trainloader) + i)
                        running_loss = 0.0

                train_loss = loss.item() * data.size(0)


            valid_loss = 0.0
            model.eval()     # Optional when not using Model Specific layer
            for data, labels in validloader:
                if torch.cuda.is_available():
                    init_data = init_data.cuda()
                    data, labels = data.cuda(), labels.cuda()

                target = model(init_data,data)
                labels = labels.double()
                loss = criterion(target,labels)


                valid_loss = loss.item() * data.size(0)

            print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(trainloader)} \t\t Validation Loss: {valid_loss / len(validloader)}')
            if min_valid_loss > valid_loss:
                print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
                min_valid_loss = valid_loss
                # Saving State Dict
                torch.save(model.state_dict(), 'saved_model.pth')

#def visualization(model,trainloader,):



#file = "/home/Mukherjee/Data/Cross_ext.h5"

#train,valid,init,all_data = train_val_test_split(h5_file = file,
#                                   train_per = 0.7,
#                                   seed = 42)
#print(len(train),len(valid))
#train_loader = torch.utils.data.DataLoader(dataset= train, batch_size= 1)
#valid_loader = torch.utils.data.DataLoader(dataset = valid, batch_size= 1)
#all_loader = torch.utils.data.DataLoader(dataset = all_data, batch_size= 1)

#model = Network.generate_model()
#model = model.double()

#criterion = nn.L1Loss()
#optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
#optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)
#train_val(model = model,
#          init_data = init,
#          all_dataloader=all_loader,
#          trainloader = train_loader,
#          validloader = valid_loader,
#          criterion= criterion,
#          optimizer= optimizer,
#          epochs= 1,plot = True)
#tensorboard --logdir=runs
def main():
    file = sys.argv[1]
    train,valid,init,all_data = train_val_test_split(h5_file = file,
                                          train_per = 0.7,
                                          seed = 42)
    train_loader = torch.utils.data.DataLoader(dataset= train, batch_size= 1)
    valid_loader = torch.utils.data.DataLoader(dataset = valid, batch_size= 1)
    all_loader = torch.utils.data.DataLoader(dataset = all_data, batch_size= 1)

    model = Network.generate_model()
    model = model.double()
    if torch.cuda.is_available():
        model = model.cuda()
    epochs = int(sys.argv[2])
    lr = float(sys.argv[3])
    #print(lr.dtype)
    criterion = nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    #optimizer = torch.optim.Adam(model.parameters(),lr = lr)




    train_val(model = model,
            init_data=init,
            all_dataloader = all_loader,
            trainloader = train_loader,
            validloader = valid_loader,
            criterion= criterion,
            optimizer= optimizer,
            epochs= epochs,plot = False)


if __name__ == "__main__":
    main()





