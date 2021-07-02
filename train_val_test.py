from numpy.core.fromnumeric import shape
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import torch
from torchvision import transforms
import numpy as np
import torch.nn as nn
import sys
import os
import pandas as pd
import time

from torch.utils.tensorboard import SummaryWriter
from DataloaderClassV2 import OCTdatasetV2
import Network
import info



#need to call dtaloader class nd split into train, val and test

def train_val(model,train_loader,valid_loader,criterion_train,criterion_val, optimizer,scheduler, model_save_loc,epochs,plot = False):
    if plot is True:
        writer_loc = info.train_writer_loc
        writer = SummaryWriter(writer_loc)

    min_valid_loss = np.inf

    count_tr = 0
    count_vl = 0
    running_t_loss = 0.0
    running_v_loss = 0.0
    for e in range(epochs):

        train_loss = 0.0
        model.train()     # Optional when not using Model Specific layer
        i = 0
        for init_data,datum,groundtruth in tqdm(train_loader):
            i = i+1

            if torch.cuda.is_available():
                init_data ,datum, groundtruth = init_data.cuda(), datum.cuda(), groundtruth.cuda()
            #print(shape(init_data),shape(datum))
            optimizer.zero_grad()
            #print(type(init_data),type(data))
            target = model(init_data,datum)
            #print(shape(target),shape(groundtruth[0]))
            groundtruth = groundtruth[0].double()

            #print(traininit.shape,data.shape)
            #print(target.dtype,groundtruth.dtype)
            t_loss = criterion_train(target,groundtruth)
            t_loss.backward()
            optimizer.step()
            if plot == True:
                running_t_loss += t_loss.item()
                if i % 100 == 99:    # every 100 mini-batches...
                    # ...log the running loss
                    count_tr = count_tr+1
                    writer.add_scalar('Loss/training_loss',
                            running_t_loss / 100,
                            e * len(train_loader) + i)
                    #writer.add_scalars('loss', {'train': running_t_loss / 100}, (e-1) * len(train_loader) + i)
                    running_t_loss = 0.0

            train_loss = t_loss.item() * datum.size(0)
        j = 0
        valid_loss = 0.0
        model.eval()     # Optional when not using Model Specific layer
        for init_data,datum,groundtruth in tqdm(valid_loader):
            j = j+1
            if torch.cuda.is_available():
                init_data ,datum, groundtruth = init_data.cuda(), datum.cuda(), groundtruth.cuda()

            target = model(init_data,datum)
            groundtruth = groundtruth[0].double()
            v_loss = criterion_val(target,groundtruth)
            if plot is True:

                running_v_loss += v_loss.item()
                if j % 10 == 9:    # every 10 mini-batches...
                    # ...log the running loss
                    count_vl = count_vl+1
                    writer.add_scalar('Loss/Validation_loss',
                            running_v_loss / 10,
                            e * len(valid_loader) + j)
                    #writer.add_scalars('loss', {'valid': running_v_loss / 10}, (e-1) * len(valid_loader) + j)
                    running_v_loss = 0.0

            valid_loss = v_loss.item() * datum.size(0)


        print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(train_loader)} \t\t Validation Loss: {valid_loss / len(valid_loader)}')
        scheduler.step()
        print('Learning Rate: ', scheduler.get_last_lr())
        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            # Saving State Dict
            #timestr = time.strftime("%Y%m%d-%H")
            if not os.path.isdir(model_save_loc):
                os.mkdir(model_save_loc)
            torch.save(model.state_dict(), model_save_loc+'saved_model.pth')
    if plot is True:
        writer.flush()
        writer.close()


def test(model,test_loader,criterion_test):
    writer_loc = info.test_writer_loc
    writer = SummaryWriter(writer_loc)
    model.eval()
    i = 0
    gt_array = []
    target_array = []
    for init_data,datum,groundtruth in tqdm(test_loader):

        if torch.cuda.is_available():
                init_data ,datum, groundtruth = init_data.cuda(), datum.cuda(), groundtruth.cuda()


        target = model(init_data,datum)

        groundtruth = groundtruth[0]
        if torch.cuda.is_available():
            target_np = target.detach().cpu().numpy()
            groundtruth_np = groundtruth.detach().cpu().numpy()
        else:
            target_np = target.detach().numpy()
            groundtruth_np = groundtruth.detach().numpy()

        #if i<10:
        #    set.append([target_np,groundtruth_np])
        diff = abs(groundtruth_np[0] - target_np[0])
        #print(diff[0])
        gt_array.append(groundtruth_np[0])
        target_array.append(target_np[0])
        writer.add_scalar("Error in x",diff[0],i)
        writer.add_scalar("Error in y",diff[1],i)
        writer.add_scalar("Error in z",diff[2],i)
        i = i+1




    writer.flush()
    writer.close()
    r2 = r2_score(gt_array,target_array)
    print("R2 score: ", r2)
    mse1 = mean_squared_error(gt_array,target_array,squared = False)
    print('MSE1: ', mse1)
    mse2 = mean_squared_error(gt_array,target_array, multioutput = 'raw_values',squared = False)
    print('MSE2: ', mse2)
    mae1 = mean_absolute_error(gt_array,target_array)
    print('MAE1: ', mae1)
    mae2 = mean_absolute_error(gt_array,target_array,multioutput = 'raw_values')
    print('MAE2: ', mae2)








#tensorboard --logdir=runs
def main():

    if sys.argv[1] == 'train':
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

        epochs = info.epochs
        #epochs = 1
        lr = info.lr
        #lr = 0.0001
        #print(lr.dtype)
        criterion_val = nn.L1Loss()
        criterion_train = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = 0.9)
        #optimizer = torch.optim.Adam(model.parameters(),lr = lr)

        scheduler = ExponentialLR(optimizer, gamma=info.gamma)

        train_val(model = model,
                train_loader = train_loader,
                valid_loader = valid_loader,
                criterion_train= criterion_train,
                criterion_val = criterion_val,
                optimizer= optimizer,
                scheduler=scheduler,
                model_save_loc = info.model_save_loc,
                epochs= epochs,plot = True)

    if sys.argv[1] == 'test':
        path = sys.argv[2]
        model = Network.generate_model()
        model = model.double()
        if torch.cuda.is_available():
            model = model.cuda()

        if torch.cuda.is_available() is False:
            model.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(path))

        criterion_test = nn.L1Loss()

        test_data = OCTdatasetV2(test= True)
        test_loader = torch.utils.data.DataLoader(dataset= test_data, batch_size= 1)
        test(model = model,
             test_loader=test_loader,
             criterion_test=criterion_test)

if __name__ == "__main__":
    main()





