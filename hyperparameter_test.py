from numpy.core.fromnumeric import shape
from functools import partial
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
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from torch.utils.tensorboard import SummaryWriter
from DataloaderClassV2 import OCTdatasetV2
import Network
import info



#need to call dtaloader class nd split into train, val and test




def test(model,test_loader,criterion_test):
    writer_loc = info.test_writer_loc
    writer = SummaryWriter(writer_loc)
    model.eval()
    i = 0
    gt_array = []
    target_array = []
    for init_data,datum,groundtruth in test_loader:

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


def model_run(config,checkpoint_dir=None):
    print("hello 0")
    model = Network.generate_model()
    model = model.double()
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(model)
    model = model.to(device)
    epochs = info.epochs

    lr = config['lr']

    criterion_val = nn.L1Loss()
    criterion_train = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = 0.9)
    #optimizer = torch.optim.Adam(model.parameters(),lr = lr)
    checkpoint_dir = info.checkpoint_dir
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    scheduler = ExponentialLR(optimizer, gamma=info.gamma)

    train_data = OCTdatasetV2(train= True)
    valid_data = OCTdatasetV2(valid= True)
    train_loader = torch.utils.data.DataLoader(dataset= train_data, batch_size= 1,shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset = valid_data, batch_size= 1)

    print("Hello 1")
    for e in range(epochs):
        print("hello 2")
        train_loss = 0.0
        model.train()     # Optional when not using Model Specific layer
        i = 0
        for init_data,datum,groundtruth in tqdm(train_loader):
            i = i+1

            if torch.cuda.is_available():
                init_data ,datum, groundtruth = init_data.cuda(), datum.cuda(), groundtruth.cuda()

            optimizer.zero_grad()
            target = model(init_data,datum)
            groundtruth = groundtruth[0].double()
            t_loss = criterion_train(target,groundtruth)
            t_loss.backward()
            optimizer.step()

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


            valid_loss = v_loss.item() * datum.size(0)

        with tune.checkpoint_dir(e) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)
        tune.report(loss=(valid_loss / len(valid_loader)))



#tensorboard --logdir=runs
def main(num_samples = 10,max_num_epochs=1, gpus_per_trial=2,mode = "train"):

    if mode == "train":
        config = {
            "lr": tune.loguniform(1e-4, 1e-1),
        }

        scheduler = ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=max_num_epochs,
            grace_period=1,
            reduction_factor=2)
        reporter = CLIReporter(
            # parameter_columns=["l1", "l2", "lr", "batch_size"],
            metric_columns=["loss", "training_iteration"])
        print("Hello 3")
        result = tune.run(
            model_run,
            resources_per_trial={"cpu": 10, "gpu": gpus_per_trial},
            config=config,
            num_samples = num_samples,
            scheduler=scheduler,
            progress_reporter=reporter)

        best_trial = result.get_best_trial("loss", "min", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))


    if mode == 'test':
        path = sys.argv[2]
        model = Network.generate_model()
        model = model.double()
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
            if torch.cuda.device_count() > 1:
                net = nn.DataParallel(model)
        model.to(device)

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
    main(num_samples = 10,max_num_epochs=1, gpus_per_trial=2,mode = "train")





