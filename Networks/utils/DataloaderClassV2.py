#will be simplistic so that it can directly be fed to the trainloader/other loaders
#will read from text files
#new mode option to read from different text files


from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import helper_func
import info


#print(type(self.train))
class OCTdatasetV2(Dataset):

    def __init__(self,transform = transforms.Compose([transforms.ToTensor()]),train = False, valid = False,test = False):
        super().__init__()
        self.octdata = helper_func.read_h5()
        self.transform = transform
        self.train = train
        self.valid = valid
        self.test = test

        if self.train is True:
            print("Train")
            self.train_labels = helper_func.read_files(info.train_file_loc)

        elif self.valid is True:
            print("Valid")
            self.valid_labels = helper_func.read_files(info.valid_file_loc)

        elif self.test is True:
            print("Test")
            self.test_labels = helper_func.read_files(info.test_file_loc)


    def __len__(self):
        if self.train is True:
            return len(self.train_labels)
        elif self.valid is True:
            return len(self.valid_labels)
        elif self.test is True:
            return len(self.test_labels)


    def __getitem__(self, index):

       if self.train is True:
           init_label,label = self.train_labels[index]
           init_data = self.octdata[init_label][0]
           datum = self.octdata[label][0]
           groundtruth = np.subtract(self.octdata[init_label][1],self.octdata[label][1])
           groundtruth = np.transpose(groundtruth)
           if self.transform is not None:
                datum = self.transform(datum)
                datum = datum.reshape(1,-1,64,64)
                init_data = self.transform(init_data)
                init_data = init_data.reshape(1,-1,64,64)
           return init_data,datum,groundtruth

       elif self.valid is True:
           init_label,label = self.valid_labels[index]
           init_data = self.octdata[init_label][0]
           datum = self.octdata[label][0]
           groundtruth = np.subtract(self.octdata[init_label][1],self.octdata[label][1])
           groundtruth = np.transpose(groundtruth)
           if self.transform is not None:
                datum = self.transform(datum)
                datum = datum.reshape(1,-1,64,64)
                init_data = self.transform(init_data)
                init_data = init_data.reshape(1,-1,64,64)
           return init_data,datum,groundtruth

       elif self.test is True:
           init_label,label = self.test_labels[index]
           init_data = self.octdata[init_label][0]
           datum = self.octdata[label][0]
           groundtruth = np.subtract(self.octdata[init_label][1],self.octdata[label][1])
           groundtruth = np.transpose(groundtruth)
           if self.transform is not None:
                datum = self.transform(datum)
                datum = datum.reshape(1,-1,64,64)
                init_data = self.transform(init_data)
                init_data = init_data.reshape(1,-1,64,64)
           return init_data,datum,groundtruth

#octdata = OCTdatasetV2(train=True)