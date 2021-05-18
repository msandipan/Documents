import h5py as h5
import numpy as np
from torch.utils.data import Dataset
from plotly import graph_objects as go
from torchvision import transforms
import pandas as pd

class OCTDataset(Dataset):
    """ OCT dataset """


    def __init__(self, h5_loc, index_list, transform = None, train = False):
        """
        Args:
            h5_file (string): Path to the h5 file with annotations and images.
            transform (callable, optional): Optional transform to be applied
            on a sample.
            index_list (dataframe): Dataframe consisting of index numbers
            init_index (int) : Defines the image that the rest of the dtaset
            should be compared to
            train (boolean,optional) : If true the data output will be in the form
            of arrays/lists else it will be a Dataset object
        """
        self.h5_file = h5.File(h5_loc, 'r')
        self.position = self.h5_file['position']
        self.octdata = self.h5_file['octdata']
        #self.init_index = init_index
        self.transform = transform
        self.train = train
        self.index_list = index_list


    def groundTruth(self, index, init_index):
        """
        Args:
            index (int): To specifiy the index which will be compared to the init
            index.
        Out:
            groundTruth (np.array of int): Gives an [xdiff, ydiff, zdiff] of int
            different between new pose and old pose(defined by init_index)

        """


        groundtruth = np.empty((3,),dtype = int)
        old_pose = self.position[str(init_index)]
        new_pose = self.position[str(index)]
        for i in range(3):
            value = new_pose[i] - old_pose[i]
            groundtruth[i] = value
        return groundtruth




    def __len__(self):
        """
        Out:
            Return the length of the dataset.
        """
        return len(self.index_list)

    def get_value(self, init_index, index):
        """
        Args:
            index (int): Index of the data/label that was called

        Out:
            if train = True, returns a tuple with datum and groundtruth
                datum (np.ndarray) : OCT 3d voxel data
                groundtruth (np.ndarray) : Voxel displacement data
            if train = False, returns a tuple with datum and pos
                datum (Dataset object) : OCT 3d voxel data
                pos (Dataset object) : Voxel position data

            if transform is not None, datum reshaped to 4D (channel data added)

        """

        if self.train is True:
            datum = np.array(self.octdata[str(index)])
            init_data = np.array(self.octdata[str(init_index)])
            pos = np.array(self.position[str(index)])
            groundT = self.groundTruth(index,init_index)
            if self.transform is not None:
                datum = self.transform(datum)
                datum = datum.reshape(1,-1,64,64)
                init_data = self.transform(init_data)
                init_data = init_data.reshape(1,-1,64,64)
            return init_data, datum, groundT
        datum = self.octdata[str(index)]
        init_data = np.array(self.octdata[str(init_index)])
        pos = self.position[str(index)]
        groundT = self.groundTruth(index,init_index)
        if self.transform is not None:
            datum = self.transform(datum)
            datum = datum.reshape(1,-1,64,64)
            init_data = self.transform(init_data)
            init_data = init_data.reshape(1,-1,64,64)
        return init_data, datum, pos


    def __getitem__(self,index):
        """
        Args:
            index (int) : Index of the singlular data to be transformed/called
            always starts from 1

        Out:

        """

        #init_index = self.init_index
        #need to implement slice stuff
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self.index_list))
            init_start,init_stop,step = index.indices(len(self.index_list))

            return [self[i] for i in range(start, stop, step)]
        elif isinstance(index, int):
            #if index >= len(self):
            #    raise IndexError('Index is out of bounds')
            #if index<len(self):
            #    index = index+1
            #    init_index = init_index
            data_index,init_index = self.index_list.loc[index]
            return self.get_value(data_index,init_index)
        else:
            raise TypeError('Invalid argument type: {}'.format(type(index)))




    def close(self):
        self.h5_file.close()



    def view_data(self, index):
        volume = np.array(self.octdata[str(index)])
        x1 = np.linspace(0, 63, 64)
        y1 = np.linspace(0, 63, 64)
        z1 = np.linspace(0, 63, 64)

        X, Y, Z = np.meshgrid(x1, y1, z1)
        fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=volume.flatten(),
        isomin=-0.5,
        isomax=0.5,
        opacityscale="uniform",
        opacity=0.5,
        caps= dict(x_show=False, y_show=False, z_show=True)
        ))
        return fig



#file = "/home/Mukherjee/Data/Cross_ext.h5"
#list_loc = "/home/Mukherjee/Data/Cross_ext_index.csv"
#trans = transforms.Compose([transforms.ToTensor()])
#data_list = pd.read_csv(list_loc,header=None)
#data = OCTDataset(h5_loc = file,transform=trans, train = True,index_list = data_list)
#data = OCTDataset(h5_file = file, train = False)
