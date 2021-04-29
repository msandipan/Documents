import h5py as h5
import numpy as np 
from torch.utils.data import Dataset
from plotly import graph_objects as go
from torchvision import transforms

class OCTDataset(Dataset):
    """ OCT dataset """
    
    
    def __init__(self, h5_file, transform = None, init_index = 1, train = False):
        """
        Args:
            h5_file (string): Path to the h5 file with annotations and images.
            transform (callable, optional): Optional transform to be applied
            on a sample.
            init_index (int) : Defines the image that the rest of the dtaset
            should be compared to
            train (boolean,optional) : If true the data output will be in the form 
            of arrays/lists else it will be a Dataset object
        """
        self.h5_file = h5.File(h5_file, 'r')
        self.position = self.h5_file['position']
        self.octdata = self.h5_file['octdata']
        self.init_index = init_index
        self.transform = transform
        self.train = train
        
        
    def groundTruth(self, index):
      
        
        groundtruth = np.empty((3,),dtype = int)
        old_pose = self.position[str(self.init_index)]
        new_pose = self.position[str(index)]
        for i in range(3):
            value = new_pose[i] - old_pose[i]
            groundtruth[i] = value      
        return groundtruth
    
    
    
        
    def __len__(self):
        return len(self.octdata)
    
    def get_value(self, index):
        
        #reshape to add another dimensiondata
        if self.train == False:
            datum = self.octdata[str(index)]
            
            pos = self.position[str(index)]
            groundT = self.groundTruth(index)
            if self.transform is not None:
                datum = self.transform(datum)
        else:
            datum = np.array(self.octdata[str(index)])            
            pos = np.array(self.position[str(index)]) 
            groundT = self.groundTruth(index)      
            if self.transform is not None:
                datum = self.transform(datum)
                datum = datum.reshape(1,-1,64,64)              
                
        return datum, groundT 
        # make it return groundtruth
    
    def __getitem__(self, index): 
        """
        Args:
            index (int) : Index of the singlular data to be transformed/called
            always starts from 1         
        """
        #need to implement slice stuff
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            #data_array = np.empty((len(self),),dtype = object)
            #gt_array = np.empty((len(self),),dtype = object)
            #for i in range(start, stop, step):
            #    data,gt = self[i]
            #    data_array[i] = data
            #    gt_array[i] = gt
            
            #return data_array,gt_array 
            return [self[i] for i in range(start, stop, step)]
        elif isinstance(index, int):
            if index >= len(self):
                raise IndexError('Index is out of bounds')
            if index<len(self):
                index = index+1
            return self.get_value(index)
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
#transform = transforms.Compose([transforms.ToTensor()])
#data = OCTDataset(h5_file = file,transform=transform, train = True)                
                