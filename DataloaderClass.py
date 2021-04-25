import h5py as h5
import numpy as np 
from torch.utils.data import Dataset
from plotly import graph_objects as go

class OCTDataset(Dataset):
    """ OCT dataset """
    
    
    def __init__(self, h5_file, transform = None, init_index = 1, train = False):
        """
        Args:
            h5_file (string): Path to the h5 file with annotations and images.
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        self.h5_file = h5.File(h5_file, 'r')
        self.position = self.h5_file['position']
        self.octdata = self.h5_file['octdata']
        self.init_index = init_index
        self.transform = transform
        self.train = train
        
    def groundTruth(self, index):
        groundtruth = []
        old_pose = np.array(self.position[str(self.init_index)])
        new_pose = np.array(self.position[str(index)])
        for i in range(3):
            value = new_pose[i] - old_pose[i]
            groundtruth.append(value)        
        return groundtruth
    
        
    def __len__(self):
        return len(self.octdata)
    
    def __getitem__(self, index): 
        """
        Args:
            index (int) : Index of the singlular data to be transformed/called
            always starts from 1         
        """
        if index < len(self.octdata):
            index = index+1
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
                          
                
        return datum, groundT 
        # make it return groundtruth
    
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
    
                
                