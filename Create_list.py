import h5py as h5
import numpy as np
import pandas as pd



def create_list(h5_loc):
    h5_file = h5.File(h5_loc,'r')
    position = h5_file['position']
    octdata = h5_file['octdata']
    length = len(octdata)
    index_list = []
    for i in range(length):
        for j in range(length):
            pair = (i+1,j+1)
            index_list.append(pair)
    index_list = pd.DataFrame(index_list)
    return index_list

def save_list(index_list,list_loc):
    df = index_list
    df.to_csv(list_loc,index = False, header = False)

def read_list(list_loc):
   df =  pd.read_csv(list_loc,header=None)
   return df


#index_list = create_list("/home/Mukherjee/Data/Cross_ext.h5")
#save_list(index_list,list_loc="/home/Mukherjee/Data/Cross_ext_index.csv")
#df = read_list(list_loc="/home/Mukherjee/Data/Cross_ext_index.csv")



