#add function for spliting
#add function for storing train,test and valid data
#functiong to read h5_files and make index like (A1,A99) and from another file (B1,B6)
# Have a text file that has all the relevant info and that will be shuffeled.
#Relevant info - ref ing index, img index, and the the pairs will be made before
# hand while reading the h5 file so there should be no mix up. Since indexing of the next
#file starts when one ends


import h5py as h5
import json
import numpy as np
import glob
import os
import scipy
from torch.functional import split
from sklearn.model_selection import train_test_split
import info

def traverse_datasets(hdf_file):

    def h5py_dataset_iterator(g, prefix=''):
        for key in g.keys():
            item = g[key]
            path = f'{prefix}/{key}'
            if isinstance(item, h5.Dataset): # test for dataset
                yield (path, item)
            elif isinstance(item, h5.Group): # test for group (go down)
                yield from h5py_dataset_iterator(item, path)

    for path, _ in h5py_dataset_iterator(hdf_file):
        yield path


def datapath(filename):
    #filename = '/home/Mukherjee/Data/Cross_ext.h5'
    with h5.File(filename, 'r') as f:
        for dset in traverse_datasets(f):
            print('Path:', dset)
            print('Shape:', f[dset].shape)
            print('Data type:', f[dset].dtype)




def read_h5(h5_dir = info.h5_loc,preprocessing = info.ispreprocessing,zoom = info.iszoom,lim = info.lim,islabels = info.islabel):
    #position_arr = []
    octdata_arr = []
    labels_arr = []
    length = 0

    print("Opening diectory....")
    for h5f in glob.iglob(h5_dir+"/*.h5"):

        h5_filename = h5f[len(h5_dir):-len('.h5')]
        h5_data = h5.File(h5f,'r')
        #print(h5_data['octdata']['1'].shape)
        position_arr = []
        print('Reading data from '+h5_filename+'....')
        #print(len(h5_data[1]))
        groups = list(h5_data.keys())
        indices = list(h5_data[groups[0]].keys())
        print("Number of Data points:", len(indices))
        #print(lim)
        for idx in range(0,len(indices)):
            #groups = list(h5_data.keys())
            #indices = list(h5_data[groups[0]].keys())
            #print(len(indices))
            shape = h5_data[groups[0]][indices[0]].shape
            octdata = np.array(h5_data[groups[0]][indices[idx]])
            position = np.array(h5_data[groups[1]][indices[idx]])

            if preprocessing is True:
                octdata = (octdata - np.mean(octdata))/ np.std(octdata)
                if zoom is True:
                    octdata = scipy.ndimage.zoom(octdata, (0.5, 0.5, 0.5))

            octdata_arr.append([octdata,position,h5_filename])

            position_arr.append(position)

        if islabels is True:

            matched_lables = match_samples(length,position_arr,lim)
            length = length + len(position_arr)
            print("Labels:",len(matched_lables))
            for labels in matched_lables:
                labels_arr.append(labels)
        print('Finished reading data from '+h5_filename+'....')
    if islabels is True:
        print("Total labels:",len(labels_arr))
        return octdata_arr,labels_arr
    else:
        print("Finished reading all data")
        return octdata_arr

def match_samples(length,pos,lim):

    mtchd = []
    xlim,ylim,zlim = lim
    for p in range(0, len(pos)):
        for j in range(0, len(pos)):
            if (abs(pos[p][0] - pos[j][0]) < xlim) and (abs(pos[p][1] - pos[j][1]) < ylim) and (abs(pos[p][2] - pos[j][2]) < zlim):
                mtchd.append([length+p,length+j])
    return mtchd


def groundTruth(index_list,pos_list):

    groundtruth_list = []
    print("Generating Groundtruths....")
    for indices in index_list:
        init_index,index = indices
        init_pos, pos = pos_list[int(init_index)],pos_list[int(index)]
        groundtruth = np.subtract(init_pos,pos)
        groundtruth_list.append(groundtruth)
    print("Groundtruths generated....:",len(groundtruth_list))
    return np.array(groundtruth_list)

def create_files(labels_list,file_path,filename):

    training_labels,test_labels = train_test_split(labels_list,test_size=0.2)
    training_labels,valid_labels = train_test_split(training_labels,test_size=0.1)

    train_filename = str(filename) + "_train_labels.txt"
    valid_filename = str(filename) + "_valid_labels.txt"
    test_filename = str(filename) + "_test_labels.txt"

    if not os.path.isdir(file_path):
        os.mkdir(file_path)

    print("Writing relatated files....")
    with open(os.path.join(file_path, train_filename), 'w') as f:
        json.dump(training_labels,f)
        #for item in training_labels:
        #    f.write("%s\n" % item)
    with open(os.path.join(file_path, valid_filename), 'w') as f:
        json.dump(valid_labels,f)
        #for item in valid_labels:
        #    f.write("%s\n" % item)
    with open(os.path.join(file_path, test_filename), 'w') as f:
        json.dump(test_labels,f)
        #for item in test_labels:
        #    f.write("%s\n" % item)

    print("Finished")
    return None

def read_files(file_loc):
    with open((file_loc),'r') as f:
        data_list = json.load(f)

    return np.array(data_list)

#print(info.h5_dir_loc)
#print(info.lim)
#octdata,labels = read_h5(islabels=True)
#octdata = read_h5()
#pos_list = np.array([col[1] for col in octdata])

#gt = groundTruth(labels,pos_list)
#file_path = "/home/Mukherjee/Data/Related_files"
#filename = "Cross_only(lim100)"
#create_files(labels,file_path,filename)
#data = read_files("/home/Mukherjee/Data/Related_files/test_2_test_labels.txt")

#x = np.array([col[0] for col in gt])
#y = np.array([col[1] for col in gt])
#z = np.array([col[2] for col in gt])


