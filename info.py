from pandas.core.indexing import is_label_like


train = True
train_file_loc = "/home/Mukherjee/Data/Related_files/test_2_train_labels.txt"
valid = False
valid_file_loc = "/home/Mukherjee/Data/Related_files/test_2_valid_labels.txt"
test = False
test_file_loc = "/home/Mukherjee/Data/Related_files/test_2_test_labels.txt"
#helper_func stuff
h5_loc = "/home/Mukherjee/Data/"
ispreprocessing = True
iszoom = True
lim = [1000000,10000000,10000000] #limits of the x,y nd z difference
islabel = False #to print labels pairs