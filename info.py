

#train = True
train_file_loc = "/home/Mukherjee/Data/Related_files/Cross_only_train_labels.txt"
#valid = False
valid_file_loc = "/home/Mukherjee/Data/Related_files/Cross_only_valid_labels.txt"
#test = False
test_file_loc = "/home/Mukherjee/Data/Related_files/Cross_only_test_labels.txt"
#helper_func stuff
h5_loc = "/home/Mukherjee/Data/"
ispreprocessing = True
iszoom = True
lim = [100,100,100] #limits of the x,y nd z difference
islabel = False #to print labels pairs
#train_valid_test
run = 'run4'
isplot = False
epochs = 1
lr = 0.00001
gamma = 0.9
#checkpoint_dir = None
train_writer_loc = 'runs/Siamese_net_train_2'
test_writer_loc = 'runs/Siamese_net_test_error_2'
model_save_loc = "/models"
checkpoint_dir = "/checkpoints"
