import pickle
import time
import shutil
import os

def return_conf_path(path, mode = 'pickle'):
    if(mode == 'pickle'):
        name = str(len(os.listdir(path))) + '.p'
    elif(mode == 'model'):
        name = str(len(os.listdir(path))) + '.h5'

    path = path + name

    return path

def pickle_load(path):
    return  pickle.load( open( path, "rb" ) )

def pickle_save(path,data):

    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def diagnostics(data):

    try:
        print('dtype = ',data.dtype)
    except:
        pass

    try:
        print('type = ',type(data))
    except:
        pass

    try:
        print('len = ',len(data))
    except:
        pass


    try:
        print('keys = ',data.keys())
    except:
        pass

    try:
        print('shape = ',data.shape)
    except:
        pass


def tic():
    global time_
    time_ = time.time()

def toc():
    global time_
    tmp = time.time()

    elapsed = tmp - time_

    print('the elapsed time is: ', elapsed)

    return elapsed


def copy_data(src,dst):
    shutil.copy(src, dst)


def clean_directories(path):

    folder = os.listdir(path)


if __name__ =='__main__':

    path = 'tests/conv/'

    clean_directories(path)


