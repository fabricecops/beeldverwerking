import os
import pickle

class FS_manager():

    def __init__(self,path=None):

        self.path                  = path
        self.path_gen,self.path_TB = self.create_dir()


    def create_dir(self):
        print('CREATE DIR IS TRIGGERED')

        list_ = os.listdir(self.path)
        if('dict.p' not in  list_):
            path_gen = self.path+str(len(os.listdir(self.path)))+'/'
        else:
            path_gen = self.path

        path_TB  =  path_gen +'tensorboard'

        if (os.path.exists(self.path)==False):
            os.mkdir(self.path)
        if (os.path.exists(path_gen)==False):
            os.mkdir(path_gen)
        if (os.path.exists(path_TB) == False):
            os.mkdir(path_TB)

        return path_gen,path_TB

    def pickle_save(self,data,mode):

        if(mode=='dict'):
            path = self.return_path_dict()
        if(mode=='hist'):
            path = self.return_path_hist()

        with open(path, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def pickle_load(self,path):
        return pickle.load(open(path, "rb"))

    def return_path(self):
        return self.path_gen

    def return_path_model(self):
        path_model = self.path_gen+'model.h5'
        return path_model

    def return_path_TB(self):
        return self.path_TB

    def return_path_dict(self):
        path_dict =  self.path_gen+'/dict.p'
        return path_dict

    def return_path_hist(self):
        path_dict = self.path_gen + 'hist.p'
        return path_dict

    def return_path_CSV(self):
        path_dict = self.path_gen + 'hist.csv'
        return path_dict






