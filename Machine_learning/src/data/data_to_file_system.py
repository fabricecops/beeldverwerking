import src.helper.helper as hf
import os
import random
import cv2


def create_train_val(path_in,path_out,val_frac):

    path_val   = path_out+'val/'
    path_train = path_out+'train/'

    if not os.path.exists(path_val):
        os.makedirs(path_val)

    if not os.path.exists(path_train):
        os.makedirs(path_train)

    names    = os.listdir(path_in)


    for name in names:

        path_n = path_in + name + '/'
        frames = os.listdir(path_n)

        frames = list(filter(lambda x: 'info' not in x,frames))

        len_frames = len(frames)
        val_frames = int(val_frac*len_frames)
        indexes    = random.sample(range(1,len_frames),val_frames)

        path_n_v   = path_val+name+'/'
        path_n_t   = path_train+name+'/'

        if not os.path.exists(path_n_v):
            os.makedirs(path_n_v)

        if not os.path.exists(path_n_t):
            os.makedirs(path_n_t)
        indexes    = random.sample(range(1,len_frames),val_frames)


        for i,frame in enumerate(frames):


                src   = path_n + frame
                image = cv2.imread(src)

                frame = frame[:-3]+'jpg'
                if(i in indexes):
                    dst = path_n_v+frame

                else:
                    dst = path_n_t+frame

                write(dst,image)

def write(dst,image):
    try:
        cv2.imwrite(dst,image)
    except:
        print('failed')

def create_train_val_LC(path_in,path_out,nr):

        path_out = path_out + str(nr) + '/'

        if not os.path.exists(path_out):
            os.makedirs(path_out)


        path_val   = path_out+'val/'
        path_train = path_out+'train/'

        if not os.path.exists(path_val):
            os.makedirs(path_val)

        if not os.path.exists(path_train):
            os.makedirs(path_train)

        names    = os.listdir(path_in)

        for name in names:

            path_n = path_in + name + '/'
            frames = os.listdir(path_n)

            frames = list(filter(lambda x: 'info' not in x, frames))

            len_frames = len(frames)
            train_frames = nr
            indexes = random.sample(range(1, len_frames), train_frames)

            path_n_v = path_val + name + '/'
            path_n_t = path_train + name + '/'

            if not os.path.exists(path_n_v):
                os.makedirs(path_n_v)

            if not os.path.exists(path_n_t):
                os.makedirs(path_n_t)

            for i, frame in enumerate(frames):

                src = path_n + frame
                image = cv2.imread(src)

                frame = frame[:-3] + 'jpg'
                if (i in indexes):
                    dst = path_n_t + frame

                else:
                    dst = path_n_v + frame

                cv2.imwrite(dst, image)

if __name__=='__main__':

    ####  faces dataset    #####

    # path_in    = './data/raw/faces94/female/'
    # path_out   = './data/processed/faces/'
    #
    # create_train_val(path_in,path_out,0.2)

    #### yale dataset     ######

    # path_in    = './data/raw/ExtendedYaleB/'
    # path_out   = './data/processed/yale/BO_data/'

    # create_train_val(path_in,path_out,0.2)

    ## generate data learning curve ##

    #
    # nr_a = [1,2,3,4,5,6,7,8,9,13,15,20,25,30,35,40,45,50,60,70,80,90,100]
    # nr_a = [1,2,3,4]
    # for nr in nr_a:fig.savefig('./notebooks/images/histograms.png')
    #
    #     path_in  = './data/raw/ExtendedYaleB/'
    #     path_out = './data/processed/yale/LC/'
    #
    #     create_train_val_LC(path_in, path_out, nr)


    ####  Own made dataset    #####

    path_in    = './data/raw/facesGroupMembers/'
    path_out   = './data/processed/OGD/'

    create_train_val(path_in,path_out,0.2)