import cv2
import os
from dotenv import find_dotenv, load_dotenv
import sys
load_dotenv(find_dotenv())

PATH_P = os.environ['PATH_P']
os.chdir(PATH_P)
sys.path.insert(0, PATH_P)
import src.helper.helper as hf
import os
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')


def plot_progress(path,Threshold):
    names = os.listdir(path)

    for i in range(len(names)):
        names[i] = int(names[i])

    names = sorted(names)



    scores = []
    count  = []
    nr_e   = []
    opt_a  = []
    best_a = []
    for name in names:
        try:

            path_g = path+str(name)+'/hist.p'

            p      = hf.pickle_load(path_g)
            val    = np.max(p['history']['val_acc'])
            count_ = p['count']

            if(val> 0.995):
                opt_value = count_

            else:
                opt_value = count_/val + 10000


            if(opt_value < Threshold):
                opt_a.append(opt_value)
                nr_e.append(len(p['history']['val_acc']))
                scores.append(np.max(p['history']['val_acc']))
                count.append(count_)

                best_a.append(min(opt_a))


        except:
            pass
    # score_f = []
    # score_c = []
    # epochs  = []
    # for c,val,epoch in zip(count,scores,nr_e):
    #         score_c.append(c)
    #         score_f.append(val)
    #         epochs.append(epochs)
    fig, ax = plt.subplots(figsize=(20, 12))
    plt.subplot(221)
    plt.plot(scores)
    plt.title('validation accuracies')
    plt.ylabel('validation accuracy')
    plt.xlabel('attempt #')

    plt.subplot(222)
    plt.plot(best_a)
    plt.title('tracking: minimum optimal value')
    plt.ylabel('minimum optimal value')

    plt.xlabel('attempt #')


    plt.subplot(223)
    plt.plot(count)
    plt.title('count nr of parameters')
    plt.ylabel('count')
    plt.xlabel('attempt #')

    plt.subplot(224)
    plt.hist(opt_a  , color = 'r', bins = 100)
    plt.title('histogram optimal values')
    plt.ylabel('optimal value')
    plt.xlabel('attempt #')
    plt.show()

    fig.savefig('./notebooks/images/BO.png')

def return_dropout(path):
    names = os.listdir(path)

    for i in range(len(names)):
        names[i] = int(names[i])

    names = sorted(names)

    dict_ = {}
    x = []
    y = []


    scores= []
    times = []
    for name in names:
        try:

            path_h = path+str(name)+'/hist.p'
            path_d = path+str(name)+'/dict.p'

            p= hf.pickle_load(path_h)
            d= hf.pickle_load(path_d)

            tmp1 = np.max(p['history']['val_acc'])
            tmp2 = d['dropout']



            if (tmp1 > 0.995):
                x.append(tmp2)
                y.append(tmp1)
        except:
            pass
    return x,y

def return_LR(path):
    names = os.listdir(path)

    for i in range(len(names)):
        names[i] = int(names[i])

    names = sorted(names)

    dict_ = {}
    x = []
    y = []


    scores= []
    times = []
    for name in names:
        try:

            path_h = path+str(name)+'/hist.p'
            path_d = path+str(name)+'/dict.p'

            p= hf.pickle_load(path_h)
            d= hf.pickle_load(path_d)

            tmp1 = np.max(p['history']['val_acc'])
            tmp2 = d['lr']



            if (tmp1 > 0.995):
                x.append(tmp2)
                y.append(tmp1)
        except:
            pass
    return x,y

def return_BN(path):


    names = os.listdir(path)

    for i in range(len(names)):
        names[i] = int(names[i])

    names = sorted(names)

    dict_ = {}
    dict_['true'] = []
    dict_['false'] = []


    scores= []
    times = []
    for name in names:
        try:

            path_h = path+str(name)+'/hist.p'
            path_d = path+str(name)+'/dict.p'

            p= hf.pickle_load(path_h)
            d= hf.pickle_load(path_d)

            tmp = np.max(p['history']['val_acc'])


            if( tmp > 0.995 ):

                if(d['batch_norm'] == True):
                    dict_['true'].append(tmp)

                if(d['batch_norm'] == False):


                    dict_['false'].append(tmp)

            times.append(np.mean(p['times']))
        except:
            pass
    return dict_

def return_filters(path):
    names = os.listdir(path)

    for i in range(len(names)):
        names[i] = int(names[i])

    names = sorted(names)

    dict_ = {}
    dict_[3] = []
    dict_[4] = []
    dict_[5] = []

    scores = []
    times = []
    for name in names:
        try:

            path_h = path + str(name) + '/hist.p'
            path_d = path + str(name) + '/dict.p'

            p = hf.pickle_load(path_h)
            d = hf.pickle_load(path_d)

            tmp = np.max(p['history']['val_acc'])

            #         if( tmp > 0.995 ):

            nr = len(d['filters'])
            if(tmp>0.995):
                dict_[nr].append(tmp)
        except:
            pass

    return dict_
