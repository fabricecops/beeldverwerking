

from keras.layers import Conv2D, MaxPooling2D, LSTM
from keras.layers import Dropout, Flatten, Dense,BatchNormalization


from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
import src.helper.helper as hf
from src.models.tests.model.model import model
import cv2
import numpy as np
import os
from PIL import Image



class Conv_net(model):

    def __init__(self,dict_p=None,path=None):
        model.__init__(self, dict_p=dict_p,path=path)


        self.model = self.create_model()
        if(path != None ):
            self.model = self.load_model()

    def create_model(self):


        model       = Sequential()
        input_shape = (self.dict_p['img_width'], self.dict_p['img_height'], 1)

        filters     = self.dict_p['filters']
        for i, filter_ in enumerate(filters):
            if (i == 0):
                conv_layer = Conv2D(filter_[0],(filter_[1], filter_[2]), activation=filter_[4], padding='same',input_shape=input_shape)
                max_pool = MaxPooling2D((filter_[3], filter_[3]), padding='same')


            else:
                conv_layer = Conv2D(filter_[0], (filter_[1], filter_[2]), activation=filter_[4], padding='same')
                max_pool = MaxPooling2D((filter_[3], filter_[3]), padding='same')


            model.add(conv_layer)
            model.add(max_pool)
            model.add(Dropout(self.dict_p['dropout']))

            if (self.dict_p['batch_norm']== True):
                model.add(BatchNormalization())




        model.add(Flatten())
        model.add(Dense(self.dict_p['output'],activation='softmax'))

        model.compile(optimizer=self.dict_p['optimizer'], loss=self.dict_p['loss'],  metrics=['acc'])

        return model

    def fit_generator(self):

        callbacks     = self.create_callbacks()

        train_datagen = ImageDataGenerator(
                                    rescale         = self.dict_p['rescale'],
                                    shear_range     = self.dict_p['shear_range'],
                                    zoom_range      = self.dict_p['zoom_range'],
                                    horizontal_flip = self.dict_p['horizontal_flip'],
                                    vertical_flip   = self.dict_p['vertical_flip'],
                                    rotation_range  = self.dict_p['rotation']
                                        )

        # this is the augmentation configuration we will use for testing:
        # only rescaling
        test_datagen = ImageDataGenerator(rescale=self.dict_p['rescale'])

        train_generator = train_datagen.flow_from_directory(
                                    self.dict_p['train_data_dir'],
                                    target_size     = (self.dict_p['img_width'], self.dict_p['img_height']),
                                    batch_size      = self.dict_p['batch_size'],
                                    color_mode      = self.dict_p['colormode'])



        validation_generator = test_datagen.flow_from_directory(
                                    self.dict_p['validation_data_dir'],
                                    target_size     = (self.dict_p['img_width'], self.dict_p['img_height']),
                                    batch_size      = self.dict_p['batch_size'],
                                    color_mode      = self.dict_p['colormode'],
                                    shuffle = self.dict_p['shuffle_val'])


        history = self.model.fit_generator(
            train_generator,
            steps_per_epoch    = self.dict_p['train_steps'],
            epochs             = self.dict_p['epochs'],
            validation_data    = validation_generator,
            validation_steps   = self.dict_p['val_steps'],
            callbacks          = callbacks,
            verbose            = self.dict_p['verbose']
        )



        count = self.model.count_params()

        return history.history,count


    def predict(self,image):

                test_datagen = self.return_img_gen()
                ### LOAD IMG replicator
                img = image
                try:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except:
                    pass

                img = Image.fromarray(img)

                if (img.mode != 'L'):
                    img = img.convert('L')


                img = img.resize((self.dict_p['img_height'], self.dict_p['img_width']), 2)
                img = img_to_array(img, data_format='channels_last')
                img = img.reshape(1, 160, 220, 1)
                img = test_datagen.flow(img)[0]

                pred = self.model.predict(img)

                return pred


    def return_img_gen(self):
        test_datagen = ImageDataGenerator(rescale=self.dict_p['rescale'])
        return test_datagen

