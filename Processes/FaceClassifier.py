import time
import cv2
import numpy as np
import SystemParameters as sp
from PIL import Image


class FaceClassifier:

    def __init__(self, queue1, queue2):
        self.readQueue = queue1
        self.writeQueue = queue2
        self.width = sp.I_WIDTH_CLASSIFIER
        self.height = sp.I_HEIGHT_CLASSIFIER


    def initModel(self):
        from keras.models import load_model
        from keras.preprocessing.image import ImageDataGenerator
        self.model = load_model(sp.MODEL)
        self.test_datagen = ImageDataGenerator(rescale=1. / (255. / 2.) - 1)

    def classify(self):
        # extract the image from the face object
        face = self.readQueue.get()

        # get frame information from face
        faceFrame = face.getFrame()

        # resize to desirable input for model
        if(faceFrame.shape[0] > 0 and faceFrame.shape[1] > 0):
            # get prediction
            prediction = self.predict(faceFrame)
            # add prediction to image object
            face.setPrediction(prediction)
            # write the prediction to the queue
            self.writeQueue.put(face)

    def predict(self,image):
        from keras.preprocessing.image import img_to_array

        ### LOAD IMG replicator
        img = image
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            pass

        img = Image.fromarray(img)

        if (img.mode != 'L'):
            img = img.convert('L')

        img = img.resize((220, 160), 2)
        img = img_to_array(img, data_format='channels_last')
        img = img.reshape(1, 160, 220, 1)
        img = self.test_datagen.flow(img)[0]

        pred = self.model.predict(img)

        return pred

