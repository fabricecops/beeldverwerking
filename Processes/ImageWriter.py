import cv2
import numpy as np


class ImageWriter():
    def __init__(self, queue, prefix):
        self.queue = queue
        self.prefix = prefix
        self.counter = 1

    def writeImages(self):
        face = self.queue.get()
        frame, frameNumber = face.getFrame(), face.getFrameNumber()
        if(np.mod(frameNumber,10) == 0):
            frameNumberString = str(self.counter)
            filename = self.prefix + '_' + frameNumberString.rjust(5, '0') + '.jpg'
            cv2.imwrite(filename, frame)
            self.counter += 1
            print(self.counter)