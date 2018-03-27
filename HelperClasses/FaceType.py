import SystemParameters as sp
import numpy as np

class FaceType:

    def __init__(self, bbox, superFrame, type, prediction=None):
        self.bbox = bbox  # bounding box, contains coordinates and width/height
        self.superFrame = superFrame
        self.prediction = prediction  # output of classifier, None if not classified
        self.type = type
        self.className = 'Undefined'

    def setClassName(self, className):
        self.className = className

    def getClassName(self,):
        self.calcClassName()
        return self.className

    def calcClassName(self):
        prediction = self.getPrediction()
        if(len(prediction) > 0):
            prediction = prediction.ravel()
            prediction = prediction.tolist()
            try:
                classIndex = np.argmax(prediction)
                className = sp.CLASS_NAMES[classIndex]
                self.setClassName(className)
            except ValueError:
               pass


    def setPrediction(self, prediction):
        self.prediction = prediction

    def getPrediction(self):
        return self.prediction

    def getFrameNumber(self):
        return self.superFrame.getFrameNumber()

    def setFrameNumber(self, frameNumber):
        self.superFrame.setFrameNumber(frameNumber)

    def getFrame(self):
        return self.superFrame.getFrame()

    def setFrame(self, frame):
        self.superFrame.setFrame(frame)

    def getBbox(self):
        return self.bbox

    def setBbox(self, bbox):
        self.bbox = bbox

    def getType(self):
        return self.type

    def setType(self, type):
        self.type = type

    def overlaps(self, face, currentFrameNumber):
        frameNumber = face.getFrameNumber()
        x,y,w,h = self.bbox
        xtmp,ytmp,wtmp,htmp = face.getBbox()

        leftTop = [xtmp, ytmp]
        rightTop = [xtmp+wtmp, ytmp]
        leftBottom = [xtmp, ytmp+htmp]
        rightBottom = [xtmp + wtmp,ytmp+htmp]

        if(currentFrameNumber == frameNumber):
            if(leftTop[0] >= x and leftTop[0] <= x+w and leftTop[1] >= y and leftTop[1] <= y+w):
                overlap = True
            elif(rightTop[0] >= x and rightTop[0] <= x+w and rightTop[1] >= y and rightTop[1] <= y+w):
                overlap = True
            elif(leftBottom[0] >= x and leftBottom[0] <= x+w and leftBottom[1] >= y and leftBottom[1] <= y+w):
                overlap = True
            elif(rightBottom[0] >= x and rightBottom[0] <= x+w and rightBottom[1] >= y and rightBottom[1] <= y+w):
                overlap = True
            else:
                overlap = False
        else:
            overlap = False

        return overlap
