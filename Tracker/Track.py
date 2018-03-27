import cv2, numpy as np

class Track:
    def __init__(self, face, dt):
        self.id = self.generateId()
        self.bbox = face.bbox
        self.kalman = self.createKalman(face.bbox, dt)
        self.age = 1
        self.totalVisibleCount = 1
        self.consecutiveInvisibilityCount = 0
        self.isAssigned = False
        self.faceType = face.type


    def createKalman(self, bbox, dt):
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        # constant velocity
        # kalman = cv2.KalmanFilter(4, 2)
        # kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        # kalman.transitionMatrix = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        # kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
        # kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.00003
        # sp = np.array([x+w/2,y+h/2,0,0], dtype='float32')

        # acceleration
        kalman = cv2.KalmanFilter(6, 2)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], np.float32)
        kalman.transitionMatrix = np.array([[1, 0, dt, 0, 0.5*dt**2, 0], [0, 1, 0, dt, 0, 0.5*dt**2], [0, 0, 1, 0, dt, 0], [0, 0, 0, 1, 0, dt], [0,0,0,0,1,0], [0,0,0,0,0,1]], np.float32)
        kalman.processNoiseCov = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0,0,0,0,1,0], [0,0,0,0,0,1]], np.float32) * 0.03
        kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.00003
        sp = np.array([x+w/2,y+h/2,0,0,0,0], dtype='float32')
        kalman.statePost = np.array(sp, np.float32)

        return kalman

    def generateId(self):
        return np.random.randint(0,2**20)

    def updatebbox(self, bbox):
        self.bbox = bbox

    def setAge(self, age):
        self.age = age

    def getAge(self):
        return self.age

    def setTotalVisibleCount(self, totalVisibleCount):
        self.totalVisibleCount = totalVisibleCount

    def getTotalVisibleCount(self):
        return self.totalVisibleCount

    def setConsecutiveInvisibilityCount(self, consecutiveInvisibilityCount):
        self.consecutiveInvisibilityCount = consecutiveInvisibilityCount

    def getConsecutiveInvisibilityCount(self):
        return self.consecutiveInvisibilityCount

    def incrementAge(self):
        self.setAge(self.getAge() + 1)

    def incrementConsecutiveInvisibilityCount(self):
        self.setConsecutiveInvisibilityCount(self.getConsecutiveInvisibilityCount() + 1)

    def incrementTotalVisibleCount(self):
        self.setTotalVisibleCount(self.getTotalVisibleCount() + 1)

    def resetConsecutiveInvisibilityCount(self):
        self.setConsecutiveInvisibilityCount(0)

    def correctKalman(self, mp):
        self.kalman.correct(mp)

    def setIsAssigned(self, isAssigned):
        self.isAssigned = isAssigned

    def getIsAssigned(self):
        return self.isAssigned

    def getFaceType(self):
        return self.faceType

    def setFaceType(self, faceType):
        self.faceType = faceType