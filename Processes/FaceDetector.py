import cv2
import numpy as np
from HelperClasses.FaceType import FaceType
from HelperClasses.SuperFrame import SuperFrame
from Tracker.Tracker import Tracker
import SystemParameters as sp


class FaceDetector:
    def __init__(self, readQueue, writeQueue):
        self.delay = sp.DELAY
        self.rescaleFactor = sp.RESCALE_FACTOR
        self.cascadeUsage = sp.CASCADE_USAGE
        self.scaleFactorCascade = sp.SCALE_FACTOR_CASCADE
        self.minNeighbors = sp.MIN_NEIGHBORS
        self.tracker = Tracker()
        self.readQueue = readQueue
        self.writeQueue = writeQueue
        self.firstDetection = False
        self.frontal_face_cascade = None
        self.profile_face_cascade = None
        self.faces = []

    def initCascades(self):
        # initialize Cascades, separate function as the cascasdes are not serializable
        if self.frontal_face_cascade is None and self.profile_face_cascade is None:
            self.frontal_face_cascade = cv2.CascadeClassifier(
                sp.FRONTAL_CASCADE_PATH)
            self.profile_face_cascade = cv2.CascadeClassifier(
                sp.PROFILE_CASCADE_PATH)

    def removeFaces(self):
        self.faces = []

    def addFaces(self, bboxes, superFrame, type):
        frame, frameNumber = superFrame.getFrame(), superFrame.getFrameNumber()
        for bbox in bboxes:
            [x, y, w, h] = bbox
            superFrame.setFrame(frame[y:y + h, x:x + w])
            self.faces.append(FaceType(bbox, superFrame, type))

    def useCascaseClassifier(self, frameNumber):
        sum = self.cascadeUsage[0] + self.cascadeUsage[1]
        modSum = np.mod(frameNumber, sum)
        if(modSum <= self.cascadeUsage[0]):
            return True
        else:
            return False

    def detectFaces(self):
        superFrame = self.readQueue.get()
        frame, frameNumber = superFrame.getFrame(), superFrame.getFrameNumber()
        frontal_faces, profile_faces = [], []

        # only detect faces using haarcascades every x-th frame
        if(self.useCascaseClassifier(frameNumber)):
            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                              (0, 0), fx=self.rescaleFactor, fy=self.rescaleFactor)
            frontal_faces = self.frontal_face_cascade.detectMultiScale(
                gray, self.scaleFactorCascade, self.minNeighbors)
            # if no frontal faces detected, use the profile cascade
            if(len(frontal_faces) == 0):
                profile_faces = self.profile_face_cascade.detectMultiScale(
                    gray, self.scaleFactorCascade, self.minNeighbors)
            else:
                profile_faces = []
        else:
            frontal_faces, profile_faces = [], []

        self.addFaces(frontal_faces, SuperFrame(frame, frameNumber), 'FRONTAL')
        self.addFaces(profile_faces, SuperFrame(frame, frameNumber), 'PROFILE')

        faces = []
        # kalman filter for tracking
        if(len(self.faces) > 0 or self.firstDetection):
            self.firstDetection = True
            self.tracker.assignTracks(self.faces)
            self.tracker.predictLocationTracks()
            self.tracker.deleteOldTracks()
            faces = self.tracker.getFaces(SuperFrame(
                frame, frameNumber), self.rescaleFactor)

        # write detected faces to the queue for the classifier
        for face in faces:
            self.writeQueue.put(face)
        self.removeFaces()
