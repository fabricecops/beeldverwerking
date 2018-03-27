import cv2
from multiprocessing import Queue
import SystemParameters as sp


class Visualizer:
    def __init__(self, predictionQueue, frameQueue):
        self.predictionQueue = predictionQueue
        self.frameQueue = frameQueue
        self.garbageQueue = Queue()
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.faces = []

    def visualize(self):
        superFrame = self.frameQueue.get()
        frame, frameNumber = superFrame.getFrame(), superFrame.getFrameNumber()

        self.faces = []
        while self.garbageQueue.qsize() > 0:
            self.faces.append(self.garbageQueue.get())

        # flush garbage faces if it has over 50 (performance reasons)
        if(len(self.faces) > sp.FLUSH):
            self.faces = []

        while self.predictionQueue.qsize() > 0:
            self.faces.append(self.predictionQueue.get())

        # self.filterOverlappingRectangles(frameNumber)

        for face in self.faces:
            if(face.getFrameNumber() == frameNumber):
                x, y, w, h = face.getBbox()
                cv2.rectangle(frame, (x, y), (x + w, y + h), sp.BBOX_COLOR, 2)
                text = face.getClassName()
                cv2.putText(frame, text, (x, y + h), self.font,
                            1, sp.TEXT_COLOR, 2, cv2.LINE_AA)
            else:
                self.garbageQueue.put(face)

                cv2.putText(frame, str(
                    frameNumber), (frame.shape[0], 30), self.font, 1, sp.TEXT_COLOR, 1, cv2.LINE_AA)
        cv2.imshow(sp.WINDOW_NAME, frame)
        cv2.waitKey(1)

    def filterOverlappingRectangles(self, frameNumber):
        nonOverlappingFaces = []
        for face in self.faces:
            overlapChecked = False
            for nface in nonOverlappingFaces:
                if(not nface.overlaps(face, frameNumber) and not overlapChecked):
                    nonOverlappingFaces.append(face)
                    overlapChecked = True

        self.faces = nonOverlappingFaces
