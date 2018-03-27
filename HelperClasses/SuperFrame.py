class SuperFrame:
    def __init__(self, frame, frameNumber):
        self.frame = frame
        self.frameNumber = frameNumber

    def getFrame(self):
        return self.frame

    def setFrame(self, frame):
        self.frame = frame

    def getFrameNumber(self):
        return self.frameNumber

    def setFrameNumber(self, frameNumber):
        self.frameNumber = frameNumber