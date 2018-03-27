import numpy as np

from HelperClasses.FaceType import FaceType
from HelperClasses.SuperFrame import SuperFrame
from Tracker.Track import Track
from Tracker.hungarian import Hungarian
import SystemParameters as sp


class Tracker:
    def __init__(self):
        self.tracks = []
        self.invisibilityThreshold = sp.INVISIBILITY_THRESHOLD
        self.ageThreshold = sp.AGE_THRESHOLD
        self.visibilityThreshold = sp.VISIBILITY_RATIO
        self.removeCost = sp.MAX_DISTANCE_TRACKER
        self.dt = sp.DELAY

    def addNewTrack(self, face):
        self.tracks.append(Track(face, self.dt))

    def removeTrack(self, track):
        self.tracks.remove(track)

    def getAssignedTracks(self):
        counter = 0
        for track in self.tracks:
            if(track.getIsAssigned()):
                counter += 1

        return counter

    def getUnassignedTracks(self):
        counter = 0
        for track in self.tracks:
            if (not track.getIsAssigned()):
                counter += 1

        return counter

    def predictLocationTracks(self):
        for track in self.tracks:
            bbox = track.bbox
            predictCenter = track.kalman.predict()
            newbbox = [int(predictCenter[0]),int(predictCenter[1]), bbox[2], bbox[3]]
            track.updatebbox(newbbox)

    def assignTracks(self, detections):
        nTracks = len(self.tracks)
        nDetections = len(detections)
        #print("det: ",nDetections)
        costMatrix = np.zeros((nTracks, nDetections))
        neglectIds = []
        i = 0
        for track in self.tracks:
            j = 0
            neglectIndex = 0
            for detect in detections:
                bboxTrack = track.bbox
                bboxDetect = detect.bbox
                distance = self.distance(bboxTrack, bboxDetect)
                costMatrix[i, j] = distance
                if(distance >= self.removeCost and track.getAge() > self.ageThreshold):
                    neglectIndex += 1
                j += 1
            if(neglectIndex == j):
                neglectIds.append(i)
            i += 1

        hungarian = Hungarian()
        hungarian.calculate(costMatrix)
        optimalSolution = hungarian.get_results()

        # correct the bbox for the appropriate solution from the hungarian matrix
        for solution in optimalSolution:
            trackId = solution[0]
            detectId = solution[1]

            # only if they are not in the neglect array (as these solutions are unfeasible)
            if trackId not in neglectIds:
                track = self.tracks[trackId]
                newbbox = detections[detectId].bbox
                mp = np.array([[np.float32(newbbox[0])], [np.float32(newbbox[1])]])
                track.correctKalman(mp)
                track.updatebbox(newbbox)
                track.incrementAge()
                track.incrementTotalVisibleCount()
                track.resetConsecutiveInvisibilityCount()
                track.setIsAssigned(True)

        # increment the age and consecutive invisibility count of unassigned tracks
        for track in self.tracks:
            if(not track.getIsAssigned()):
                track.incrementAge()
                track.incrementConsecutiveInvisibilityCount()

        # create a new track for the detections that were not assigned to a track
        nUnassignedDetections = nDetections - (self.getAssignedTracks() + self.getUnassignedTracks())
        for i in range(0, nUnassignedDetections):
            self.addNewTrack(detections[i])

        # set all tracks back at unassigned
        for track in self.tracks:
            track.setIsAssigned(False)

    def deleteOldTracks(self):
        if(self.tracks):
            garbageTracks = []
            for track in self.tracks:
                visibility = track.totalVisibleCount / track.age
                if((visibility < self.visibilityThreshold) or (track.consecutiveInvisibilityCount >= self.invisibilityThreshold)):
                    garbageTracks.append(track)
                    print('removed track')

            for track in garbageTracks:
                self.removeTrack(track)

    def getFaces(self, superFrame, rescale):
        frame, frameNumber = superFrame.getFrame(), superFrame.getFrameNumber()
        faces = []
        for track in self.tracks:
            if(track.age >= self.ageThreshold):
                bbox =  np.array(track.bbox)/rescale
                bbox = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                x, y, w, h = bbox
                faceFrame = SuperFrame(frame[y:y + h, x:x + w], frameNumber)
                faces.append(FaceType(bbox, faceFrame, track.getFaceType()))

        return faces

    def distance(self, bbox1, bbox2):
        return np.sqrt((bbox1[0]-bbox2[0])**2 + (bbox1[1]-bbox2[1])**2)
