from multiprocessing import Process, Queue
import time
from Processes.FaceDetector import FaceDetector
from Processes.ImageWriter import ImageWriter
from HelperClasses.SuperFrame import SuperFrame
import SystemParameters as sp
import cv2


def imageWriterProcess(imageWriter):
    while True:
        imageWriter.writeImages()

def faceDetectionProcess(detector):
    # initialize Cascades here, as these are not serializable as this makes for problems
    detector.initCascades()
    while True:
        if(not detector.readQueue.empty()):
            detector.detectFaces()


# defines all processes and starts them
def initProcesses(detector, imageWriter):
    detect_p = Process(target=faceDetectionProcess, args=(detector,))
    detect_p.daemon = True
    detect_p.start()

    writer_p = Process(target=imageWriterProcess, args=(imageWriter,))
    writer_p.daemon = True
    writer_p.start()

    return [detect_p, writer_p]


def terminateProcesses(process_list):
    for process in process_list:
        process.terminate()


def stopQueues(queue_list):
    for queue in queue_list:
        queue.close()


if __name__ == '__main__':

    cap = cv2.VideoCapture(0)
    delay = sp.DELAY
    frameIndex = 0

    # define the queues
    queue_to_detector = Queue()
    queue_to_imageWriter = Queue()
    qs = [queue_to_detector, queue_to_imageWriter]

    # make the classes with the correct queues
    detector = FaceDetector(queue_to_detector, queue_to_imageWriter)
    imageWriter = ImageWriter(queue_to_imageWriter, 'DataGenerator\FaceGeneratorData\Jeffrey\Jeffrey1')

    # starts processes
    ps = initProcesses(detector, imageWriter)

    startTime = time.time()
    try:
        while (cap.isOpened()):
            if (time.time() - startTime >= delay):
                ret, frame = cap.read()
                superFrame = SuperFrame(frame, frameIndex)
                queue_to_detector.put(superFrame)
                frameIndex += 1
                #print(1/(time.time()-startTime))
                startTime = time.time()
    except KeyboardInterrupt:
        terminateProcesses(ps)
        stopQueues(qs)
        cap.release()
        cv2.destroyAllWindows()
    print('face classification and detection terminated')
