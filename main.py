from multiprocessing import Process, Queue
import time
from Processes.FaceClassifier import FaceClassifier
from Processes.FaceDetector import FaceDetector
from HelperClasses.SuperFrame import SuperFrame
from Processes.Visualizer import Visualizer
import SystemParameters as sp
import cv2


def visualizeProcess(visualizer):
    while True:
        visualizer.visualize()


def faceDetectionProcess(detector):
    # initialize Cascades here, as these are not serializable as this makes for problems
    detector.initCascades()
    f = open('detectionTime.txt', 'w')
    while True:
        if(not detector.readQueue.empty()):
            detector.detectFaces()

def faceClassificationProcess(classifier):
    # load Keras here, otherwise problems with memory occur due to multiple loads of keras
    classifier.initModel()
    while True:
        if(not classifier.readQueue.empty()):
            classifier.classify()


# defines all processes and starts them
def initProcesses(detector, classifier, visualizer):
    classify_p = Process(target=faceClassificationProcess, args=(classifier,))
    classify_p.daemon = True
    classify_p.start()

    detect_p = Process(target=faceDetectionProcess, args=(detector,))
    detect_p.daemon = True
    detect_p.start()

    visualize_p = Process(target=visualizeProcess, args=(visualizer,))
    visualize_p.daemon = True
    visualize_p.start()

    return [detect_p, classify_p, visualize_p]


def terminateProcesses(process_list):
    for process in process_list:
        process.terminate()


def stopQueues(queue_list):
    for queue in queue_list:
        queue.close()


if __name__ == '__main__':

    cap = cv2.VideoCapture("testvideo.mp4")
    cap.set(3, sp.FRAME_WIDTH)
    cap.set(4, sp.FRAME_HEIGHT)
    delay = sp.DELAY
    frameIndex = 0

    # define the queues
    queue_to_detector = Queue()
    queue_to_classifier = Queue()
    queue_to_output = Queue()
    queue_to_visualizer = Queue()
    qs = [queue_to_detector, queue_to_classifier,
          queue_to_output, queue_to_visualizer]

    # make the classes with the correct queues
    detector = FaceDetector(queue_to_detector, queue_to_classifier)
    classifier = FaceClassifier(queue_to_classifier, queue_to_output)
    visualizer = Visualizer(queue_to_output, queue_to_visualizer)

    # starts processes
    ps = initProcesses(detector, classifier, visualizer)

    startTime = time.time()
    try:
        while (cap.isOpened()):
            if (time.time() - startTime >= delay):
                ret, frame = cap.read()
                frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                superFrame = SuperFrame(frame, frameIndex)
                queue_to_detector.put(superFrame)
                queue_to_visualizer.put(superFrame)
                frameIndex += 1
                #print(1/(time.time()-startTime))
                startTime = time.time()
    except KeyboardInterrupt:
        terminateProcesses(ps)
        stopQueues(qs)
        cap.release()
        cv2.destroyAllWindows()
    print('face classification and detection terminated')
