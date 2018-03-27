
# General Parameters
FPS = 30
DELAY = 1/FPS
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Detector parameters

# Downscale factor of each captured frame used in detection and classifier
RESCALE_FACTOR = 1/4
# First number specifies the amount of consecutive frames we use the haar cascade, second number the amount of frame we don't use the cascade (relies solely on Tracker)
CASCADE_USAGE = [4, 2]
# Haar cascade -  Parameter specifying how much the image size is reduced at each image scale
SCALE_FACTOR_CASCADE = 1.3
# Haar cascade - Parameter specifying how many neighbors each candidate rectangle should have to retain it
MIN_NEIGHBORS = 3
# Haar cascade - pretrained cascade path files
FRONTAL_CASCADE_PATH = 'Cascades/haarcascade_frontalface_default.xml'
PROFILE_CASCADE_PATH = 'Cascades/haarcascade_profileface.xml'


# Tracker parameters

# max distance (pixels) between 2 objects to still be associated with the same object
MAX_DISTANCE_TRACKER = 50
# max frames the object can be invisible to before deletion
INVISIBILITY_THRESHOLD = 25
# Visibility ratio an object needs to have before it is shown on the screen
VISIBILITY_RATIO = 0.25
# min age of a track to be shown
AGE_THRESHOLD = 5

# Classifier parameters

# model path
MODEL = 'models/model_best.h5'
# input dimensions image for classifier
I_WIDTH_CLASSIFIER = 160
I_HEIGHT_CLASSIFIER = 220
CLASS_NAMES = ["Fabrice", "Jeffrey","Pieter","Undefined"]

# Visualiser paramerers

# Specifies how many faces will be kept in the buffer, beyond this number we flush (performance issues)
FLUSH = 75
# Window name of visualiser
WINDOW_NAME = 'Visualization'
# Color of the rectangle which is drawn around the faces
BBOX_COLOR = (0, 0, 255)
# Color of the text which indicates the class of the faces
TEXT_COLOR = (255, 255, 255)

