# import the necessary packages
from imutils.video import VideoStream, FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import RPi.GPIO as gpio
#######################################################
# Constants for distance estimation (in inches) ***
KNOWN_DISTANCE_CAR = 25  # Example: Distance from camera to car (in inches)
KNOWN_WIDTH_CAR = 6  # Example: Width of the car (in inches)
KNOWN_DISTANCE_PERSON = 25  # Example: Distance from camera to person (in inches)
KNOWN_WIDTH_PERSON = 3.5  # Example: Width of a person (shoulder width) (in inches)

# Focal length can be pre-calculated or measured for the specific camera
focalLength_car = (KNOWN_DISTANCE_CAR * 576) / KNOWN_WIDTH_CAR
focalLength_person = (KNOWN_DISTANCE_PERSON * 336) / KNOWN_WIDTH_PERSON

# Function to calculate distance
def distance_to_camera(knownWidth, focalLength, pixelWidth):
    return (knownWidth * focalLength) / pixelWidth

GPIO_TRIGGER=3
GPIO_ECHO=2

in11 = 17
in12 = 22
en = 25
in21=23
in22=24
object_captured=0

gpio.setmode(gpio.BCM)
gpio.setup(in11,gpio.OUT)
gpio.setup(in12,gpio.OUT)
gpio.setup(in21,gpio.OUT)
gpio.setup(in22,gpio.OUT)
gpio.setup(en,gpio.OUT)
#set GPIO direction for Ultrasonic Sensors (IN / OUT)
gpio.setup(GPIO_TRIGGER, gpio.OUT)#trigger
gpio.setup(GPIO_ECHO, gpio.IN)#echo

gpio.output(in11,gpio.LOW)
gpio.output(in12,gpio.LOW)
gpio.output(in21,gpio.LOW)
gpio.output(in22,gpio.LOW)

p=gpio.PWM(en,1000)

p.start(30)

def forward():
    #print ("forward")
    gpio.output(in11,gpio.HIGH)
    gpio.output(in12,gpio.LOW)
    gpio.output(in21,gpio.LOW)
    gpio.output(in22,gpio.LOW)
    
def stop():
    global object_captured
    #print ("stop")
    gpio.output(in11,gpio.LOW)
    gpio.output(in12,gpio.LOW)
    gpio.output(in21,gpio.LOW)
    gpio.output(in22,gpio.LOW)
    object_captured = 0

def reverse(sec):
    gpio.output(in11,gpio.LOW)
    gpio.output(in12,gpio.HIGH)
    gpio.output(in21,gpio.LOW)
    gpio.output(in22,gpio.LOW)
    time.sleep(sec)
    #gpio.cleanup()
def left_turn(sec):
    gpio.output(in11,gpio.HIGH)
    gpio.output(in12,gpio.LOW)
    gpio.output(in21,gpio.HIGH)
    gpio.output(in22,gpio.LOW)
    time.sleep(sec)
    #gpio.cleanup()
def right_turn(sec):
    gpio.output(in11,gpio.HIGH)
    gpio.output(in12,gpio.LOW)
    gpio.output(in21,gpio.LOW)
    gpio.output(in22,gpio.HIGH)
    time.sleep(sec)
    #gpio.cleanup()

def distance():
  time.sleep(0.15)
  gpio.setmode(gpio.BCM)
  # set Trigger to HIGH
  gpio.output(GPIO_TRIGGER, True)
 
  # set Trigger after 0.01ms to LOW
  time.sleep(0.00001)
  gpio.output(GPIO_TRIGGER, False)
 
  StartTime = time.time()
  StopTime = time.time()
 
  # save StartTime
  while gpio.input(GPIO_ECHO) == 0:
      StartTime = time.time()
 
  # save time of arrival
  while gpio.input(GPIO_ECHO) == 1:
      StopTime = time.time()
 
  # time difference between start and arrival
  TimeElapsed = StopTime - StartTime
  # multiply with the sonic speed (34300 cm/s)
  # and divide by 2, because there and back
  distance = (TimeElapsed * 34300) / 2
  #avgDistance=avgDistance+distance
 
  #avgDistance = avgDistance/5
  return distance

#######################################################

def steer_around_obstacle():
    # Threshold distance for obstacle detection (in centimeters)
    obstacle_threshold = 60

    # Check the distance to the obstacle
    dist = distance()

    # If the obstacle is within the threshold distance
    if dist < obstacle_threshold:
        # Turn left and check the distance
        stop()
        reverse(1.5)
        left_turn(1.5)
        dist_left = distance()

        # Turn right twice and check the distance
        reverse(1.5)
        right_turn(1.5)
        dist_right = distance()

        # Choose the direction with more free space
        if dist_left > dist_right:
            reverse(1.5)
            left_turn(1.5)
            forward()
        else:
            forward()

    else:
        forward()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-u", "--movidius", type=bool, default=0,
	help="boolean indicating if the Movidius should be used")
args = vars(ap.parse_args())
# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
# specify the target device as the Myriad processor on the NCS
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
fps = FPS().start()

frame_skip = 5
frame_count = 0

processed_frames = 0
min_action_interval = 0.2  # Set a minimum interval between function calls (in seconds)
last_action_time = time.time()
start_time = time.time()
# loop over the frames from the video stream
try:
    while True:
        #time.sleep(0.175)
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()
        frame_count += 1
        processed_frames += 1
        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > args["confidence"]:
                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                # draw the prediction on the frame
                label = "{}: {:.2f}%".format(CLASSES[idx],
                    confidence * 100)
                if CLASSES[idx] in ["person", "car"]:
                    pixelWidth = endX - startX
                    if CLASSES[idx] == "person":
                        distance_camera = distance_to_camera(KNOWN_WIDTH_PERSON, focalLength_person, pixelWidth)
                    elif CLASSES[idx] == "car":
                        distance_camera = distance_to_camera(KNOWN_WIDTH_CAR, focalLength_car, pixelWidth)
                    # Add the distance to the label
                    elapsed_time = time.time() - start_time
                    current_fps = processed_frames / elapsed_time
                    label = "{}: {:.2f}% - {:.2f}in - {:.2f}fps".format(CLASSES[idx], confidence * 100, distance_camera, current_fps)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                object_captured = 1 ################################################
        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        # update the FPS counter
        fps.update()
        dist = distance()
        #if frame_count % frame_skip == 0:
            #print ("Measured Distance = %.1f cm" % dist)
            #time.sleep(0.1)
        if time.time() - last_action_time > min_action_interval:
            steer_around_obstacle()
            last_action_time = time.time()
except KeyboardInterrupt:
    print("Interrupted")
except Exception as e:
    print("Unexpected error:", e)
finally:
    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
    gpio.cleanup()    
    #dist = distance()
    # if frame_count % frame_skip == 0:
    #      print ("Measured Distance = %.1f cm" % dist)
    #time.sleep(0.1)
    #if (object_captured == 1 and dist < 30) or dist < 15:
        #print ("stop")
        #stop()
    #else:
        #print ("forward")
        #forward()
# stop the timer and display FPS information
#fps.stop()
#print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
#print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
#gpio.cleanup()
#cv2.destroyAllWindows()
#vs.stop()
