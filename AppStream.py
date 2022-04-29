from tkinter import *
from PIL import ImageTk, Image
import cv2
from matplotlib.pyplot import text
import os
from Main import DetectionAndTrackingIntegration
import Object_Detection
from Object_Detection import Simulation
from tkinter import filedialog


# '.' + os.sep + 'Datasets' + os.sep + 'video_datasets' + os.sep + '*.mp4'

root = Tk()
# Create a frame
app = Frame(root, bg="white")
app.grid()
root.geometry("1300x750")
# Create a label in the frame
lmain = Label(app)
lmain.grid()


det = Object_Detection.ObjectDetection()


def choose_file():
    filename = filedialog.askopenfilename()
    return cv2.VideoCapture(filename)


button = Button(text="Open a video file", command=choose_file)
button.grid()

cap = choose_file()


def video_stream():
    _, frame = cap.read()
    image_box, classes, scores, boxes = DetectionAndTrackingIntegration(frame, det)
    #Simulation().DistanceEstimation(boxes, classes, scores, image_box)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(40, video_stream)


# function for video streaming



try:
    detect_button = Button(text="Start detections", command=video_stream)
    detect_button.grid()
    root.mainloop()
except Exception:
    print("unable to open video ")
