# import glob
import Object_Detection
import Helpers
import cv2
from PIL import Image
import glob
img_path = [file for file in glob.glob(
    '.\\Datasets\\image_datasets\\problem_datasets\\*.jpg')]


det = Object_Detection.ObjectDetection()


for i in range(len(img_path)):
    img = cv2.imread(img_path[i])
    boxes, classes, scores = det.get_localization(img, det.detect_fn)
    for j in range(len(boxes)):
        box = boxes[j]
        oclass = classes[j]
        score = scores[j]
        Helpers.draw_box_label(img, box, (0, 255, 255), oclass, score, True)
        cv2.imshow("test", img)
    cv2.waitKey(1000)


# print(scores)
