import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from sklearn.utils.linear_assignment_ import linear_assignment
from Helpers import *
import Helpers
import Object_Tracking
import time


frame_count = 0
max_age = 4
min_hits = 1
tracker_list = []
track_id_list = deque(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'])
debug = False


def assign_detections_to_trackers(trackers, detections, iou_thrd=0.3):
    IOU_mat = np.zeros((len(trackers), len(detections)), dtype=np.float32)
    for t, trk in enumerate(trackers):
        for d, det in enumerate(detections):
            IOU_mat[t, d] = box_iou2(trk, det)
    matched_idx = linear_assignment(-IOU_mat)
    print('matched indices ', type(matched_idx), ' ', matched_idx)
    unmatched_trackers, unmatched_detections = [], []
    for t, trk in enumerate(trackers):
        if(t not in matched_idx[:, 0]):
            unmatched_trackers.append(t)
    for d, det in enumerate(detections):
        if(d not in matched_idx[:, 1]):
            unmatched_detections.append(d)
    matches = []
    for m in matched_idx:
        if(IOU_mat[m[0], m[1]] < iou_thrd):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if(len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def DetectionAndTrackingIntegration(img, det):
    global frame_count
    global tracker_list
    global max_age
    global min_hits
    global track_id_list
    global debug

    frame_count += 1

    img_dim = (img.shape[1], img.shape[0])
    start = time.time()
    z_box, classes, scores = det.get_localization(
        img, det.detect_fn)  # measurement
    end = time.time()
    print("detection time : ", end-start, " secs")
    if debug:
        print('Frame:', frame_count)

    x_box = []
    if debug:
        for i in range(len(z_box)):
            img1 = Helpers.draw_box_label(img, z_box[i], box_color=(255, 0, 0))
            plt.imshow(img1)
        plt.show()

    if len(tracker_list) > 0:
        for trk in tracker_list:
            x_box.append(trk.box)

    matched, unmatched_dets, unmatched_trks \
        = assign_detections_to_trackers(x_box, z_box, iou_thrd=0.3)
    if debug:
        print('Detection: ', z_box)
        print('x_box: ', x_box)
        print('matched:', matched)
        print('unmatched_det:', unmatched_dets)
        print('unmatched_trks:', unmatched_trks)

    if matched.size > 0:
        for trk_idx, det_idx in matched:
            z = z_box[det_idx]
            z = np.expand_dims(z, axis=0).T
            tmp_trk = tracker_list[trk_idx]
            tmp_trk.kalman_filter(z)
            xx = tmp_trk.x_state.T[0].tolist()
            xx = [xx[0], xx[2], xx[4], xx[6]]
            x_box[trk_idx] = xx
            tmp_trk.box = xx
            tmp_trk.hits += 1
            tmp_trk.no_losses = 0
            tmp_trk.obj_class = classes[det_idx]
            tmp_trk.score = scores[det_idx]

    if len(unmatched_dets) > 0:
        for idx in unmatched_dets:
            z = z_box[idx]
            z = np.expand_dims(z, axis=0).T
            tmp_trk = Object_Tracking.Tracker()  # Create a new tracker
            x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
            tmp_trk.x_state = x
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx = [xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.box = xx
            tmp_trk.obj_class = classes[idx]
            if(len(track_id_list)):
                tmp_trk.id = track_id_list.popleft()
                tracker_list.append(tmp_trk)
            x_box.append(xx)

    if len(unmatched_trks) > 0:
        for trk_idx in unmatched_trks:
            tmp_trk = tracker_list[trk_idx]
            tmp_trk.no_losses += 1
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx = [xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.box = xx
            x_box[trk_idx] = xx

    good_tracker_list = []

    for trk_idx in range(len(tracker_list)):
        trk = tracker_list[trk_idx]
        if ((trk.hits >= min_hits) and (trk.no_losses <= max_age)):
            good_tracker_list.append(trk)
            x_cv2 = trk.box
            if debug:
                print('updated box: ', x_cv2)
            car = (0, 255, 255)
            pedestrain = (0, 255, 0)
            truck = (0, 0, 255)
            biker = (255, 0, 0)
            paint = car
            if(trk.obj_class == 2):
                paint = pedestrain
            elif(trk.obj_class == 8):
                paint = truck
            elif(trk.obj_class == 9):
                paint = biker
            img = Helpers.draw_box_label(img, x_cv2, paint, trk.obj_class,trk.score)
    deleted_tracks = filter(lambda x: x.no_losses > max_age, tracker_list)
    for trk in deleted_tracks:
        track_id_list.append(trk.id)
    tracker_list = [x for x in tracker_list if x.no_losses <= max_age]
    if debug:
        print('Ending tracker_list: ', len(tracker_list))
        print('Ending good tracker_list: ', len(good_tracker_list))

    return img, classes, scores, z_box
