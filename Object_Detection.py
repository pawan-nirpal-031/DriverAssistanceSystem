import collections
from matplotlib.pyplot import box
import numpy as np
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import six
import cv2
import os
import tensorflow as tf
from tensorflow import keras
cwd = os.path.dirname(os.path.realpath(__file__))


STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange'
]


class ObjectDetection(object):
    def __init__(self):
        os.chdir(cwd)
        PATH_TO_SAVED_MODEL = "./ActiveModel"
        self.detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
        self.detection_graph = tf.Graph()

    def box_normal_to_pixel(self, box, dim):
        height, width = dim[0], dim[1]
        box_pixel = [int(box[0]*height), int(box[1]*width),
                     int(box[2]*height), int(box[3]*width)]
        return np.array(box_pixel)

    def draw_bounding_box_on_image(self, image,
                                   ymin,
                                   xmin,
                                   ymax,
                                   xmax,
                                   color='red',
                                   thickness=4,
                                   display_str_list=(),
                                   use_normalized_coordinates=True):
        draw = ImageDraw.Draw(image)
        im_width, im_height = image.size
        if use_normalized_coordinates:
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                          ymin * im_height, ymax * im_height)
        else:
            (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
        draw.line([(left, top), (left, bottom), (right, bottom),
                   (right, top), (left, top)], width=thickness, fill=color)
        try:
            font = ImageFont.truetype('arial.ttf', 24)
        except IOError:
            font = ImageFont.load_default()

        text_bottom = top
        for display_str in display_str_list[::-1]:
            text_width, text_height = font.getsize(display_str)
            margin = np.ceil(0.05 * text_height)
            draw.rectangle(
                [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                                  text_bottom)],
                fill=color)
            draw.text(
                (left + margin, text_bottom - text_height - margin),
                display_str,
                fill='black',
                font=font)
            text_bottom -= text_height - 2 * margin

    def draw_bounding_box_on_image_array(self, image,
                                         ymin,
                                         xmin,
                                         ymax,
                                         xmax,
                                         color='red',
                                         thickness=4,
                                         display_str_list=(),
                                         use_normalized_coordinates=True):
        image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
        self.draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color,
                                        thickness, display_str_list,
                                        use_normalized_coordinates)
        np.copyto(image, np.array(image_pil))

    def draw_bounding_boxes_on_image_array(self, image,
                                           boxes,
                                           color='red',
                                           thickness=4,
                                           display_str_list_list=()):
        image_pil = Image.fromarray(image)
        self.draw_bounding_boxes_on_image(image_pil, boxes, color, thickness,
                                          display_str_list_list)
        np.copyto(image, np.array(image_pil))

    def draw_bounding_boxes_on_image(self, image,
                                     boxes,
                                     color='red',
                                     thickness=4,
                                     display_str_list_list=()):

        boxes_shape = boxes.shape
        if not boxes_shape:
            return
        if len(boxes_shape) != 2 or boxes_shape[1] != 4:
            raise ValueError('Input must be of size [N, 4]')
        for i in range(boxes_shape[0]):
            display_str_list = ()
            if display_str_list_list:
                display_str_list = display_str_list_list[i]
            self.draw_bounding_box_on_image(image, boxes[i, 0], boxes[i, 1], boxes[i, 2],
                                            boxes[i, 3], color, thickness, display_str_list)

    def draw_keypoints_on_image_array(self, image,
                                      keypoints,
                                      color='red',
                                      radius=2,
                                      use_normalized_coordinates=True):
        image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
        self.draw_keypoints_on_image(image_pil, keypoints, color, radius,
                                     use_normalized_coordinates)
        np.copyto(image, np.array(image_pil))

    def draw_keypoints_on_image(self, image,
                                keypoints,
                                color='red',
                                radius=2,
                                use_normalized_coordinates=True):
        draw = ImageDraw.Draw(image)
        im_width, im_height = image.size
        keypoints_x = [k[1] for k in keypoints]
        keypoints_y = [k[0] for k in keypoints]
        if use_normalized_coordinates:
            keypoints_x = tuple([im_width * x for x in keypoints_x])
            keypoints_y = tuple([im_height * y for y in keypoints_y])
        for keypoint_x, keypoint_y in zip(keypoints_x, keypoints_y):
            draw.ellipse([(keypoint_x - radius, keypoint_y - radius),
                          (keypoint_x + radius, keypoint_y + radius)],
                         outline=color, fill=color)

    def draw_mask_on_image_array(self, image, mask, color='red', alpha=0.7):
        if image.dtype != np.uint8:
            raise ValueError('`image` not of type np.uint8')
        if mask.dtype != np.float32:
            raise ValueError('`mask` not of type np.float32')
        if np.any(np.logical_or(mask > 1.0, mask < 0.0)):
            raise ValueError('`mask` elements should be in [0, 1]')
        rgb = ImageColor.getrgb(color)
        pil_image = Image.fromarray(image)

        solid_color = np.expand_dims(
            np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
        pil_solid_color = Image.fromarray(
            np.uint8(solid_color)).convert('RGBA')
        pil_mask = Image.fromarray(np.uint8(255.0*alpha*mask)).convert('L')
        pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
        np.copyto(image, np.array(pil_image.convert('RGB')))

    def visualize_boxes_and_labels_on_image_array(self, image,
                                                  boxes,
                                                  classes,
                                                  scores,
                                                  category_index,
                                                  instance_masks=None,
                                                  keypoints=None,
                                                  use_normalized_coordinates=False,
                                                  max_boxes_to_draw=20,
                                                  min_score_thresh=.5,
                                                  agnostic_mode=False,
                                                  line_thickness=4):
        box_to_display_str_map = collections.defaultdict(list)
        box_to_color_map = collections.defaultdict(str)
        box_to_instance_masks_map = {}
        box_to_keypoints_map = collections.defaultdict(list)
        if not max_boxes_to_draw:
            max_boxes_to_draw = boxes.shape[0]
        for i in range(min(max_boxes_to_draw, boxes.shape[0])):
            if scores is None or scores[i] > min_score_thresh:
                box = tuple(boxes[i].tolist())
                if instance_masks is not None:
                    box_to_instance_masks_map[box] = instance_masks[i]
                if keypoints is not None:
                    box_to_keypoints_map[box].extend(keypoints[i])
                if scores is None:
                    box_to_color_map[box] = 'black'
                else:
                    if not agnostic_mode:
                        if classes[i] in category_index.keys():
                            class_name = category_index[classes[i]]
                        else:
                            class_name = 'N/A'
                        display_str = '{}: {}%'.format(
                            class_name,
                            int(100*scores[i]))
                    else:
                        display_str = 'score: {}%'.format(int(100 * scores[i]))
                    box_to_display_str_map[box].append(display_str)
                    if agnostic_mode:
                        box_to_color_map[box] = 'DarkOrange'
                    else:
                        box_to_color_map[box] = STANDARD_COLORS[
                            classes[i] % len(STANDARD_COLORS)]

        # Draw all boxes onto image.
        for box, color in six.iteritems(box_to_color_map):
            ymin, xmin, ymax, xmax = box
            if instance_masks is not None:
                self.draw_mask_on_image_array(
                    image,
                    box_to_instance_masks_map[box],
                    color=color
                )
            self.draw_bounding_box_on_image_array(
                image,
                ymin,
                xmin,
                ymax,
                xmax,
                color=color,
                thickness=line_thickness,
                display_str_list=box_to_display_str_map[box],
                use_normalized_coordinates=use_normalized_coordinates)
            if keypoints is not None:
                self.draw_keypoints_on_image_array(
                    image,
                    box_to_keypoints_map[box],
                    color=color,
                    radius=line_thickness / 2,
                    use_normalized_coordinates=use_normalized_coordinates)

    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    def get_localization(self, frame, detect_fn):
        input_tensor = tf.convert_to_tensor(frame)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]
        category_index = {1: {'id': 1, 'name': u'car'},
                          2: {'id': 2, 'name': u'pedestrian'},
                          3: {'id': 3, 'name': u'trafficLight-GreenLeft'},
                          4: {'id': 4, 'name': u'trafficLight-Green'},
                          5: {'id': 5, 'name': u'trafficLight-Red'},
                          6: {'id': 6, 'name': u'trafficLight-RedLeft'},
                          7: {'id': 7, 'name': u'trafficLight'},
                          8: {'id': 8, 'name': u'truck'},
                          9: {'id': 9, 'name': u'biker'},
                          10: {'id': 10, 'name': u'trafficLight-Yellow'},
                          11: {'id': 11, 'name': u'trafficLight-YellowLeft'}}
        # input_tensor = np.expand_dims(image_np, 0)
        detections = detect_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(
            np.int64)

        image_np_with_detections = frame.copy()

        self.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.40,
            agnostic_mode=False)
        boxes = np.squeeze(detections['detection_boxes'])
        classes = np.squeeze(detections['detection_classes'])
        scores = np.squeeze(detections['detection_scores'])

        cls = classes.tolist()

        obj_classes = [1, 2, 3, 4, 5, 6, 7, 10, 13, 14]
        idx_vec = [i for i, v in enumerate(cls) if (
            (v in obj_classes) and (scores[i] > 0.3))]

        if len(idx_vec) == 0:
            print('no detection!')
            self.Objkt_boxes = []
        else:
            tmp_Objkt_boxes = []
            for idx in idx_vec:
                dim = image_np_with_detections.shape[0:2]
                box = self.box_normal_to_pixel(boxes[idx], dim)
                box_h = box[2] - box[0]
                box_w = box[3] - box[1]
                ratio = box_h/(box_w + 0.01)
                if ((ratio < 0.8) and (box_h > 20) and (box_w > 20)):
                    tmp_Objkt_boxes.append(box)
                    print(box, ', confidence: ',
                          scores[idx], 'ratio:', ratio, "class : ", classes[idx])
                else:
                    tmp_Objkt_boxes.append(box)
                    print(' unfiltered box sizes ', box, ', confidence: ',
                          scores[idx], 'ratio:', ratio, "class : ", classes[idx])
            self.Objkt_boxes = tmp_Objkt_boxes
        return self.Objkt_boxes, classes, scores


# testing Object detection

class Simulation():  # Simulte od

    def __init__(self):
        pass

    def DistanceEstimation(self, boxes, classes, scores, np_img):
        threshold_dist = 500
        for i in range(len(boxes)):
            if classes[i] == 1 or classes[i] == 2 or classes[i] == 8 or classes[i] == 9:
                if scores[i] >= 0.40:
                    mid_x = (boxes[i][1]+boxes[i][3])/2
                    mid_y = (boxes[i][0]+boxes[i][2])/2
                    apx_distance = round(
                        ((1 - (boxes[i][3] - boxes[i][1]))**2), 1)
                    cv2.putText(np_img, '{}'.format(apx_distance), (int(
                        mid_x*420), int(mid_y*220)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    if apx_distance <= threshold_dist:
                        cv2.putText(np_img, 'WARNING !!!', (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
