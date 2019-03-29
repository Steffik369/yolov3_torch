import sys
import time
from PIL import Image, ImageDraw
# from models.tiny_yolo import TinyYoloNet
from utils import *
from image import letterbox_image, correct_yolo_boxes
from darknet import Darknet
import train
import cv2
import json
import argparse

from pycocotools import coco
from pycocotools import cocoeval

namesfile = None


def main():
    data_images = FLAGS.files
    cfgfile = FLAGS.config
    annotation_file = FLAGS.annotation_file
    weightfile = FLAGS.weights
    dataloader = load_images(data_images, 1, 1)

    global m
    m = Darknet(cfgfile)
    m.print_network()
    m.load_weights(weightfile)

    global json_data

    json_data = []

    for i, (img_paths, img) in enumerate(dataloader):
        json_data = detect_cv2(cfgfile, weightfile, img_paths[0], json_data, annotation_file)

    with open("data_result_test.json", "r+") as read_file:
        read_file.seek(0)
        read_file.write(json.dumps(json_data))
        read_file.truncate()

    coco_gt = coco.COCO(annotation_file=annotation_file)
    coco_dt = coco_gt.loadRes("data_result_test.json")
    coco_res = cocoeval.COCOeval(coco_gt, coco_dt, "bbox")
    coco_res.evaluate()
    coco_res.accumulate()
    coco_res.summarize()


def detect_cv2(cfgfile, weightfile, imgfile, json_data, annotation_file):
    import cv2
    namesfile = FLAGS.data

    use_cuda = True
    if use_cuda:
        m.cuda()

    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    class_names = load_class_names(namesfile)

    width = img.shape[1]
    height = img.shape[0]

    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int(round((box[0] - box[2] / 2.0) * width))
        y1 = int(round((box[1] - box[3] / 2.0) * height))
        x2 = int(round((box[0] + box[2] / 2.0) * width))
        y2 = int(round((box[1] + box[3] / 2.0) * height))

        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]

            if eval:
                image_id_file = ""

                with open(annotation_file, "r+") as read_file:
                    data = json.load(read_file)

                    for i in data["images"]:
                        if i['file_name'] in imgfile:
                            image_id_file = i["id"]
                            break

                det_box = {
                    "image_id": image_id_file,
                    "category_id": int(cls_id + 1),
                    "bbox": [
                        x1,
                        y1,
                        (x2 - x1),
                        (y2 - y1)
                    ],
                    "score": cls_conf
                }

                json_data.append(det_box)

    return json_data

class load_images():  # for inference
    import cv2

    def __init__(self, path, batch_size=1, img_size=416):
        import glob

        if os.path.isdir(path):
            self.files = sorted(glob.glob('%s/*.*' % path))
        elif os.path.isfile(path):
            self.files = [path]

        self.nF = len(self.files)  # number of image files
        self.nB = math.ceil(self.nF / batch_size)  # number of batches
        self.batch_size = batch_size
        self.height = img_size

        self.nF = len(self.files)  # number of image files

        assert self.nF > 0, 'No images found in path %s' % path

        # RGB normalization values
        # self.rgb_mean = np.array([60.134, 49.697, 40.746], dtype=np.float32).reshape((3, 1, 1))
        # self.rgb_std = np.array([29.99, 24.498, 22.046], dtype=np.float32).reshape((3, 1, 1))

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nB:
            raise StopIteration
        img_path = self.files[self.count]

        # Read image
        img = cv2.imread(img_path)  # BGR

        # Padded resize

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        # img -= self.rgb_mean
        # img /= self.rgb_std
        img /= 255.0

        return [img_path], img

    def __len__(self):
        return self.nB  # number of batches

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='data/toplogo.names', help='path to data.cfg')
    parser.add_argument('--files', '-f', type=str, default='data\\toplogo\\images\\test', help='path to image directory')
    parser.add_argument('--annotation_file', '-a', type=str, default='data/coco/annotations/test.json', help='path to image directory')
    parser.add_argument('--config', '-c', type=str, default='cfg/yolov3.cfg', help='network configuration file')
    parser.add_argument('--weights', '-w', type=str, default='weights/yolov3.weights', help='initial weights file')

    FLAGS, _ = parser.parse_known_args()
    main()

