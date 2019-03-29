import sys
import time
from PIL import Image, ImageDraw
#from models.tiny_yolo import TinyYoloNet
from utils import *
from image import letterbox_image, correct_yolo_boxes
from darknet import Darknet
import train
import cv2
import json
import argparse

from pycocotools import coco
from pycocotools import cocoeval

namesfile=None

def main():
    data_images    = FLAGS.files
    cfgfile    = FLAGS.config
    weightfile = FLAGS.weights
    dataloader = load_images(data_images, 1, 1)

    global m
    m = Darknet(cfgfile)
    m.print_network()
    m.load_weights(weightfile)

    for i, (img_paths, img) in enumerate(dataloader):
        detect_cv2(cfgfile, weightfile, img_paths[0],)


def detect(cfgfile, weightfile, imgfile):
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    # if m.num_classes == 20:
    #     namesfile = 'data/voc.names'
    # elif m.num_classes == 80:
    namesfile = FLAGS.data
    # else:
    #     namesfile = 'data/names'
    
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        m.cuda()

    img = Image.open(imgfile).convert('RGB')
    sized = letterbox_image(img, m.width, m.height)

    start = time.time()
    boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
    correct_yolo_boxes(boxes, img.width, img.height, m.width, m.height)

    finish = time.time()
    print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    class_names = load_class_names(namesfile)
    plot_boxes(img, boxes, 'predictions.jpg', class_names)

def detect_cv2(cfgfile, weightfile, imgfile):
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
            print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    class_names = load_class_names(namesfile)
    img= plot_boxes_cv2(img, boxes, imgfile, class_names=class_names)

    width = img.shape[1]
    height = img.shape[0]
    pomer = 540 / height
    img_scaled = cv2.resize(img, None, fx=pomer, fy=pomer, interpolation=cv2.INTER_LINEAR)
    cv2.imshow('Image', img)
    cv2.waitKey(0)


def detect_skimage(cfgfile, weightfile, imgfile):
    from skimage import io
    from skimage.transform import resize
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'
    
    use_cuda = True
    if use_cuda:
        m.cuda()

    img = io.imread(imgfile)
    sized = resize(img, (m.width, m.height)) * 255
    
    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    class_names = load_class_names(namesfile)
    #plot_boxes_cv2(img, boxes, savename='Predistions/predistion.jpg', class_names=class_names)

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
    parser.add_argument('--config', '-c', type=str, default='cfg/toplogo-tiny.cfg', help='network configuration file')
    parser.add_argument('--weights', '-w', type=str, default='weights/test72.weights', help='initial weights file')

    FLAGS, _ = parser.parse_known_args()
    main()

