import cv2 as cv
import matplotlib.pyplot as plt

config_file = "model_files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
frozen_model = "model_files/frozen_inference_graph.pb"
model = cv.dnn.DetectionModel(frozen_model, config_file)
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean([127.50, 127.50, 127.50])
model.setInputSwapRB(True)

filename = 'model_files/labels.txt'
with open(filename, 'rt') as f:
    class_labels = f.read().rstrip('\n').split('\n')

def detect_objects_image():
    image = cv.imread('images/tennis_racket.jpg')

    class_index, confidence, bbox = model.detect(image, confThreshold=0.75)

    font_scale = 3
    font = cv.FONT_HERSHEY_PLAIN
    for class_index, confidence, boxes in zip(class_index.flatten(), confidence.flatten(), bbox):
        label = "{}: {:.2f}%".format(class_labels[class_index], confidence * 100)
        cv.rectangle(image, boxes, (255, 0, 0), 2)
        cv.putText(image,class_labels[class_index-1], (boxes[0] + 10, boxes[1] + 40), font, font_scale, (0, 255, 0), 2)

    cv.imshow('Image', image)
    k = cv.waitKey(0)
    if k == 27:  # Esc key to stop
        cv.destroyAllWindows()
