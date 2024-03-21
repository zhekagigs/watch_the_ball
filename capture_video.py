import cv2 as cv
from object_detection import model, class_labels

filepath = 'videos/short_game.mp4'
cap = cv.VideoCapture(filepath)
fps = cap.get(cv.CAP_PROP_FPS)
delay = int(100/fps )

if not cap.isOpened():
    print("Error opening file")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame")
        break


    class_index, confidence, bbox = model.detect(frame, confThreshold=0.6)

    print(class_index, confidence)

    font_scale = 3
    font = cv.FONT_HERSHEY_PLAIN
    if (len(class_index) != 0):
        for class_index, confidence, boxes in zip(class_index.flatten(), confidence.flatten(), bbox):
            if class_index <= 80:
                cv.rectangle(frame, boxes, (255, 0, 0), 2)
                cv.putText(frame, class_labels[class_index - 1], (boxes[0] + 10, boxes[1] + 40), font, font_scale, (0, 255, 0),
                       2)

    cv.imshow('frame', frame)

    if cv.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
