from ultralytics import YOLO

model = YOLO('yolov8x')
result = model.predict("videos/short_game.mp4", save=True)
