from ultralytics import YOLO
import shutil
from roboflow import Roboflow

rf = Roboflow(api_key="XLCwBcZbaxnQjKvVoY2a")
project = rf.workspace("viren-dhanwani").project("tennis-ball-detection")
version = project.version(6)
dataset = version.download("yolov5")

shutil.move("tennis-ball-detection-6/train",
"tennis-ball-detection-6/tennis-ball-detection-6/train",
)
shutil.move("tennis-ball-detection-6/test",
"tennis-ball-detection-6/tennis-ball-detection-6/test",
)
shutil.move("tennis-ball-detection-6/valid",
"tennis-ball-detection-6/tennis-ball-detection-6/valid",
)
# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(data=f'{dataset.location}/data.yaml', epochs=50, imgsz=640, device='mps')

