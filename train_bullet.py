from ultralytics import YOLO
import comet_ml

experiment = comet_ml.Experiment(api_key='b49Yzx0tym59w3GrqoasrOYSg', project_name='large_lr_1e-3')

# Load a model
model = YOLO('yolov8l-seg.yaml')  # build a new model from YAML
model = YOLO('yolov8l-seg.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8l-seg.yaml').load('yolov8l.pt')  # build from YAML and transfer weights

# Train the model
model.train(data='Bullet-Holes-6/data.yaml', epochs=100, imgsz=640, batch=4, optimizer='Adam', lr0=0.001, cos_lr=True)
experiment.end()