from ultralytics import YOLO


# Load a model
# model = YOLO("yolo11n.pt")  # load an official detection model
model = YOLO("YOLO11n.pt")  # load an official detection model

# Track with the model
# results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True)

# Evaluate model performance on the validation set
print(model.names)

from ultralytics.utils.benchmarks import benchmark

# Benchmark
benchmark(model="YOLO11n.pt", data="coco8.yaml", imgsz=640, half=True)










# Perform object detection on an image
# results = model("path/to/image.jpg")
# results[0].show()

#results = model.track(source=0, show=True, tracker="bytetrack.yaml")
# results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")


# import TrackEval.trackeval as trackeval
#
# # Настройки
# eval_config = {
#     'USE_PARALLEL': False,
#     'NUM_PARALLEL_CORES': 8,
# }
# dataset_config = {
#     'GT_FOLDER': 'data/gt/mot_challenge/MOT17/train',
#     'TRACKERS_FOLDER': 'data/trackers/mot_challenge/MOT17',
#     'BENCHMARK': 'MOT17',
#     'TRACKERS_TO_EVAL': ['tracker_name'],
# }
# metrics_list = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
#
# # Создаём экземпляры
# evaluator = trackeval.Evaluator(eval_config)
# dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
#
# # Запуск оценки
# dataset_list = [dataset]
# raw_results, messages = evaluator.evaluate(dataset_list, metrics_list)
#
# print(raw_results)
