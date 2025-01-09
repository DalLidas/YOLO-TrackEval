# import cv2 as cv
# from ultralytics import YOLO
# import numpy as np
#
#
# def apply_yoloV8onVideo():
#     cache_points = []
#     flag = True
#     writer = None
#     (W, H) = (None, None)
#
#     model = YOLO('yolo11n.pt')
#     model.classes = [0]
#     path = "MOT16Lite-02_video.avi"
#     # path = "people-walking-on.mp4"
#     # path = "traffic_video2.mp4"
#     vs = cv.VideoCapture(path)
#     results = model.track(source=path, stream=True)
#
#     for result in results:
#         print("names:-", result.verbose())
#         print("classes:--", result.boxes.cls)
#
#         i = 0
#         image = result.plot()
#         if result:  # drawing custom boxes
#             loc = result.boxes.xywh[0]
#             loc = loc.numpy()
#             cache_points.append(loc)
#             # image = draw_CustomBox(loc, image)
#             cache_points.append(loc)
#
#         # saving the video file
#         if writer is None:
#             # initialize our video writer
#             fourcc = cv.VideoWriter_fourcc(*"MJPG")
#             writer = cv.VideoWriter("tracked_" + path + ".mp4", fourcc, 30,
#                                     (image.shape[1], image.shape[0]), True)
#         print("writing")
#         writer.write(image)
#
#     # release the file pointers
#     print("[INFO] cleaning up...")
#     writer.release()
#     vs.release()
#
#
# def draw_CustomBox(loc, image):
#     center_coordinates = (int(loc[0]), int(loc[1]))
#     radius = 80
#     color = (0, 255, 0)
#     thickness = -1
#     sample = cv.circle(image, center_coordinates, radius, color, thickness)
#     return sample
#
#
# def custom_line(cache_points, image):
#     if len(cache_points) >= 2:
#
#         p1 = (int(cache_points[-1][0]), int(cache_points[-1][1]))
#         p2 = (int(cache_points[-2][0]), int(cache_points[-2][1]))
#         p1 = (33, 33)
#         p2 = (900, 400)
#         image = cv.line(image, p1, p2, (0, 255, 0), thickness=10)
#     return image
#
#
# if __name__ == "__main__":
#     apply_yoloV8onVideo()
