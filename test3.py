import os
import cv2 as cv
from ultralytics import YOLO
import TrackEval.trackeval as trackeval


def read_sequences_from_file(filepath):
    """
    Читает список последовательностей из файла.
    """
    sequences = []
    try:
        with open(filepath, 'r') as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith("name"):
                    sequences.append(line)
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
    return sequences


def create_video_from_images(mot16_path, seq, output_video_path):
    """
    Создает видео из изображений последовательности MOT16.
    """
    seq_path = os.path.join(mot16_path, seq, "img1")  # Путь к папке с изображениями
    img_files = sorted(os.listdir(seq_path))

    # Проверка, что изображения существуют
    if not img_files:
        print(f"Error: No images found in {seq_path}")
        return

    first_img_path = os.path.join(seq_path, img_files[0])
    first_img = cv.imread(first_img_path)
    height, width, _ = first_img.shape

    # Инициализация видео записи
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    video_writer = cv.VideoWriter(output_video_path, fourcc, 30.0, (width, height))

    # Процесс добавления изображений в видео
    for img_name in img_files:
        if img_name.endswith(".jpg"):
            img_path = os.path.join(seq_path, img_name)
            img = cv.imread(img_path)
            if img is None:
                print(f"Error reading image: {img_path}")
                continue
            video_writer.write(img)

    video_writer.release()  # Завершаем запись видео
    print(f"Video saved to {output_video_path}")


def create_video_with_ground_truth(mot16_path, seq, output_video_path):
    """
    Создает видео из изображений последовательности MOT16 с отображением Ground-Truth боксов.
    """
    seq_path = os.path.join(mot16_path, seq, "img1")  # Путь к папке с изображениями
    gt_file_path = os.path.join(mot16_path, seq, "gt", "gt.txt")  # Файл Ground-Truth
    img_files = sorted(os.listdir(seq_path))

    # Проверка, что изображения существуют
    if not img_files:
        print(f"Error: No images found in {seq_path}")
        return

    first_img_path = os.path.join(seq_path, img_files[0])
    first_img = cv.imread(first_img_path)
    height, width, _ = first_img.shape

    # Инициализация видео записи
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    video_writer = cv.VideoWriter(output_video_path, fourcc, 30.0, (width, height))

    # Загрузка Ground-Truth данных
    gt_data = {}
    if os.path.exists(gt_file_path):
        with open(gt_file_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                frame_id, obj_id, x, y, w, h, conf, label, _ = map(float, parts[:9])
                if frame_id not in gt_data:
                    gt_data[frame_id] = []
                gt_data[frame_id].append((obj_id, x, y, w, h, label))

    # Процесс добавления изображений в видео
    for img_name in img_files:
        if img_name.endswith(".jpg"):
            frame_id = int(os.path.splitext(img_name)[0])  # Получаем frame_id из имени файла
            img_path = os.path.join(seq_path, img_name)
            img = cv.imread(img_path)
            if img is None:
                print(f"Error reading image: {img_path}")
                continue

            # Рисуем боксы Ground-Truth на изображении
            if frame_id in gt_data:
                for obj_id, x, y, w, h, label in gt_data[frame_id]:
                    x1, y1 = int(x), int(y)
                    x2, y2 = int(x + w), int(y + h)
                    color = (0, 0, 255)  # Красный цвет для Ground-Truth
                    cv.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv.putText(
                        img, f"GT ID:{int(obj_id)}", (x1, y1 - 10),
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                    )

            video_writer.write(img)

    video_writer.release()  # Завершаем запись видео
    print(f"Ground-Truth video saved to {output_video_path}")


def process_video_with_yolo(video_path, output_path, model_path, seq):
    """
    Обрабатывает видео с помощью YOLO и сохраняет детекции в формате TrackEval.
    """
    # Загрузка модели YOLO
    model = YOLO(model_path)
    # model.classes = [0]  # Ограничение на класс "человек" (pedestrian) # почему-то не работает

    # Открываем видео
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Получаем информацию о видео
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # Инициализируем видео писатель для сохранения выходного видео
    video_output_path = os.path.join(output_path, f"{seq}_output.avi")
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    video_writer = cv.VideoWriter(video_output_path, fourcc, 30.0, (frame_width, frame_height))

    frame_id = 0
    det_file_path = os.path.join(output_path, f"{seq}.txt")

    with open(det_file_path, "w") as det_file:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1

            # Выполняем трекинг
            results = model.track(frame, persist=True)

            # Обрабатываем результаты трекинг
            if results[0].boxes is not None and results[0].boxes.id is not None:  # Проверка на наличие объектов
                # Получение координат боксов и идентификаторов треков
                boxes = results[0].boxes.xywh.numpy()  # xywh координаты боксов
                track_ids = results[0].boxes.id.int().numpy()  # идентификаторы треков

                confidences = results[0].boxes.conf.numpy()  # коэффициент соответствия
                obj_classs = results[0].boxes.cls.numpy()  # класс объекта

                # Отрисовка треков
                for track_id, box, obj_class, confidence in zip(track_ids, boxes, obj_classs, confidences):

                    # Преобразование индекса классов. В TrackEval индекс 1 для класса "человек" (pedestrian)
                    if obj_class == 0:
                        obj_class = 1
                    else:
                        continue

                    # Индекса используемые в датасете MOT16
                    # self.class_name_to_class_id = {'pedestrian': 1, 'person_on_vehicle': 2, 'car': 3, 'bicycle': 4,
                    #                                'motorbike': 5,
                    #                                'non_mot_vehicle': 6, 'static_person': 7, 'distractor': 8,
                    #                                'occluder': 9,
                    #                                'occluder_on_ground': 10, 'occluder_full': 11, 'reflection': 12,
                    #                                'crowd': 13}

                    x_center, y_center, width, height = box  # координаты центра и размеры бокса

                    # Преобразуем в формат (x, y, w, h)
                    x_top_left = x_center - width / 2
                    y_top_left = y_center - height / 2

                    # Рисуем бокс на изображении
                    p1 = (int(x_top_left), int(y_top_left))
                    p2 = (int(x_top_left + width), int(y_top_left + height))
                    color = (0, 255, 0)  # Зеленый цвет для боксов YOLO
                    cv.rectangle(frame, p1, p2, color, 4)

                    cv.putText(
                        frame, f"ID:{track_id if track_id is not None else 'N/A'} Conf:{confidence:.2f}",
                        (int(x_top_left), int(y_top_left) - 10),
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                    )

                    # Записываем строку в формате TrackEval
                    det_file.write(
                        f"{frame_id},{track_id if track_id is not None else -1},{x_top_left:.2f},{y_top_left:.2f},"
                        f"{width:.2f},{height:.2f},{confidence:.4f},{obj_class},-1\n"
                    )

            # Записываем кадр с боксов на видео
            video_writer.write(frame)

    cap.release()
    video_writer.release()  # Завершаем запись видео
    print(f"Detections saved to {det_file_path} and output video saved to {video_output_path}")


def process_mot_with_yolo(mot16_path, output_path, model_path, sequences):
    """
    Обрабатывает изображения из MOT16 с помощью YOLO, сначала создает видео, а затем выполняет детекцию.
    """
    # Создаем папку для сохранения результатов
    os.makedirs(output_path, exist_ok=True)

    for seq in sequences:
        print(f"Processing sequence: {seq}")

        # Создаем видео из изображений
        video_output_path = os.path.join(output_path, f"{seq}_video.avi")
        create_video_from_images(mot16_path, seq, video_output_path)

        # Создаем видео с Ground-Truth
        gt_video_output_path = os.path.join(output_path, f"{seq}_ground_truth.avi")
        create_video_with_ground_truth(mot16_path, seq, gt_video_output_path)

        # Обрабатываем видео с помощью YOLO
        process_video_with_yolo(video_output_path, output_path, model_path, seq)


if __name__ == "__main__":
    # Путь к MOT16 данным
    mot16_path = os.path.join(
        os.path.dirname(__file__), 'data', 'gt', 'mot_challange', 'MOT16-train'
    )

    # Путь для сохранения результатов YOLO
    output_path = os.path.join(
        os.path.dirname(__file__), 'data', 'tracker', 'mot_challange', 'MOT16-train', 'YOLOv11n', 'data'
    )

    # Путь к модели YOLO
    model_path = "yolo11n.pt"

    # Путь к метданным датасете ("список имён видео")
    seqmaps_path = os.path.join(
        os.path.dirname(__file__), 'data', 'gt', 'mot_challange', 'seqmaps', 'MOT16-train.txt'
    )
    sequences = read_sequences_from_file(seqmaps_path)


    # Обрабатываем изображения из MOT16
    process_mot_with_yolo(mot16_path, output_path, model_path, sequences)


    # Конфигурация TrackEval
    eval_config = {
        'USE_PARALLEL': False,  # Если True, можно использовать несколько ядер
        'NUM_PARALLEL_CORES': 8,  # Количество ядер
        'PRINT_RESULTS': True,  # Печать результатов
        'PRINT_CONFIG': True,  # Печать конфигурации
    }

    # Конфигурация данных для TrackEval
    dataset_config = {
        'GT_FOLDER': os.path.join(os.path.dirname(__file__), 'data', 'gt', 'mot_challange'),  # Путь к GT
        'TRACKERS_FOLDER': os.path.join(os.path.dirname(__file__), 'data', 'tracker', 'mot_challange'),  # Путь к трекерам
        'TRACKERS_TO_EVAL': ['YOLOv11n'],  # Название трекера
        'BENCHMARK': 'MOT16',  # Название бенчмарка
        'DO_PREPROC': True,  # Обработка данных (выделение только рамок класса пешеход)
        'SPLIT_TO_EVAL': 'train',  # Оценивать train-часть
        'CLASSES_TO_EVAL': ['pedestrian'],  # Оценивать пешеходов
    }

    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]

    # Список вычисляемых критериев, метрик
    metrics_list = [
        trackeval.metrics.HOTA(),
        trackeval.metrics.CLEAR(),
        trackeval.metrics.Identity(),
    ]

    # Запускаем оценку
    results, messages = evaluator.evaluate(dataset_list, metrics_list)

    # Печать результатов
    print("Evaluation complete.")
    for metric, result in results.items():
        print(f"{metric}: {result}")
