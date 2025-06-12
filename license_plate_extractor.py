import cv2
from ultralytics import YOLO
from PIL import Image, ImageTk
import os


def extract_license_plates_from_video(video_path: str, license_plate_model_path: str, coco_model_path: str, cap=None):
    coco_model = YOLO(coco_model_path)
    license_plate_detector = YOLO(license_plate_model_path)

    results = {}

    plate_images = []
    pil_images = []
    ret = True

    frame_nmr = -1
    ret = True

    vehicles = [2, 3, 5, 7]

    frame_nmr = -1
    counter = 0
    ret = True
    plate_id = 0
    while ret:
        frame_nmr += 1
        ret, frame = cap.read()
        if counter % 300 == 0:
            if ret:
                results[frame_nmr] = {}
                detections = coco_model(frame)[0]
                detections_ = []
                for detection in detections.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = detection
                    if int(class_id) in vehicles:
                        detections_.append([x1, y1, x2, y2, score])

                license_plates = license_plate_detector(frame)[0]
                for license_plate in license_plates.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = license_plate

                    plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                    plate_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(plate_rgb)
                    pil_images.append(pil_image)
                    tk_image = ImageTk.PhotoImage(pil_image)
                    plate_images.append(tk_image)
        counter += 1
    return plate_images, pil_images
