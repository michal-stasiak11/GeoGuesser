import cv2
from inference_sdk import InferenceHTTPClient
from ultralytics import YOLO
import os
import csv
import numpy as np
import sys

# ===========================================================
# PERSON DETECTION
# ===========================================================


def detect_and_crop_person(image: np.ndarray, conf_threshold: float = 0.1):
    print("Wykrywanie osób...")
    yolo_model = YOLO('fine_tuned_yolov8s.pt')
    cropped_people = []
    results = yolo_model(image, verbose=False)

    for r in results:
        for box in r.boxes:
            if box.conf[0] > conf_threshold:
                class_id = int(box.cls[0])
                class_name = yolo_model.names[class_id]
                if class_name == 'pedestrian':
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    h, w, _ = image.shape
                    padding_percent = 0.001
                    padding_x = int((x2 - x1) * padding_percent)
                    padding_y = int((y2 - y1) * padding_percent)

                    x1_pad = max(x1 - padding_x, 0)
                    y1_pad = max(y1 - padding_y, 0)
                    x2_pad = min(x2 + padding_x, w)
                    y2_pad = min(y2 + padding_y, h)
                    crop = image[y1_pad:y2_pad, x1_pad:x2_pad]
                    cropped_people.append(crop)
            else:
                print(f"Osoba wykryta, ale pewność ({box.conf[0]:.2f}) jest poniżej progu ({conf_threshold}).")

    return cropped_people


# ===========================================================
# FACE DETECTION
# ===========================================================

def detect_faces(image, confidence_threshold=0.4):
    print("Wykrywanie jednej twarzy...")
    (h, w) = image.shape[:2]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    prototxt_path = os.path.join(script_dir, "deploy.prototxt.txt")
    model_path = os.path.join(script_dir, "res10_300x300_ssd_iter_140000.caffemodel")

    try:
        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    except cv2.error as e:
        print("BŁĄD KRYTYCZNY: Nie można wczytać plików modelu.")
        sys.exit(1)

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    if detections.shape[2] > 0:
        confidence = detections[0, 0, 0, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            margin_percent = 0.01 
            if margin_percent > 0:
                face_width = endX - startX
                face_height = endY - startY          
                margin_w = int(face_width * margin_percent)
                margin_h = int(face_height * margin_percent)
                startX = startX - (margin_w // 2)
                endX = endX + (margin_w // 2)
                startY = startY - (margin_h // 2)
                endY = endY + (margin_h // 2)
        
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w, endX), min(h, endY)
            return image[startY:endY, startX:endX]

    print("Nie wykryto twarzy z wystarczającą pewnością.")
    return None


# ===========================================================
# RACE PREDICTION
# ===========================================================

# --- KONFIGURACJA ---
API_KEY = "uqCeGhsQZJDyzIvgQ06V"
MODEL_ID = "human-race-detection/7"
CSV_FILE = "race_country_probabilities.csv"


try:
    CLIENT = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=API_KEY)
except Exception as e:
    print(f"Błąd krytyczny podczas inicjalizacji klienta: {e}")
    exit()

def load_country_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Plik {filepath} nie został znaleziony. Upewnij się, że jest w tym samym folderze co skrypt.")

    country_profiles = {}
    print(f"Wczytuję dane z pliku: {filepath}")
    with open(filepath, mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile, delimiter=';')
        headers = next(reader)[1:] 

        for row in reader:
            if not row: continue
            country_name = row[0].split(" (")[0]
            profiles = {headers[i]: float(row[i+1]) / 100.0 for i in range(len(headers))}
            country_profiles[country_name] = profiles
            
    print("Dane o krajach wczytane pomyślnie.")
    return country_profiles, headers

try:
    COUNTRY_PROFILES, RACE_HEADERS = load_country_data(CSV_FILE)
except FileNotFoundError as e:
    print(f"BŁĄD: {e}")
    exit()


def predict_race(image_object):

    
    print("Przewidywanie rasy...")
    try:
        result = CLIENT.infer(image_object, model_id=MODEL_ID)
    except Exception as e:
        print(f"Błąd podczas komunikacji z API Roboflow: {e}")
        return None
    detected_races = {race: 0.0 for race in RACE_HEADERS}
    for prediction in result.get('predictions', []):
        detected_class = prediction['class']
        confidence = prediction['confidence']
        
        if detected_class in detected_races:
            detected_races[detected_class] = max(detected_races[detected_class], confidence)
        else:
            print(f"[Ostrzeżenie] Model zwrócił nieznaną klasę: '{detected_class}'. Zostanie zignorowana.")   
    country_scores = {}
    for country, profile in COUNTRY_PROFILES.items():
        score = sum(detected_races[race] * probability_in_country * confidence for race, probability_in_country in profile.items())
        country_scores[country] = score

    total_score = sum(country_scores.values())
    if total_score == 0:
        return {country: 0.0 for country in COUNTRY_PROFILES.keys()}
    final_probabilities = {country: (score *confidence*confidence) / total_score for country, score in country_scores.items()}
    print(f"\nZnaleziony: {detected_class} z pewnością {confidence:.2%}")

    return final_probabilities



# ===========================================================
# COMBINE PROBABILITIES
# ===========================================================

def combine_probabilities(list_of_probabilities: list[dict]) -> dict:
    print("Łączenie prawdopodobieństw...")
    if not list_of_probabilities:
        return {}

    if len(list_of_probabilities) == 1:
        return list_of_probabilities[0]

    combined_scores = {country: 0.0 for country in list_of_probabilities[0].keys()}
    
    for prob_dict in list_of_probabilities:
        for country, probability in prob_dict.items():
            if country in combined_scores:
                combined_scores[country] += probability

    total_score = sum(combined_scores.values())
    if total_score == 0:
        return combined_scores

    final_probabilities = {country: score / total_score for country, score in combined_scores.items()}
    
    return final_probabilities


# ===========================================================
# MAIN FUNCTION 
# ===========================================================

def race_prediction(image):
    person_images = detect_and_crop_person(image)
    probabilities = []
    if person_images:
        for person_image in person_images:
            face_image = detect_faces(person_image)
            if face_image is None:
                print("Nie wykryto twarzy w obrazie.")
                continue
            probabilities.append(predict_race(face_image))
        return combine_probabilities(probabilities)
    else:
        print("Nie wykryto żadnej osoby na obrazie.")
        return None




# if __name__ == "__main__":
#     probabilities = []
#     image = cv2.imread("image.jpg")
    
#     probabilities = race_prediction(image)
#     if probabilities is not None:
#         sorted_results = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
#         print("--- Posortowane wyniki ---")
#         if not sorted_results:
#             print("Brak wyników do wyświetlenia.")
#         else:
#             for kraj, prawdopodobienstwo in sorted_results:
#                 print(f"{kraj}: {prawdopodobienstwo:.2%}")