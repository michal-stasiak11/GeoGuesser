import random
import tkinter as tk
from tkinter import filedialog, ttk
import tensorflow as tf
import cv2
from PIL import Image, ImageTk
from license_plate_extractor import extract_license_plates_from_video
import numpy as np
import pickle


# def predict_country(image_input, model_path='license_plate_classifier.h5', config_path='model_config.pkl'):
#     model = tf.keras.models.load_model(model_path)
    
#     with open(config_path, 'rb') as f:
#         config = pickle.load(f)

#     classes = config['classes']
#     img_height = config['img_height']
#     img_width = config['img_width']
    
#     if image_input.mode != 'RGB':
#         image_input = image_input.convert('RGB')

#     image_input = image_input.resize((img_width, img_height))

#     img_array = np.array(image_input) / 255.0

#     img_array = np.expand_dims(img_array, axis=0)

#     probs = model.predict(img_array)[0]

#     return {classes[i]: float(probs[i]) for i in range(len(classes))}

def predict_country_batch(image_list, model_path='license_plate_classifier.h5', config_path='model_config.pkl'):
    model = tf.keras.models.load_model(model_path)

    with open(config_path, 'rb') as f:
        config = pickle.load(f)

    classes = config['classes']
    img_height = config['img_height']
    img_width = config['img_width']

    processed_images = []

    for image_input in image_list:
        if image_input.mode != 'RGB':
            image_input = image_input.convert('RGB')

        image_input = image_input.resize((img_width, img_height))
        img_array = np.array(image_input) / 255.0
        processed_images.append(img_array)

    batch_array = np.array(processed_images)  # shape: (N, H, W, 3)

    probs_batch = model.predict(batch_array)  # shape: (N, num_classes)
    avg_probs = np.mean(probs_batch, axis=0)  # shape: (num_classes,)

    return {classes[i]: float(avg_probs[i]) for i in range(len(classes))}


class VideoAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Analizator Wideo")
        self.root.state('zoomed')
        
        # Stałe rozmiary
        self.VIDEO_WIDTH = 800
        self.VIDEO_HEIGHT = 450
        
        # Zmienne
        self.video_path = ""
        self.cap = None
        self.is_playing = False
        self.analysis_mode = False
        self.countries = ["Polska", "Niemcy", "Francja", "Włochy", "Hiszpania", 
                         "USA", "Kanada", "Japonia", "Chiny", "Brazylia"]
        
        # Przykładowe dane dla boxów
        self.detected_data = {
            "znaki": [self.generate_sample_image() for _ in range(3)],
            "napisy": [self.generate_sample_image() for _ in range(4)],
            "rejestracje": [],
            "pojazdy": [self.generate_sample_image() for _ in range(2)],
            "osoby": [self.generate_sample_image() for _ in range(6)]
        }
        self.current_indices = {key: 0 for key in self.detected_data.keys()}
        # Przypisanie losowego kraju do każdego typu wykrycia
        self.detected_countries = {key: random.choice(self.countries) for key in self.detected_data.keys()}
        
        # GUI elements
        self.create_widgets()
        
    def generate_sample_image(self):
        img = Image.new('RGB', (150, 100), color=(random.randint(100, 200), 
                                              random.randint(100, 200), 
                                              random.randint(100, 200)))
        return ImageTk.PhotoImage(img)
        
    def create_widgets(self):
        # Główny kontener
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Górna część (wideo + ranking)
        self.top_frame = tk.Frame(self.main_frame)
        self.top_frame.pack(fill=tk.BOTH, expand=True)
        
        # Panel wideo (lewa strona)
        self.video_frame = tk.Frame(self.top_frame, bg='white')
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Kontener dla przycisków
        self.button_frame = tk.Frame(self.video_frame)
        self.button_frame.pack(fill=tk.X, pady=5)
        
        # Przyciski wideo
        self.btn_open = tk.Button(self.button_frame, text="Wczytaj wideo", command=self.open_video)
        self.btn_open.pack(side=tk.LEFT, padx=5)
        
        self.btn_play = tk.Button(self.button_frame, text="Odtwarzaj", command=self.toggle_play, state=tk.DISABLED)
        self.btn_play.pack(side=tk.LEFT, padx=5)
        
        self.btn_analyze = tk.Button(self.button_frame, text="Analizuj", command=self.toggle_analysis, state=tk.DISABLED)
        self.btn_analyze.pack(side=tk.LEFT, padx=5)
        
        # Etykieta z wideo - teraz bez stałego rozmiaru
        self.lbl_video = tk.Label(self.video_frame, bg='black')
        self.lbl_video.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Panel rankingu (prawa strona)
        self.ranking_frame = tk.Frame(self.top_frame, width=300, bg='#f0f0f0', padx=10, pady=10)
        self.ranking_frame.pack(side=tk.RIGHT, fill=tk.Y, expand=False)
        
        # Nagłówek rankingu
        self.lbl_ranking = tk.Label(self.ranking_frame, text="Ranking najbardziej prawdopodobnych krajów", 
                                   font=('Arial', 12, 'bold'), bg='#f0f0f0')
        self.lbl_ranking.pack(pady=10)
        
        # Lista rankingu
        self.ranking_list = tk.Listbox(self.ranking_frame, width=30, height=20, font=('Arial', 11))
        self.ranking_list.pack(fill=tk.BOTH, expand=True)
        
        # Pasek przewijania
        scrollbar = ttk.Scrollbar(self.ranking_list)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.ranking_list.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.ranking_list.yview)
        
        # Dolna część (boxy z wykryciami)
        self.bottom_frame = tk.Frame(self.main_frame, height=250, bg='#e0e0e0')
        self.bottom_frame.pack(fill=tk.X, side=tk.BOTTOM)

    def create_detection_boxes(self):
        box_titles = [
            "Wykryte znaki",
            "Wykryte napisy",
            "Wykryte rejestracje",
            "Wykryte pojazdy",
            "Wykryte osoby"
        ]
        
        self.detection_boxes = []
        for i, title in enumerate(box_titles):
            box_frame = tk.Frame(self.bottom_frame, width=200, height=220, 
                                bd=2, relief=tk.GROOVE, bg='white')
            box_frame.pack_propagate(False)
            box_frame.grid(row=0, column=i, padx=5, pady=5, sticky='nsew')
            
            # Tytuł boxu
            lbl_title = tk.Label(box_frame, text=title, font=('Arial', 10, 'bold'), bg='white')
            lbl_title.pack(pady=5)
            
            # Etykieta z obrazkiem
            img_label = tk.Label(box_frame, bg='white')
            img_label.pack(expand=True)
            
            # Przyciski nawigacyjne
            btn_frame = tk.Frame(box_frame, bg='white')
            btn_frame.pack(fill=tk.X)
            
            btn_prev = tk.Button(btn_frame, text="Poprzednie", width=10,
                               command=lambda k=list(self.detected_data.keys())[i]: self.prev_image(k))
            btn_prev.pack(side=tk.LEFT, padx=10)
            
            btn_next = tk.Button(btn_frame, text="Następne", width=10,
                               command=lambda k=list(self.detected_data.keys())[i]: self.next_image(k))
            btn_next.pack(side=tk.RIGHT, padx=10)
            
            # Etykieta kraju (na dole boxu)
            country_label = tk.Label(box_frame, text="Moduł wykrył kraj: ---", font=('Arial', 9), bg='white')
            country_label.pack(pady=(5, 5))

            #referencje do elementów
            self.detection_boxes.append({
                'frame': box_frame,
                'label': img_label,
                'prev_btn': btn_prev,
                'next_btn': btn_next,
                'country_label': country_label
            })
            
            self.bottom_frame.grid_columnconfigure(i, weight=1)
        
        # Inicjalizacja pierwszych obrazków
        self.update_detection_boxes()
    
    def update_detection_boxes(self):
        for i, key in enumerate(self.detected_data.keys()):
            if self.detected_data[key]:
                box = self.detection_boxes[i]
                current_idx = self.current_indices[key]
                img = self.detected_data[key][current_idx]
                box['label'].config(image=img)
                box['label'].image = img
                
                box['prev_btn'].config(state=tk.NORMAL if current_idx > 0 else tk.DISABLED)
                box['next_btn'].config(state=tk.NORMAL if current_idx < len(self.detected_data[key])-1 else tk.DISABLED)
                box['country_label'].config(text=f"Moduł wykrył kraj: {self.detected_countries[key]}")
    
    def prev_image(self, key):
        if self.current_indices[key] > 0:
            self.current_indices[key] -= 1
            self.update_detection_boxes()
    
    def next_image(self, key):
        if self.current_indices[key] < len(self.detected_data[key])-1:
            self.current_indices[key] += 1
            self.update_detection_boxes()
        
    def open_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Pliki wideo", "*.mp4 *.avi *.mov *.mkv")])
        if self.video_path:
            self.btn_analyze.config(state=tk.NORMAL)
            self.btn_play.config(state=tk.NORMAL)
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(self.video_path)
            self.show_frame()
    
    def show_frame(self):
        if self.cap and self.is_playing:
            ret, frame = self.cap.read()
            if ret:
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                
                video_width = self.video_frame.winfo_width() - 10
                video_height = self.video_frame.winfo_height() - 50  
                
                img.thumbnail((video_width, video_height))
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.lbl_video.config(image=imgtk)
                self.lbl_video.image = imgtk
            
            self.lbl_video.after(30, self.show_frame)
        else:
            if self.cap:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    
    def toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.config(text="Pauza")
            self.show_frame()
        else:
            self.btn_play.config(text="Odtwarzaj")
    
    def toggle_analysis(self):
        self.analysis_mode = not self.analysis_mode
        if self.analysis_mode:
            self.btn_analyze.config(text="Analizowenie w toku...", state=tk.DISABLED)
            # biore co 300 klate z filmiki bo mi sie nie chcialo czekac
            self.detected_data['rejestracje'], pil_images = extract_license_plates_from_video(self.video_path, "license_plate_detector.pt", "yolov8n.pt",self.cap)
            print(f"Wykryto {len(self.detected_data['rejestracje'])} tablic rejestracyjnych.")
            prediction = predict_country_batch(pil_images)
            print(prediction)
            self.create_detection_boxes()
            self.analysis_mode = not self.analysis_mode
            self.btn_analyze.config(text="Analizuj", state=tk.NORMAL)
    
    def on_closing(self):
        if self.cap:
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoAnalyzerApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()