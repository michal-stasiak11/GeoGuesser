def live_predict(model_path, setting, wait_key, classNames, video_path=None):
    """
    Perform live object detection using YOLO model.

    Parameters:
    - model_path (str): Path to the YOLO model weights file.
    - setting (str): Mode of operation, either 'live' for webcam or 'static' for video file.
    - wait_key (int): Time in milliseconds to wait between frames. A value of 0 means wait indefinitely.
    - classNames (list of str): List of class names that the model has been trained to recognize.
    - video_path (str, optional): Path to the video file for 'static' setting. Required if setting is 'static'.

    Raises:
    - ValueError: If 'setting' is not 'live' or 'static', or if 'video_path' is not provided for 'static' setting.
    """

    # Initialize video capture based on the setting
    if setting == 'live':
        # For live webcam feed
        cap = cv2.VideoCapture(0)  # Open default webcam
        cap.set(3, 640)  # Set the width of the frame to 640 pixels
        cap.set(4, 480)  # Set the height of the frame to 480 pixels
    elif setting == 'static':
        # For video file
        if video_path is None:
            raise ValueError("In 'static' setting you must pass video_path")
        cap = cv2.VideoCapture(video_path)  # Load video file
    else:
        # Raise an error if setting is invalid
        raise ValueError(f"Invalid setting '{setting}'. Expected 'live' or 'static'.")

    # Load the YOLO model from the specified path
    model = YOLO(model_path)

    # Define specific colors for selected classes
    classColors = {
        "different traffic sign": (255, 100, 50),  # Blue
        "pedestrian": (128, 0, 128),  # Purple
        "car": (0, 255, 0),  # Green
        "truck": (255, 165, 0),  # Orange
        "warning sign": (0, 255, 255),  # Yellow
        "prohibition sign": (0, 0, 255),  # Red
        "pedestrian crossing": (173, 216, 230),  # Light Blue
        "speed limit sign": (255, 192, 203)  # Pink
    }

    # Define colors for remaining classes
    remaining_colors = {
        "dark green": (0, 100, 0),  # Dark Green
        "dark yellow": (255, 255, 0)  # Dark Yellow
    }

    road_sign_classes = {
        "different traffic sign",
        "prohibition sign",
        "warning sign",
        "speed limit sign"
    }

    output_dir = "cropped_signs"
    os.makedirs(output_dir, exist_ok=True)
    crop_count = 0

    # Assign colors to the remaining classes
    remaining_color_list = list(remaining_colors.values())
    for i, cls in enumerate(classNames):
        if cls not in classColors:
            classColors[cls] = remaining_color_list[i % len(remaining_color_list)]

    while True:
        # Read a frame from the video capture
        success, img = cap.read()
        if not success:
            break  # End of video or cannot read frame

        # Perform object detection on the current frame
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Extract bounding box coordinates and convert to integers
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Get the color for the bounding box based on the detected class
                cls = classNames[int(box.cls[0])]
                color = classColors.get(cls, (255, 255, 255))  # Default to white if class not found in color map

                # Draw a thin rectangle around the detected object
                # cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)  # Thickness set to 2 for thin rectangles

                # Calculate the confidence score and format it
                conf = math.floor(box.conf[0] * 100) / 100

                # Display class name and confidence score
                # cvzone.putTextRect(img, f"{cls}", (max(0, x1), max(35, y1)), scale=1.5, thickness=2, offset=3, colorR=color, colorT=(0, 0, 0))

                                # Only crop and save if class is a road sign
                if conf > 0.8 and cls in road_sign_classes:
                    h, w, _ = img.shape
                    padding_x = int((x2 - x1) * 0.15)
                    padding_y = int((y2 - y1) * 0.15)

                    x1_pad = max(x1 - padding_x, 0)
                    y1_pad = max(y1 - padding_y, 0)
                    x2_pad = min(x2 + padding_x, w)
                    y2_pad = min(y2 + padding_y, h)

                    crop = img[y1_pad:y2_pad, x1_pad:x2_pad]

                    
                    region = analyze_sign_origin(crop)
                    crop_filename = f"{output_dir}/sign_{region}_{cls}_{crop_count}.jpg"
                    cv2.imwrite(crop_filename, crop)

                    # Now analyze the sign
                   

                    crop_count += 1


        # Display the resulting frame in a window
        cv2.imshow("Image", img)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(wait_key) & 0xFF == ord('q'):
            break

    # Release the video capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    # Define class names for different settings

    class_names_finetuned = [
        "car", "different traffic sign", "green traffic light", "motorcycle", "pedestrian", "pedestrian crossing",
        "prohibition sign", "red traffic light", "speed limit sign", "truck", "warning sign"
    ]

    # Run the live_predict function with the fine-tuned model and specified settings
    live_predict(
        model_path='fine_tuned_yolov8s.pt',
        setting='static',
        wait_key=5,
        classNames=class_names_finetuned,
        video_path='filmy/test.mp4'
    )