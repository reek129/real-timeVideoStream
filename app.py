from flask import Flask, render_template, Response, jsonify
import cv2
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np

from ultralytics import YOLO

app = Flask(__name__)

# Load the PyTorch model
model = YOLO('models/saved_yolo_glare_model3.pt')
# model.eval()  # Set model to evaluation mode
# last_detected_crop = None

# Load the image classification model
classification_model = torch.load('models/resnet18_model.pt')  # Replace with your classification model path
classification_model.eval()  # Set the classification model to evaluation mode

# Class mapping dictionary
class_mapping = {
    0: "addedLane",
    1: "bicycleCrossing",
    2: "curveLeft",
    3: "curveLeftOnly",
    4: "curveRightOnly",
    5: "doNotBlock",
    6: "doNotEnter",
    7: "doNotStop",
    8: "endRoadwork",
    9: "exitSpeedAdvisory25",
    10: "exitSpeedAdvisory30",
    11: "exitSpeedAdvisory45",
    12: "keepLeft",
    13: "keepRight",
    14: "laneEnds",
    15: "merge",
    16: "noLeftTurn",
    17: "noLeftOrUTurn",
    18: "noRightTurn",
    19: "noUTurn",
    20: "oneWay",
    21: "pedestrianCrossing",
    22: "rampSpeedAdvisory25",
    23: "rampSpeedAdvisory30",
    24: "roadworkAhead",
    25: "school",
    26: "shiftLeft",
    27: "shiftRight",
    28: "signalAhead",
    29: "speedLimit25",
    30: "speedLimit30",
    31: "speedLimit35",
    32: "speedLimit40",
    33: "speedLimit45",
    34: "speedLimit55",
    35: "speedLimit55Ahead",
    36: "speedLimit65",
    37: "stop",
    38: "turnRight",
    39: "workersAhead",
    40: "yield"
}


transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# Initialize the camera
camera = cv2.VideoCapture(0)  # 0 is the default camera


# Variables to hold the last detected crop, class label, and adversarial image
last_detected_crop = np.zeros((120, 160, 3), dtype=np.uint8)
last_class_label = "No detection"
attacked_image = np.zeros((120, 160, 3), dtype=np.uint8)
attacked_class_label = "No attack result"

def adversarial_attack(image, epsilon=0.25):
    """Apply a simple adversarial perturbation to the image."""
    noise = np.random.uniform(-epsilon, epsilon, image.shape) * 255
    attacked = np.clip(image + noise, 0, 255).astype(np.uint8)
    return attacked

def classify_image(image):
    """Classify the image and return the class label."""
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    transformed_img = transform(pil_img).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = classification_model(transformed_img)
        _, predicted = outputs.max(1)
        class_number = predicted.item()  # Get the class number
        class_name = class_mapping.get(class_number, "Unknown")
    return class_name

# def generate_frames():
#     while True:
#         # Capture frame-by-frame
#         success, frame = camera.read()  # Read the camera frame
#         if not success:
#             break
#         else:
#             results = model(frame)
#             detections = results[0].boxes

#             if detections:
#                 for box in detections:
#                     x1, y1, x2, y2 = box.xyxy[0]  # Get box coordinates
#                     confidence = box.conf[0]      # Get confidence score
#                     class_id = box.cls[0]   

#                     cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

#                     last_detected_crop = frame[int(y1):int(y2), int(x1):int(x2)].copy()
#             else:
#                 pass


#             # Encode the frame in JPEG format
#             ret, buffer = cv2.imencode('.jpg', frame)
#             if not ret:
#                 continue

#             frame = buffer.tobytes()
#             # Yield the output frame in byte format
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_main_feed():
    global last_detected_crop, last_class_label, attacked_image,attacked_class_label
    while True:
        success, frame = camera.read()  # Capture frame-by-frame
        if not success:
            break
        else:
            # Perform inference
            results = model(frame)  # Model inference on the frame
            detections = results[0].boxes  # Get boxes for the first frame in batch

            # Check if there are detections
            if detections and len(detections) > 0:
                # Take the first detection
                box = detections[0]
                x1, y1, x2, y2 = [int(coord.item()) for coord in box.xyxy[0]]  # Convert coordinates to integers
                
                # Ensure coordinates are within image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                # Check if bounding box coordinates are valid
                if x1 < x2 and y1 < y2:
                    confidence = box.conf[0]  # Get confidence score
                    # class_id = int(box.cls[0])  # Get class ID

                    # Draw bounding box on the frame
                    # label = f"{model.names[class_id]} {confidence:.2f}"

                    
                    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # frame = cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    # Crop the detected region and store it as the last detected crop
                    last_detected_crop = frame[y1:y2, x1:x2].copy()
                    last_class_label = classify_image(last_detected_crop)

                    attacked_frame = adversarial_attack(frame)
                    # attacked_frame = adversarial_attack(last_detected_crop)

                    attacked_image = attacked_frame[y1:y2, x1:x2].copy()
                    attacked_class_label = classify_image(attacked_image) 

                    print(last_class_label,attacked_class_label)

                    # # Convert the crop to PIL format for classification
                    # pil_img = Image.fromarray(cv2.cvtColor(last_detected_crop, cv2.COLOR_BGR2RGB))
                    # transformed_img = transform(pil_img).unsqueeze(0)  # Add batch dimension

                    # # Run the classification model
                    # with torch.no_grad():
                    #     outputs = classification_model(transformed_img)
                    #     _, predicted = outputs.max(1)
                    #     last_class_label = f"Class: {predicted.item()}"  # Update the class label with prediction

                    
            


            
            # Encode the main frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame = buffer.tobytes()

            # Yield the main frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



def generate_cropped_feed():
    global last_detected_crop
    while True:
        # Use the last detected crop or a blank image if none is available
        if last_detected_crop is None:
            # Create a blank image if no detections
            blank_image = np.zeros((120, 160, 3), dtype=np.uint8)
            ret, buffer = cv2.imencode('.jpg', blank_image)
        else:
            # Use the last detected crop
            ret, buffer = cv2.imencode('.jpg', last_detected_crop)

        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def generate_attacked_feed():
    global attacked_image
    while True:
        # Serve the attacked image
        ret, buffer = cv2.imencode('.jpg', attacked_image)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/')
def index():
    global last_class_label, attacked_class_label
    print("Rendering index.html")
    return render_template('index.html', 
        class_label=last_class_label, 
        attacked_class_label=attacked_class_label
        )  # Render an HTML template

@app.route('/video_feed')
def video_feed():
    print("print from video feed")
    print(last_class_label,attacked_class_label)

    # Video streaming route
    return Response(generate_main_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/cropped_feed')
def cropped_feed():
    # Video streaming route for cropped feed
    return Response(generate_cropped_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/attacked_feed')
def attacked_feed():
    # Video streaming route for attacked feed
    return Response(generate_attacked_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_labels')
def get_labels():
    global last_class_label, attacked_class_label
    # Return the current class labels as JSON
    return jsonify({
        'class_label': last_class_label,
        'attacked_class_label': attacked_class_label
    })


if __name__ == "__main__":
    app.run(debug=True)
