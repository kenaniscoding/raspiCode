import torch
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
import time
from picamera2 import Picamera2, Preview

# Define classification mapping
class_labels_ripeness = ['green', 'yellow_green', 'yellow']
class_labels_bruises = ['bruised', 'unbruised']
ripeness_scores = {'yellow': 3, 'yellow_green': 2, 'green': 1}
bruiseness_scores = {'bruised': 1, 'unbruised': 2}

# Load the models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ripeness model
model_ripeness = EfficientNet.from_pretrained('efficientnet-b0', num_classes=len(class_labels_ripeness))
model_ripeness.load_state_dict(torch.load("ripeness.pth", map_location=device))
model_ripeness.eval()
model_ripeness.to(device)

# Bruises model
model_bruises = EfficientNet.from_pretrained('efficientnet-b0', num_classes=len(class_labels_bruises))
model_bruises.load_state_dict(torch.load("bruises.pth", map_location=device))
model_bruises.eval()
model_bruises.to(device)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def classify_image(image, model, class_labels):
    """Classifies a given image and returns the predicted class."""
    image = transform(image).unsqueeze(0).to(device)
    output = model(image)
    _, predicted = torch.max(output, 1)
    return class_labels[predicted.item()]

def capture_image():
    """Captures an image using the Raspberry Pi Camera with proper cleanup."""
    picam2 = Picamera2()
    camera_config = picam2.create_still_configuration(main={"size": (1920, 1080)})
    picam2.configure(camera_config)
    picam2.start()
    time.sleep(3)  # Allow the camera to warm up
    image = picam2.capture_array()
    picam2.stop()
    picam2.close()  # Ensure camera resources are released
    return Image.fromarray(image)

# Capture and classify top part
print("Capturing top part of the mango...")
top_image = capture_image()
if top_image:
    top_class_ripeness = classify_image(top_image, model_ripeness, class_labels_ripeness)
    top_class_bruises = classify_image(top_image, model_bruises, class_labels_bruises)
    print(f"Top Ripeness Classification: {top_class_ripeness}")
    print(f"Top Bruises Classification: {top_class_bruises}")

# Wait for 10 seconds
time.sleep(10)

# Capture and classify bottom part
print("Capturing bottom part of the mango...")
bottom_image = capture_image()
if bottom_image:
    bottom_class_ripeness = classify_image(bottom_image, model_ripeness, class_labels_ripeness)
    bottom_class_bruises = classify_image(bottom_image, model_bruises, class_labels_bruises)
    print(f"Bottom Ripeness Classification: {bottom_class_ripeness}")
    print(f"Bottom Bruises Classification: {bottom_class_bruises}")

# Compute final mango score
if top_image and bottom_image:
    ripeness_score = (ripeness_scores.get(top_class_ripeness, 0) + ripeness_scores.get(bottom_class_ripeness, 0)) / 2
    bruiseness_score = (bruiseness_scores.get(top_class_bruises, 0) + bruiseness_scores.get(bottom_class_bruises, 0)) / 2
    print(f"Final Ripeness Score: {ripeness_score:.1f}")
    print(f"Final Bruiseness Score: {bruiseness_score:.1f}")
