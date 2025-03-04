import torch
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image, ImageTk
import time
import customtkinter as ctk
from picamera2 import Picamera2

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

def update_gui():
    global top_image, bottom_image
    
    top_image = capture_image()
    top_class_ripeness = classify_image(top_image, model_ripeness, class_labels_ripeness)
    top_class_bruises = classify_image(top_image, model_bruises, class_labels_bruises)
    
    time.sleep(10)
    
    bottom_image = capture_image()
    bottom_class_ripeness = classify_image(bottom_image, model_ripeness, class_labels_ripeness)
    bottom_class_bruises = classify_image(bottom_image, model_bruises, class_labels_bruises)
    
    ripeness_score = (ripeness_scores.get(top_class_ripeness, 0) + ripeness_scores.get(bottom_class_ripeness, 0)) / 2
    bruiseness_score = (bruiseness_scores.get(top_class_bruises, 0) + bruiseness_scores.get(bottom_class_bruises, 0)) / 2
    
    ripeness_val.configure(text=top_class_ripeness)
    brs_val.configure(text=top_class_bruises)
    image_lbl.configure(image=ctk.CTkImage(light_image=top_image, size=(300,400)))

# GUI Setup
window = ctk.CTk()
window.geometry("800x600")

frame_1 = ctk.CTkFrame(window, fg_color="#B3B792")
frame_1.grid(row=0, column=0, columnspan=2, rowspan=2, padx=10, pady=10, sticky="nswe")

frame_1_1 = ctk.CTkFrame(frame_1, fg_color="#000")
frame_1_1.grid(row=0, column=0, padx=10, pady=10, sticky="nswe")

image_lbl = ctk.CTkLabel(frame_1_1, text="Mango Image")
image_lbl.pack(fill="both", expand=True)

frame_1_2 = ctk.CTkFrame(frame_1, fg_color="transparent")
frame_1_2.grid(row=0, column=1, padx=10, pady=10, sticky="nswe")

details_lbl = ctk.CTkLabel(frame_1_2, text="Mango Details", text_color="white", anchor="center")
details_lbl.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="nswe")

ripeness_lbl = ctk.CTkLabel(frame_1_2, text="Ripeness:", text_color="white")
ripeness_val = ctk.CTkLabel(frame_1_2, text="----", text_color="white")
ripeness_lbl.grid(row=3, column=0, padx=5, pady=5)
ripeness_val.grid(row=3, column=1, padx=5, pady=5)

brs_lbl = ctk.CTkLabel(frame_1_2, text="Bruises:", text_color="white")
brs_val = ctk.CTkLabel(frame_1_2, text="----", text_color="white")
brs_lbl.grid(row=7, column=0, padx=5, pady=5)
brs_val.grid(row=7, column=1, padx=5, pady=5)

frame_2 = ctk.CTkFrame(window, fg_color="#B3B792")
frame_2.grid(row=0, column=2, padx=10, pady=10)

btn_frame = ctk.CTkFrame(frame_2, fg_color="#000")
btn_frame.grid(row=0, column=0, padx=10, pady=10)

start_btn = ctk.CTkButton(btn_frame, text="Start", fg_color="#8AD879", command=update_gui)
start_btn.pack()

window.mainloop()
