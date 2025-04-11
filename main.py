import cv2
import os
from ultralytics import YOLO


model = YOLO(r"best (1).pt")


image_folder = "images"
output_folder = "outputs"


if not os.path.exists(output_folder):
    os.makedirs(output_folder)


image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.jpg', '.png', '.jpeg'))]


for image_path in image_paths:

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        continue

 
    results = model(img)
    result = results[0]

  
    boxes = result.boxes.xyxy  
    confidences = result.boxes.conf 
    class_ids = result.boxes.cls 

    for box, confidence, class_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = map(int, box)  
        label = f"{model.names[int(class_id)]} {confidence:.2f}" 


        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

        coord_text = f"({x1}, {y1})"
        full_text = f"{label} {coord_text}"
        cv2.putText(img, full_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 70, 90), 2)


    cv2.imshow(f"YOLO Inference - {os.path.basename(image_path)}", img)


    output_path = os.path.join(output_folder, f"output_{os.path.basename(image_path)}")
    output_path = os.path.splitext(output_path)[0] + ".jpg"
    cv2.imwrite(output_path, img)


    cv2.waitKey(0)


cv2.destroyAllWindows()
