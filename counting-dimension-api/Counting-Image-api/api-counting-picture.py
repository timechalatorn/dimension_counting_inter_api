from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import cv2
from ultralytics import YOLO
import numpy as np
import uvicorn
import base64

app = FastAPI()

# Load the model
model = YOLO("yolov8x.pt")

# In-memory storage for detection results
detection_storage = {}

@app.post("/detection/detect/")
async def detect(file: UploadFile = File(...)):
    # Read the image file
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    im0 = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    assert im0 is not None, "Error reading image file"

    # Perform initial detection
    results = model(im0)

    # Collect all detected classes
    detected_classes = set()
    for result in results:
        for box in result.boxes:
            class_index = int(box.cls.item())
            detected_classes.add(model.names[class_index])

    # Store results in memory
    detection_id = len(detection_storage)
    detection_storage[detection_id] = (im0, results, detected_classes)

    # Return the list of detected classes
    return JSONResponse(content={"detection_id": detection_id, "detected_classes": list(detected_classes)})

@app.post("/detection/annotate/")
async def annotate(detection_id: int = Form(...), classes: str = Form(...)):
    if detection_id not in detection_storage:
        raise HTTPException(status_code=404, detail="Detection ID not found")

    im0, results, detected_classes = detection_storage[detection_id]

    # Process user input
    desired_classes = [cls.strip() for cls in classes.split(",")]
    detect_all_classes = 'all' in [cls.lower() for cls in desired_classes]

    # Initialize variables for counting and annotating
    filtered_tracks = []
    class_counts = {cls: 0 for cls in detected_classes}  # Initialize with detected classes

    # Filter detections based on user input
    if detect_all_classes:
        for result in results:
            for box in result.boxes:
                class_index = int(box.cls.item())
                class_name = model.names[class_index]
                filtered_tracks.append({
                    'bbox': box.xyxy,
                    'class': class_name,
                    'confidence': box.conf
                })
                class_counts[class_name] += 1
    else:
        for result in results:
            for box in result.boxes:
                class_index = int(box.cls.item())
                class_name = model.names[class_index]
                if class_name in desired_classes:
                    filtered_tracks.append({
                        'bbox': box.xyxy,
                        'class': class_name,
                        'confidence': box.conf
                    })
                    class_counts[class_name] += 1

    # Annotate the image with the class name and count
    for track in filtered_tracks:
        bbox = track['bbox'][0].cpu().numpy()
        cv2.rectangle(im0, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        cv2.putText(im0, track['class'], (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the count of each class on the image
    y_offset = 30
    for cls, count in class_counts.items():
        if count > 0:
            cv2.putText(im0, f"Number of {cls}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            y_offset += 30

    # Save the output image
    output_image_path = f"object_counting_output_{detection_id}.jpg"
    
    
    # Encode the resulting image as JPEG binary data
    retval, buffer = cv2.imencode('.jpg', im0)
    image_jpg = buffer.tobytes()
    base64_image = base64.b64encode(image_jpg).decode('utf-8')


    # Return the base64 image in JSON response
    return JSONResponse(content={"image": base64_image, "class_counts": class_counts})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
