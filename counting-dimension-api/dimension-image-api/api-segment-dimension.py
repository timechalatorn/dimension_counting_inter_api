from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import torch
from fastsam import FastSAM, FastSAMPrompt
import base64

app = FastAPI()

# Initialize the model
model = FastSAM('FastSAM-s.pt')

# Determine the device to use (CUDA, MPS, or CPU)
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

intermediate_image_base64 = None  # Global variable to store intermediate image
everything_results = None  # Global variable to store results

@app.post("/dimension/upload_image/")
async def upload_image(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.6)
):
    global everything_results, intermediate_image_base64
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Get the segmentation results
    everything_results = model(
        source=frame,
        device=DEVICE,
        retina_masks=True,
        imgsz=600,
        conf=confidence_threshold,
        iou=0.2,
    )

    # Draw bounding boxes on the image and measure the dimensions
    for i, box in enumerate(everything_results[0].boxes):
        box = box.xyxy.cpu().numpy()[0]
        
        # Create a mask for the object
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), 255, -1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the minimum area rectangle
        if len(contours) > 0:
            rect = cv2.minAreaRect(contours[0])
            box_points = cv2.boxPoints(rect)
            box_points = np.intp(box_points)
            cv2.drawContours(frame, [box_points], 0, (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {i}', (int(box[0]), int(box[1]) - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # Encode the image as base64
    retval, buffer = cv2.imencode('.jpg', frame)
    image_jpg = buffer.tobytes()
    intermediate_image_base64 = base64.b64encode(image_jpg).decode('utf-8')

    # Return the base64 image in JSON response
    return JSONResponse(content={"image": intermediate_image_base64})

@app.post("/dimension/process_image/")
async def process_image(
    reference_height: float = Form(...),
    reference_width: float = Form(...),
    reference_box_id: int = Form(...)
):
    global everything_results, intermediate_image_base64
    frame = cv2.imdecode(np.frombuffer(base64.b64decode(intermediate_image_base64), np.uint8), cv2.IMREAD_COLOR)

    # Determine the conversion factors using the reference box
    reference_object_height_real_world = reference_height
    reference_object_width_real_world = reference_width
    reference_box = everything_results[0].boxes[reference_box_id].xyxy.cpu().numpy()[0]

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.rectangle(mask, (int(reference_box[0]), int(reference_box[1])), (int(reference_box[2]), int(reference_box[3])), 255, -1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(contours[0])
    width_pixels = rect[1][0]
    height_pixels = rect[1][1]

    # Calculate the conversion factors
    height_conversion_factor = reference_object_height_real_world / height_pixels  # cm per pixel
    width_conversion_factor = reference_object_width_real_world / width_pixels  # cm per pixel

    # Redraw bounding boxes with dimensions
    for i, box in enumerate(everything_results[0].boxes):
        if i == reference_box_id:
            continue  # Skip the reference box
        
        box = box.xyxy.cpu().numpy()[0]
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), 255, -1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            rect = cv2.minAreaRect(contours[0])
            box_points = cv2.boxPoints(rect)
            box_points = np.intp(box_points)
            cv2.drawContours(frame, [box_points], 0, (0, 255, 0), 2)
            width_pixels = rect[1][0]
            height_pixels = rect[1][1]
            width_real_world = width_pixels * width_conversion_factor
            height_real_world = height_pixels * height_conversion_factor
            cv2.putText(frame, f'W: {height_real_world:.2f} cm', (int(box[0]), int(box[1]) - 210), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(frame, f'H: {width_real_world:.2f} cm', (int(box[0]), int(box[1]) - 130), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # Encode the resulting image as JPEG binary data
    retval, buffer = cv2.imencode('.jpg', frame)
    image_jpg = buffer.tobytes()
    base64_image = base64.b64encode(image_jpg).decode('utf-8')

    # Return the base64 image in JSON response
    return JSONResponse(content={"image": base64_image})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
