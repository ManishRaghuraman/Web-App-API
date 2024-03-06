from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
from starlette.responses import JSONResponse

app = FastAPI()

MODEL = tf.keras.models.load_model("/Users/manish/MAC_Courses/IP2/appledisease_2/models/1")
class_names = ['Apple_Scab', 'Cedar_Apple_Rust', 'Healthy', 'Powdery_mildew']


def read_file_as_image(data) -> np.ndarray:
    try:
        image = Image.open(BytesIO(data))
        # Resize the image to match the model's expected input size
        image = image.resize((256, 256))
        return np.array(image)
    except UnidentifiedImageError:
        # This exception is raised if the file is not an image
        return None


@app.post("/prediction")
async def prediction(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = read_file_as_image(image_data)

        if image is None:
            raise HTTPException(status_code=400, detail="The file uploaded is not a valid image.")

        img_batch = np.expand_dims(image, 0)  # Prepare image for the model
        predictions = MODEL.predict(img_batch)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        return JSONResponse(content={"class": predicted_class, "confidence": confidence})
    except Exception as e:
        return JSONResponse(status_code=500,
                            content={"message": "An error occurred during prediction.", "detail": str(e)})


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8081)
