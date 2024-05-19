import PIL.Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import io
import numpy as np
import tensorflow as tf

app = FastAPI()

model_path = "../Model/potato-model"
new_model = tf.keras.models.load_model(model_path)

categories = ["Potato___Early_blight", "Potato___healthy", "Potato___Late_blight"]


@app.get("/")
def Home():
    return {"message": "Hello World I am Daram Nikhil, Aspiring Data Scientist"}


def image_read(data_file):
    image = PIL.Image.open(io.BytesIO(data_file))
    image = np.array(image)
    return image


@app.post("/predict")
async def Prediction(file: UploadFile = File(...)):
    image = image_read(await file.read())
    resized_image = tf.image.resize(image, [256, 256])
    resized_image = resized_image / 255.0
    pred = new_model.predict(np.expand_dims(resized_image, axis=0))
    final_pred = np.argmax(pred)
    cat_index = categories[final_pred]
    return cat_index
