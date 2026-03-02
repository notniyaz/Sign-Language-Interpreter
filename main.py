from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from model_utils import build_lstm_model

app = FastAPI()

# Allow frontend to connect
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# CONFIGURATION
ACTIONS = np.array(['Hello', 'Thank You', 'Namaste']) # Add your Kaggle classes here
model = build_lstm_model(len(ACTIONS))
# model.load_weights('isl_weights.h5') # Uncomment once you have trained weights

sequence = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    global sequence
    try:
        while True:
            data = await websocket.receive_json()
            landmarks = data['landmarks']
            
            sequence.append(landmarks)
            sequence = sequence[-30:] # Keep the last 30 frames
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                prediction = ACTIONS[np.argmax(res)]
                confidence = float(res[np.argmax(res)])
                
                if confidence > 0.7: # Only send if model is sure
                    await websocket.send_json({"prediction": prediction, "confidence": confidence})
    except Exception as e:
        print(f"Connection closed: {e}")