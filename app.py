import gradio as gr
import numpy as np
import tensorflow as tf
from speech_utils import extract_features
from transformers import pipeline
import os
# Force Gradio to save microphone recordings to your D: drive
os.environ["GRADIO_TEMP_DIR"] = "D:/projects/EmotionSystem/temp"
if not os.path.exists("D:/projects/EmotionSystem/temp"):
    os.makedirs("D:/projects/EmotionSystem/temp")

    
# Load Speech Model
speech_model = tf.keras.models.load_model('models/speech_emotion_model.h5')
speech_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful']

# Load Text Model (Pre-trained RoBERTa)
text_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

def analyze_emotion(audio_file, text_input):
    # --- Process Audio ---
    audio_features = extract_features(audio_file)
    audio_features = audio_features.reshape(1, -1)
    speech_preds = speech_model.predict(audio_features)[0]
    
    # --- Process Text ---
    text_preds = text_classifier(text_input)[0]
    
    # --- Simple Late Fusion Logic ---
    # We will return both for comparison
    return {
        "Top Text Emotion": text_preds['label'],
        "Text Confidence": round(text_preds['score'], 2),
        "Top Speech Emotion": speech_labels[np.argmax(speech_preds)],
        "Speech Confidence": round(np.max(speech_preds), 2)
    }

# Gradio Interface
demo = gr.Interface(
    fn=analyze_emotion,
    inputs=[gr.Audio(type="filepath", label="Record Speech"), gr.Textbox(label="Enter what was said")],
    outputs="json",
    title="Multi-Modal Emotion Recognizer",
    description="Upload audio and enter the transcript to see how they compare!"
)

if __name__ == "__main__":
    demo.launch()