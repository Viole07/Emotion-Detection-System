---

# Multi-Modal Emotion Recognition System (Text & Speech)

This project implements an automated system for identifying human emotions by analyzing both **Speech (Acoustic)** and **Text (Semantic)** data. It utilizes a **Late Fusion** architecture to combine the strengths of Deep Neural Networks for audio features and Transformer models for text analysis, providing a robust emotional verdict.

## 🚀 Features
* **Dual-Branch Analysis**: Simultaneous processing of audio waveforms and textual transcripts.
* **State-of-the-Art NLP**: Employs a pre-trained **RoBERTa** model for deep semantic analysis.
* **Acoustic Feature Extraction**: Utilizes **MFCCs** via Librosa to capture vocal timber and energy.
* **Real-Time UI**: Features an interactive dashboard built with **Gradio** for testing via microphone or file upload.
* **Performance**: Achieved ~71% validation accuracy on the speech branch using the **RAVDESS** dataset.

## 🛠️ Tech Stack
* **Language**: Python 3.13
* **Speech Processing**: Librosa, Soxr
* **Deep Learning**: TensorFlow, Keras, PyTorch
* **NLP**: Hugging Face Transformers (RoBERTa)
* **Frontend**: Gradio

## 📂 Project Structure
```text
EmotionSystem/
├── data/               # RAVDESS Dataset folders (Actor_01 - Actor_24)
├── models/             # Trained Speech Model (.h5)
├── hf_cache/           # Hugging Face model cache
├── speech_utils.py     # MFCC extraction logic
├── train_speech.py     # Speech model training script
├── app.py              # Gradio Multi-Modal interface
└── requirements.txt    # Project dependencies
```

## ⚙️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Viole07/Emotion-Detection-System.git
   cd Emotion-Detection-System
   ```

2. **Set up Virtual Environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Dataset**:
   Download the **RAVDESS Emotional Speech** dataset and place the `Actor_XX` folders inside the `data/` directory.

## 🖥️ Usage

### 1. Train the Speech Model
Analyze the audio files and generate the acoustic "brain" for the system:
```bash
python train_speech.py
```

### 2. Run the Multi-Modal App
Launch the web interface to test text and speech integration:
```bash
python app.py
```
Access the interface at `[http://127.0.0.1:7860](http://127.0.0.1:7860).

## 📊 Methodology
The system employs **Late Fusion**, combining a Speech Branch (MLP trained on MFCCs) and a Text Branch (DistilRoBERTa Transformer). The final emotion classification is reached by cross-referencing confidence scores from both branches.


```
