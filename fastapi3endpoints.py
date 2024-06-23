from fastapi import FastAPI, File, UploadFile, HTTPException, Body
import whisper
import uvicorn
import os
import shutil
from transformers import pipeline
import librosa

# Initialize FastAPI app
app = FastAPI()

# Create temp directory if it doesn't exist
os.makedirs("temp", exist_ok=True)

# Load Whisper model (ensure you have the appropriate model files)
model = whisper.load_model("base")

# Initialize summarization pipeline from Hugging Face Transformers
summarizer = pipeline("summarization")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Save the uploaded file to a temporary directory
        file_path = f"temp/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Transcribe the audio file using Whisper
        result = model.transcribe(file_path)

        # Summarize the transcription using Hugging Face pipeline
        summarized_text = summarizer(result["text"], max_length=150, min_length=30, do_sample=False)[0]['summary_text']

        # Extract timestamps
        timestamps = extract_timestamps(file_path)

        # Clean up the temporary file
        os.remove(file_path)

        return {
            "transcription": result["text"],
            "summary": summarized_text,
            "timestamps": timestamps
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize")
async def summarize_text(text: str = Body(..., embed=True)):
    try:
        # Summarize the provided text
        summarized_text = summarizer(text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        return {"summary": summarized_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/timestamps")
async def extract_timestamps_endpoint(file: UploadFile = File(...)):
    try:
        # Save the uploaded file to a temporary directory
        file_path = f"temp/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract timestamps from the audio file
        timestamps = extract_timestamps(file_path)

        # Clean up the temporary file
        os.remove(file_path)

        return {"timestamps": timestamps}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def extract_timestamps(audio_file_path):
    # Dummy implementation to extract timestamps (replace with actual logic)
    timestamps = []
    audio, sr = librosa.load(audio_file_path, sr=None)  # Load audio with librosa

    # Example: extract timestamps every 10 seconds
    duration = librosa.get_duration(y=audio, sr=sr)
    for i in range(0, int(duration), 10):
        timestamps.append(i)

    return timestamps

# Run the FastAPI application with uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)