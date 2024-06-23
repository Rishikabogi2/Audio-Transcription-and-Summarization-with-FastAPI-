from fastapi import FastAPI, File, UploadFile, HTTPException
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

# Load Whisper model 
model = whisper.load_model("base")

# Initialize summarization pipeline from Hugging Face Transformers
summarizer = pipeline("summarization")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Save the uploaded file to a temporary directory
        with open(f"temp/{file.filename}", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Transcribe the audio file using Whisper
        result = model.transcribe(f"temp/{file.filename}")

        # Summarize the transcription using Hugging Face pipeline
        summarized_text = summarizer(result["text"], max_length=150, min_length=30, do_sample=False)[0]['summary_text']

        segments = result["segments"]

        timestamps = [
            {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            }
            for segment in segments
        ]
        

        # Clean up the temporary file
        os.remove(f"temp/{file.filename}")

        return {
            "transcription": result["text"],
            "summary": summarized_text,
            "timestamps": timestamps
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI application with uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)