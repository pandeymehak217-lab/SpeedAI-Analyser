
import os
import uuid
from fastapi  import FastAPI, UploadFile, File, HTTPException
from fastapi.responses  import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors  import CORSMiddleware
from starlette.requests  import Request
from main  import run_analysis 

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


app = FastAPI(title="SpeedVision AI Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze-video")
async def analyze_video_endpoint(video_file: UploadFile = File(...)):
    """Handles video file upload, runs analysis, and returns results."""
    
    
    file_id = "job-" + str(uuid.uuid4())
    
   
    file_extension = os.path.splitext(video_file.filename)[1]
    video_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_extension}")
    
    try:
        
        with open(video_path, "wb") as buffer:
            
            while chunk := await video_file.read(1024 * 1024): 
                buffer.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

   
    try:
        
        final_data = run_analysis(video_path, OUTPUT_DIR, file_id)
        
       
        os.remove(video_path) 
        
        
        return {
            "status": "success",
            "message": "Analysis completed successfully.",
            "file_id": file_id,
            "total_objects_per_class": final_data.get('total_objects_per_class', {})
        }

    except Exception as e:
        print(f"Analysis failed for {file_id}: {e}")
        
        if os.path.exists(video_path):
            os.remove(video_path)
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {e}")


@app.get("/download/{file_type}/{file_id}")
async def download_file(file_type: str, file_id: str):
    """Serves processed files (video, csv, json) by file_id."""
    
    
    file_map = {
        "video": f"{file_id}_processed_video.mp4",
        "csv": f"{file_id}_report.csv",
        "json": f"{file_id}_results.json"
    }
    
    if file_type not in file_map:
        raise HTTPException(status_code=400, detail="Invalid file type requested.")
        
    file_name = file_map[file_type]
    file_path = os.path.join(OUTPUT_DIR, file_name)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found for ID: {file_id}")
    
    
    return FileResponse(
        path=file_path, 
        filename=file_name, 
        media_type=f"application/{'json' if file_type == 'json' else 'octet-stream'}"
    )


@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    """Serves the main HTML page for the frontend application."""
    try:
        with open("frontend/index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "Frontend files not found. Ensure 'frontend/index.html' exists."