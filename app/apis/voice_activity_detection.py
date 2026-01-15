

from fastapi import APIRouter, WebSocket
from app.services.audio_analysis_service  import analyze_audio_chunk

router = APIRouter(prefix="/audio-analysis", tags=["Audio Proctoring"])

@router.websocket("/ws/audio")
async def audio_ws(websocket: WebSocket):    
    await websocket.accept()
    print("WebSocket connected")

    buffer = bytearray()
    try:
        while True:
            chunk = await websocket.receive_bytes()
            buffer.extend(chunk)

            if len(buffer) >= 160000:  
                result = await analyze_audio_chunk(buffer)
                await websocket.send_json(result)
                buffer.clear()

    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()
