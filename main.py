from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
import logging
import time
from processing import extract_lab_data_from_image

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI()

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    log.info(f"Request processed in {process_time:.4f} seconds")
    return response

@app.post("/get-lab-tests")
async def get_lab_tests_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
         log.warning(f"Invalid file type received: {file.content_type}")
         raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}. Please upload an image (PNG, JPG, etc.).")

    log.info(f"Received file: {file.filename}, Content-Type: {file.content_type}")

    try:
        image_bytes = await file.read()

        result = extract_lab_data_from_image(image_bytes)

        if isinstance(result, dict) and "is_success" in result:
            if not result["is_success"]:
                 log.error(f"Processing failed for {file.filename}. Error: {result.get('error', 'Unknown processing error')}")
                 return JSONResponse(
                     status_code=500,
                     content=result
                 )
            else:
                 log.info(f"Successfully processed {file.filename}. Found {len(result.get('data', []))} results.")
                 return JSONResponse(
                     status_code=200,
                     content=result
                 )
        else:
             log.error(f"Processing function returned unexpected format for {file.filename}.")
             raise HTTPException(status_code=500, detail="Internal server error: Unexpected result format from processing logic.")

    except HTTPException as http_exc:
         raise http_exc
    except Exception as e:
        log.exception(f"Unexpected error in endpoint for file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)