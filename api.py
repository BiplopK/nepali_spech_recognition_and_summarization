
import os   
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import uuid
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from uuid import UUID
from os.path import splitext
from pydantic import BaseModel
import subprocess 
from fastapi.responses import HTMLResponse
from summarizer import get_summary_from_text
from main import generateTranscriptForFile, generateTranscriptForFileUsingHF, generateTranscriptFromHuggingFaceModel
from main import load_huggingface_model
from main import load_local_model
import time

app = FastAPI()
import gc

##############################3

origins = [
    "*",
    
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

    
    
# endpoint for textinput
# for extractive text summarizer
class text(BaseModel):    
    texts: str
@app.get("/input-audio")
async def summary():
    try: 
        hf_model = load_huggingface_model("anish-shilpakar/wav2vec2-nepali")
        # l_model, l_processor =  load_local_model("./model", "./processor")
        t1 = time.time()
        # no need to save transcript in file
        # generateTranscriptForFileUsingHF('./input/anushasan.m4a',hf_model)
        transcript_op = generateTranscriptForFileUsingHF('./input/pustakalaya.m4a',hf_model)
        print(transcript_op)
        t2 = time.time()
        print(f"Time for huggingface model: {t2-t1} seconds")
        # with open(f"./transcripts/anushasan.txt",'r',encoding="utf-8") as f:
        #     article_text=f.read()
        t1 = time.time()
        summary=get_summary_from_text(transcript_op)
        t2 = time.time()
        # return summary
        return {"summary":summary,"time":round(t2-t1,4)}
    except Exception as e:
        print(e)
        return "fail"
        
# ## load the model and processor 
# @app.post("/loadmodel")  
# async def loadthemodels():
#     load_model.loadModelInitial()
#     abstractive_predict.load_model()
#     return True

      
# @app.post("/audio")
# def create_upload_file(audio: UploadFile = File(...)):
#     try:         
#         ext=audio.filename.split('.').pop()        
#         file_location = f"static/audio/{uuid.uuid1()}{audio.filename}"
#         with open(file_location, "wb+") as file_object:
#             file_object.write(audio.file.read())   
#         if ext == 'wav' or ext == 'flac':         
#             transcript=predict_from_speech(file_location)
#             os.remove(file_location)
#         else:
#             dest_path=f'static/audio/{uuid.uuid1()}coverted.flac'    
#             command = f'ffmpeg -i {file_location} {dest_path}'
#             subprocess.call(command,shell=True)  
#             transcript=predict_from_speech(dest_path)
#             os.remove(dest_path)
#             os.remove(file_location)
#         return transcript
#     except:
#         return "fail"



# # ###########for abstractive text summarizer for file ##########
# @app.get("/abstract-file")
# async def create_upload_file(text: UploadFile = File(...)):
#     try:       
#         print("hello")
#         t1 = time.time()
#         file_location = f"static/text/{uuid.uuid1()}{text.filename}"
#         with open(file_location, "wb+") as file_object:
#             file_object.write(text.file.read())
#         summary=abstractive_predict.abstractive_summarization_from_file(file_location)
#         # print(summary)
#         os.remove(file_location)
#         t2 = time.time()
#         return {"summary":summary,"time":round(t2-t1,4)}
#     except:
#         return {"summary":"couldnot handle request, Try again!", "time":0}

       
    