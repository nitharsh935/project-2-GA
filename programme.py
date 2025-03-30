from fastapi import FastAPI, File, Form, UploadFile, HTTPException
import zipfile
import pandas as pd
import io
import openai
import os
import ssl
import uvicorn

# Ensure SSL module is available
if not hasattr(ssl, 'create_default_context'):
    raise ImportError("SSL module is missing. Please check your Python installation.")

app = FastAPI()

# OpenAI API Key (ensure to set this as an environment variable)
OPENAI_API_KEY = "p76sZWRPQIKOwRzF8jzW8nIKroZrzee3pmp7Nt6rEdodHT4wA5EQferKCK9jW8acQ6RprBhC9nAcjwEY3yKJgpKWLHL15K8FR7llD7i1HVqlJpT6sovskutzoPt9GNOH"
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key is missing. Set it as an environment variable.")

def query_gpt(question: str) -> str:
    """Queries OpenAI GPT for text-based answers."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": question}],
            api_key=OPENAI_API_KEY
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying GPT: {str(e)}")

@app.post("/api/")
async def process_request(question: str = Form(...), file: UploadFile = File(None)):
    if file:
        # Ensure file is a zip
        if not file.filename.endswith(".zip"):
            raise HTTPException(status_code=400, detail="Uploaded file must be a ZIP archive")
        
        try:
            # Read ZIP file
            with zipfile.ZipFile(io.BytesIO(await file.read()), 'r') as zip_ref:
                file_list = zip_ref.namelist()
                
                # Find CSV file inside ZIP
                csv_file = next((f for f in file_list if f.endswith(".csv")), None)
                if not csv_file:
                    raise HTTPException(status_code=400, detail="No CSV file found in the ZIP")
                
                with zip_ref.open(csv_file) as csv_ref:
                    df = pd.read_csv(csv_ref, encoding='utf-8')
                    
                    if "answer" not in df.columns:
                        raise HTTPException(status_code=400, detail="No 'answer' column in CSV")
                    
                    answer_value = df["answer"].iloc[0]
            
            return {"answer": str(answer_value)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    else:
        # Handle text-based question using GPT
        gpt_answer = query_gpt(question)
        return {"answer": gpt_answer}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
