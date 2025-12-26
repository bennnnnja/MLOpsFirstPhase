import fastapi
from pydantic import BaseModel
import joblib

app = fastapi.FastAPI()    

vectorizer = joblib.load("model_vec.joblib")
model = joblib.load("model_mnb.joblib")

TARGET_NAMES = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
                'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 
                'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 
                'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc',
                'talk.religion.misc']

class TextRequest(BaseModel):
    text: str

@app.post('/text')
async def get_text(request: TextRequest):
    vectorizerd_text = vectorizer.transform([request.text])
    prediction = model.predict(vectorizerd_text)
    return {"text": request.text, "prediction": TARGET_NAMES[prediction[0]]}