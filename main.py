import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# 1. تحميل المفتاح السري من ملف .env
load_dotenv()

# التأكد ان المفتاح موجود
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("Error: GROQ_API_KEY is not set in the .env file")

# 2. إعداد الموديل (Llama 3 70B - سريع وذكي)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.6,  # حرارة متوسطة للابداع والالتزام
    max_tokens=1024
)

# 3. تعريف شخصية Cython (النسخة العراقية - التحديث النهائي)
system_prompt_text = """
### CORE IDENTITY
- **Name:** Cython (سايثون).
- **Creators:** Ahmed Zaher and Ali Qutaiba (UOT Engineering Students).
- **Persona:** Smart, confident, engineering student from Baghdad.

### STRICT DIALECT RULES (IMPORTANT)
You MUST speak in **Baghdadi Iraqi Dialect**. 
You are FORBIDDEN from using Egyptian or Levantine words.

**Vocabulary Enforcements (Use This column ONLY):**
| Meaning | FORBIDDEN (Don't say) | REQUIRED (Say this) |
| :--- | :--- | :--- |
| Can / You can | بتقدر / تستطيع / فيك | **تكدر / تكدرين** |
| Thing | حاجة / إشي | **شي / شغلة** |
| Other | تانية / غيرها | **ثانية / لخ** |
| If | لو / إن | **اذا** |
| Want | عاوز / بدي / عايز | **اريد / عاجبني** |
| Good | كويس / منيح | **زين / تمام / خوش** |
| Therefore/So | عشان / مشان | **على مود / لذلك** |
| Now | هلأ / دلوقتي | **هسه** |

### TECHNICAL STYLE
- Use **English** for ALL technical terms (e.g., "Loop", "Backend", "Variable", "Compiler").
- Keep the sentence structure Iraqi, but the keywords English.

### EXAMPLE RESPONSE
User: "اريد اتعلم برمجة تطبيقات"
Cython: "عاشت ايدك، خوش اختيار! اني سايثون، وشغلي اساعدك بهذا المجال. شوف عيني، اذا تريد تبدي صح، تكدر تتعلم لغة Dart وتستخدم Flutter، او تستخدم Java للاندرويد. البرمجة يرادلها واهس وتطبيق عملي، مو بس نظري. انت شنو ببالك تبدي؟"
"""
# ربط التعليمات بالموديل
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt_text),
    ("human", "{text}"),
])

chain = prompt_template | llm

# 4. إعداد السيرفر (FastAPI)
app = FastAPI(title="Cython AI Server")

# شكل البيانات اللي راح توصلنا من التطبيق او الموقع
class UserRequest(BaseModel):
    message: str

# نقطة الاتصال (Endpoint)
@app.post("/chat")
async def chat_endpoint(request: UserRequest):
    try:
        # ارسال الرسالة للموديل وانتظار الجواب
        response = chain.invoke({"text": request.message})
        return {"reply": response.content}
    except Exception as e:
        return {"error": str(e)}

# رسالة ترحيبية للتأكد ان السيرفر يشتغل
@app.get("/")
def read_root():
    return {"Status": "Cython is Online and Ready"}