from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Chat AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SENTIMENT_MODEL_PATH = os.path.join(BASE_DIR, "src/results/sst/checkpoint-1602")
PARAPHRASE_MODEL_PATH = os.path.join(BASE_DIR, "src/results/sst2/checkpoint-16838")

sentiment_model = None
sentiment_tokenizer = None
paraphrase_model = None
paraphrase_tokenizer = None

def load_models():
    global sentiment_model, sentiment_tokenizer, paraphrase_model, paraphrase_tokenizer
    try:
        print(f"Loading sentiment model from: {SENTIMENT_MODEL_PATH}")
        sentiment_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        sentiment_model = AutoModelForSequenceClassification.from_pretrained(
            SENTIMENT_MODEL_PATH,
            local_files_only=True
        )
        sentiment_model.eval()
        print("Sentiment model loaded!")
    except Exception as e:
        print(f"Failed to load sentiment model: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        print(f"Loading paraphrase model from: {PARAPHRASE_MODEL_PATH}")
        paraphrase_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        paraphrase_model = AutoModelForSequenceClassification.from_pretrained(
            PARAPHRASE_MODEL_PATH,
            local_files_only=True
        )
        paraphrase_model.eval()
        print("Paraphrase model loaded!")
    except Exception as e:
        print(f"Failed to load paraphrase model: {e}")
        import traceback
        traceback.print_exc()

load_models()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = "default"

class ChatResponse(BaseModel):
    message: ChatMessage
    done: bool = True

@app.get("/")
async def root():
    return FileResponse("index.html")

@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    user_message = None
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_message = msg.content
            break
    
    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found")
    
    response_text = generate_response(user_message)
    
    return {
        "id": "chatcmpl-" + str(hash(user_message))[:8],
        "object": "chat.completion",
        "created": 1234567890,
        "model": request.model or "default",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(user_message.split()),
            "completion_tokens": len(response_text.split()),
            "total_tokens": len(user_message.split()) + len(response_text.split())
        }
    }

def analyze_sentiment(text: str) -> str:
    if sentiment_model is None or sentiment_tokenizer is None:
        return "Sentiment model not loaded."
    
    try:
        inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = sentiment_model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()
        
        sentiment_labels = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}
        confidence = torch.softmax(outputs.logits, dim=1)[0][prediction].item()
        
        return f"Sentiment: {sentiment_labels.get(prediction, 'Unknown')} ({confidence*100:.1f}% confidence)"
    except Exception as e:
        return f"Error analyzing sentiment: {str(e)}"

def check_paraphrase(text1: str, text2: str) -> str:
    if paraphrase_model is None or paraphrase_tokenizer is None:
        return "Paraphrase model not loaded."
    
    try:
        combined = text1 + " [SEP] " + text2
        inputs = paraphrase_tokenizer(combined, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = paraphrase_model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()
        
        labels = {0: "Not Paraphrases", 1: "Paraphrases"}
        confidence = torch.softmax(outputs.logits, dim=1)[0][prediction].item()
        
        return f"Result: {labels[prediction]} ({confidence*100:.1f}% confidence)"
    except Exception as e:
        return f"Error checking paraphrase: {str(e)}"

def generate_response(user_input: str) -> str:
    user_input_lower = user_input.lower()
    words = user_input_lower.split()
    
    import re
    
    if " vs " in user_input_lower:
        parts = user_input.split(" vs ", 1)
        if len(parts) == 2:
            return check_paraphrase(parts[0].strip(), parts[1].strip())
    
    if "sentence1:" in user_input_lower and "sentence2:" in user_input_lower:
        s1 = re.search(r'sentence1:\s*(.+)', user_input, re.IGNORECASE)
        s2 = re.search(r'sentence2:\s*(.+)', user_input, re.IGNORECASE)
        if s1 and s2:
            return check_paraphrase(s1.group(1).strip(), s2.group(1).strip())
    
    paraphrase_words = ["paraphrase", "similar", "same meaning", "duplicate"]
    if any(kw in user_input_lower for kw in paraphrase_words):
        return "I can detect paraphrases! Send me two sentences: 'Sentence1: Hello Sentence2: Hi there'"
    
    sentiment_keywords = ["sentiment", "emotion", "feel", "feeling", "mood", "positive", "negative", "analyze"]
    if any(kw in user_input_lower for kw in sentiment_keywords):
        if ":" in user_input:
            text_part = user_input.split(":", 1)[1].strip()
            return analyze_sentiment(text_part)
        return "I can analyze sentiment! Send me text like: 'I love this product!' or 'I am so tired'"
    
    help_keywords = ["help", "capabilities", "what can you do", "features"]
    if any(kw in user_input_lower for kw in help_keywords):
        return "I'm a chat assistant with capabilities:\n\nSentiment Analysis - 'I love this!'\nParaphrase Detection - 'Hello vs Hi there'\nGeneral Chat - Just talk to me!"
    
    if any(word in words for word in ["hello", "hi", "hey", "greetings"]):
        return "Hello! I'm here to help."
    
    if len(words) >= 2 and len(words) <= 20:
        return analyze_sentiment(user_input)
    
    return f"I understand: \"{user_input}\". Try sending me a sentence for sentiment analysis!"

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)