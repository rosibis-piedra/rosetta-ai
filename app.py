from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import numpy as np
import os

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per day"]
)

app = Flask(__name__)
CORS(app, 
     resources={r"/*": {"origins": "*"}},
     allow_headers=["Content-Type"],
     methods=["GET", "POST", "OPTIONS"])

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.route('/')
def home():
    return jsonify({"status": "Rosetta AI Backend Running"})

@app.route('/analyze', methods=['POST', 'OPTIONS'])
@limiter.limit("10 per minute")
def analyze():
    if request.method == 'OPTIONS':
        return '', 204
    
    data = request.json
    word = data.get('word', '')
    
    if not word:
        return jsonify({"error": "No word provided"}), 400
    
    try:
        print(f"Analyzing: {word}")
        
        # Análisis
        word_emb = get_embedding(word)
        
        # Contextos
        contexts = {
    "technical": ["file", "document", "code", "system"],
    "emotional": ["feeling", "heart", "soul", "emotion"],
    "physical": ["object", "material", "body", "thing"],
    "abstract": ["idea", "concept", "thought", "notion"]
    }
        
        scores = {}
        for name, words in contexts.items():
            ctx_embs = [get_embedding(w) for w in words]
            ctx_avg = np.mean(ctx_embs, axis=0)
            scores[name] = float(cosine_similarity(word_emb, ctx_avg))
        
        # Calcular ambigüedad y clarity
        score_values = list(scores.values())
        max_score = max(score_values)
        clarity = max_score * 100
        ambiguity = 1 - max_score
        
        # Interpretación
        if ambiguity > 0.7:
            interpretation = "Very ambiguous"
            recommendation = "Use a more specific word in your prompts"
            level = "high"
            emoji = "🚨"
        elif ambiguity > 0.4:
            interpretation = "Moderately ambiguous"
            recommendation = "Consider a clearer alternative"
            level = "medium"
            emoji = "⚡"
        else:
            interpretation = "Clear word"
            recommendation = "Good choice for prompting"
            level = "low"
            emoji = "✅"
        
        result = {
            "word": word,
            "contexts": scores,
            "clarity": {
                "score": float(clarity),
                "ambiguity": float(ambiguity),
                "interpretation": interpretation,
                "recommendation": recommendation,
                "level": level,
                "emoji": emoji
            }
        }
        
        print(f"Clarity: {clarity:.1f}% - {interpretation}")
        return jsonify(result)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Para Render
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)