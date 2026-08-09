from flask import Flask, render_template, request, jsonify
from ollama import chat

# ----------------------------------------
# Create Flask Application
# ----------------------------------------

app = Flask(__name__)


# ----------------------------------------
# Function to interact with Ollama
# ----------------------------------------

def ask_llm(question):

    response = chat(
        model="llama3.2",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI tutor."
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    return response.message.content


# ----------------------------------------
# Home Page - Browser Chatbot
# ----------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    question = ""
    answer = ""

    if request.method == "POST":

        question = request.form["question"]

        answer = ask_llm(question)

    return render_template(
        "index3.html",
        question=question,
        answer=answer
    )


# ----------------------------------------
# API Endpoint
# ----------------------------------------

@app.route("/api/chat", methods=["POST"])
def api_chat():

    data = request.get_json()

    question = data.get("question")

    if not question:
        return jsonify({
            "error": "question is required"
        }), 400

    answer = ask_llm(question)

    return jsonify({
        "question": question,
        "answer": answer
    })


# ----------------------------------------
# Health Check
# ----------------------------------------

@app.route("/health")
def health():

    return {
        "status": "UP",
        "application": "AWS Flask Chatbot"
    }


# ----------------------------------------
# Main
# ----------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )