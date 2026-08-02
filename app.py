from flask import Flask, render_template, request
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
# Home Page
# ----------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    question = ""
    answer = ""

    if request.method == "POST":

        question = request.form["question"]

        answer = ask_llm(question)

    return render_template(
        "index.html",
        question=question,
        answer=answer
    )

# ----------------------------------------
# Main
# ----------------------------------------

if __name__ == "__main__":
    #app.run(debug=True)
    app.run(
    host="0.0.0.0",
    port=5000,
    debug=False
)