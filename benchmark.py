from flask import Flask, render_template, request
import io
import sys

from qa_vector_system import ask as vector_ask
from qa_triples import qa_system as triple_ask
from qa_hybrid_system import answer as hybrid_ask

app = Flask(__name__)


def capture_print_output(func, question):
    old_stdout = sys.stdout
    buffer = io.StringIO()
    sys.stdout = buffer

    try:
        result = func(question)
    except Exception as e:
        result = f"Error: {e}"
    finally:
        sys.stdout = old_stdout

    printed_output = buffer.getvalue()

    if result is not None:
        return str(result)

    return printed_output


@app.route("/", methods=["GET", "POST"])
def home():
    question = ""
    system = "hybrid"
    result = ""

    if request.method == "POST":
        question = request.form.get("question", "").strip()
        system = request.form.get("system", "hybrid")

        if question:
            if system == "vector":
                result = capture_print_output(vector_ask, question)
            elif system == "triple":
                result = triple_ask(question)
            elif system == "hybrid":
                result = capture_print_output(hybrid_ask, question)
            else:
                result = "Invalid system selected."

    return render_template(
        "index.html",
        question=question,
        system=system,
        result=result
    )


if __name__ == "__main__":
    app.run(debug=True)