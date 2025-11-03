# ── app.py ───────────────────────────────────────────────────────────────────
import os
import re
import io
import json
import configparser
import pathlib
from datetime import datetime

import pandas as pd
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash, send_from_directory

try:
    from google import genai
except Exception: 
    genai = None

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret")

# ── Config ───────────────────────────────────────────────────────────────────

script_directory = str(pathlib.Path(__file__).parent.resolve())
config = configparser.ConfigParser()
config.read(script_directory + "/.config")


app.config["UPLOAD_FOLDER"] = config.get("APP", "UPLOAD_FOLDER", fallback= script_directory + "/uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

df = None
filepath = None
current_index = 0
cols = []

# Load runtime options from config
KEYWORDS = json.loads(config.get("APP", "KEYWORDS", fallback="[]"))
CLASSES = json.loads(config.get("APP", "CLASSES", fallback="[\"Positive\",\"Negative\"]"))
HIGHLIGHT_COLOR = config.get("APP", "HIGHLIGHT_COLOR", fallback="#ffdebf").replace("\"","")
DEFAULT_PROMPT = config.get("GEMINI", "PROMPT", fallback="Classify the text into one of: {classes}.")[1:-2]
GEMINI_MODEL = config.get("GEMINI", "MODEL", fallback="gemini-1.5-pro").replace("\"","")
GEMINI_API_KEY = config.get("GEMINI", "API_KEY", fallback=os.environ.get("GOOGLE_API_KEY", "")).replace("\"","")

# ── Helpers ──────────────────────────────────────────────────────────────────

def list_previous_files():
    files = []
    for fname in sorted(os.listdir(app.config["UPLOAD_FOLDER"])):
        fpath = os.path.join(app.config["UPLOAD_FOLDER"], fname)
        if os.path.isfile(fpath) and os.path.splitext(fname)[1].lower() in {".csv", ".tsv", ".xlsx", ".parquet", ".json"}:
            files.append({
                "name": fname,
                "size": os.path.getsize(fpath),
                "mtime": datetime.fromtimestamp(os.path.getmtime(fpath)),
            })
    return files


def load_dataframe(path, selected_columns=None):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df_local = pd.read_csv(path)
    elif ext == ".tsv":
        df_local = pd.read_csv(path, sep="\t")
    elif ext == ".xlsx":
        df_local = pd.read_excel(path)
    elif ext == ".parquet":
        df_local = pd.read_parquet(path)
    elif ext == ".json":
        df_local = pd.read_json(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    if selected_columns:
        keep = [c for c in selected_columns if c in df_local.columns]
        df_local = df_local[keep]
    return df_local.reset_index(drop=True)


def highlight_keywords(text, keywords=KEYWORDS):
    if pd.isna(text):
        return ""
    if not isinstance(text, str):
        text = str(text)
    if not keywords:
        return BeautifulSoup(text, "html.parser").prettify() if ("<" in text and ">" in text) else text

    flags = re.IGNORECASE
    # Escape keywords for regex and join
    pattern = re.compile(r"(" + "|".join(re.escape(k) for k in keywords) + r")", flags)

    def repl(m):
        return f'<mark style="background:{HIGHLIGHT_COLOR}; padding:0 2px; border-radius:3px;">{m.group(0)}</mark>'

    return pattern.sub(repl, text)


def gemini_client():
    if not (genai and GEMINI_API_KEY):
        return None
    try:
        return genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        return None


def gemini_classify(text, labels=CLASSES, prompt_template=DEFAULT_PROMPT, model=GEMINI_MODEL):
    """Return one of the labels for `text` using Gemini; fall back to None.
    The prompt should strongly encourage structured output.
    """
    client = gemini_client()
    if not client:
        return None, {"error": "Gemini not configured"}

    prompt = prompt_template.format(classes=", ".join(labels))
    sys_msg = (
        "You are a strict classifier. Return ONLY a JSON object with keys 'label' and 'confidence'. "
        "'label' must be one of: " + ", ".join(labels) + "."
    )
    user_text = f"Text to classify:\n{text}\n\nValid labels: {labels}"

    try:
        resp = client.models.generate_content(
            model=model,
            contents=prompt + "\n\n" + user_text
        )
        out = resp.text if hasattr(resp, "text") else json.dumps(resp, default=str)
        # Try to extract JSON
        m = re.search(r"\{[\s\S]*\}", out)
        if m:
            data = json.loads(m.group(0))
            label = data.get("label") if data.get("label") in labels else None
            return label, {"raw": out, "parsed": data}
        return None, {"raw": out}
    except Exception as e:  # pragma: no cover
        return None, {"error": str(e)}


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    files = list_previous_files()
    return render_template("welcome.html", files=files)


@app.route("/upload", methods=["POST"])
def upload():
    global filepath, df, current_index, cols
    f = request.files.get("file")
    if not f:
        flash("No file uploaded", "error")
        return redirect(url_for("index"))

    filename = f.filename
    if not filename:
        flash("Invalid filename", "error")
        return redirect(url_for("index"))

    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    f.save(save_path)
    filepath = save_path
    df = None
    cols = []
    current_index = 0
    return redirect(url_for("select_columns", fname=os.path.basename(filepath)))


@app.route("/select/<path:fname>")
def select_file(fname):
    global filepath, df, current_index, cols
    path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
    if not os.path.exists(path):
        flash("File not found", "error")
        return redirect(url_for("index"))
    filepath = path
    df = None
    cols = []
    current_index = 0
    return redirect(url_for("select_columns", fname=os.path.basename(filepath)))

@app.route("/download/<path:fname>", methods=['GET', 'POST'])
def download_file(fname):
    return send_from_directory(app.config["UPLOAD_FOLDER"], fname)

@app.route("/columns")
def select_columns():
    if not filepath:
        return redirect(url_for("index"))

    # Load columns quickly (no filtering yet)
    try:
        tmp = load_dataframe(filepath)
    except Exception as e:
        flash(f"Failed to read file: {e}", "error")
        return redirect(url_for("index"))

    return render_template("select_columns.html", fname=os.path.basename(filepath), columns=list(tmp.columns))

@app.route("/load", methods=["POST"])
def load_selected_columns():
    global df, cols, current_index, index_column
    if not filepath:
        return redirect(url_for("index"))

    cols = request.form.getlist("columns")
    index_column = request.form.get("index_column")  # <-- NEW

    try:
        df = load_dataframe(filepath, selected_columns=cols or None)
        current_index = 0
        if index_column not in df.columns:
            flash("Invalid index column", "error")
            return redirect(url_for("select_columns"))
    except Exception as e:
        flash(f"Failed to load data: {e}", "error")
        return redirect(url_for("select_columns"))

    return redirect(url_for("label_view"))

@app.route("/label")
def label_view():
    global df, index_column
    if df is None:
        return redirect(url_for("index"))
    row = df.iloc[current_index].to_dict()
    highlighted = {k: highlight_keywords(v) for k, v in row.items()}
    index_values = df[index_column].fillna("").astype(str).tolist()  # <-- sidebar labels

    return render_template(
        "label.html",
        index=current_index,
        total=len(df),
        row=row,
        row_highlighted=highlighted,
        index_values=index_values,
        index_column=index_column,
        classes=CLASSES,
        keywords=KEYWORDS,
        fname=os.path.basename(filepath) if filepath else None,
    )


@app.route("/nav", methods=["POST"]) 
def navigate():
    global current_index
    if df is None:
        return jsonify({"error": "No data loaded"}), 400

    direction = request.json.get("direction")
    if direction == "next":
        current_index = min(current_index + 1, len(df) - 1)
    elif direction == "prev":
        current_index = max(current_index - 1, 0)
    elif direction == "first":
        current_index = 0
    elif direction == "last":
        current_index = len(df) - 1

    df_filled = df.fillna('')
    row = df_filled.iloc[current_index].to_dict()
    highlighted = {k: highlight_keywords(v) for k, v in row.items()}
    return jsonify({
        "index": current_index,
        "total": len(df),
        "row": row,
        "row_highlighted": highlighted,
    })

@app.route("/goto", methods=["POST"])
def goto_row():
    global current_index
    if df is None:
        return jsonify({"error": "No data loaded"}), 400
    target = request.json.get("row")
    try:
        target = int(target)
    except Exception:
        return jsonify({"error": "Invalid row number"}), 400
    if target < 0 or target >= len(df):
        return jsonify({"error": "Row out of range"}), 400
    current_index = target
    df_filled = df.fillna('')
    row = df_filled.iloc[current_index].to_dict()
    highlighted = {k: highlight_keywords(v) for k, v in row.items()}
    return jsonify({"index": current_index, "total": len(df), "row": row, "row_highlighted": highlighted})

@app.route("/set_class", methods=["POST"]) 
def set_class():
    global filepath
    if df is None:
        return jsonify({"error": "No data loaded"}), 400

    label = request.json.get("label")
    if label not in CLASSES:
        return jsonify({"error": "Invalid class"}), 400

    # Create or update a _label column
    col = "_label"
    df.loc[current_index, col] = label
    
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        df.to_csv(filepath, index=False)
    elif ext == ".tsv":
        df.to_csv(filepath, sep="\t", index=False)
    elif ext == ".xlsx":
        df.to_excel(filepath, index=False)
    elif ext == ".parquet":
        df.to_parquet(filepath, index=False)
    elif ext == ".json":
        df.to_json(filepath, index=False)

    return jsonify({"ok": True, "index": current_index, "label": label})


@app.route("/auto_classify", methods=["POST"]) 
def auto_classify():
    if df is None:
        return jsonify({"error": "No data loaded"}), 400

    # Concatenate selected cols (or all if none selected)
    use_cols = cols or list(df.columns)
    text = " | ".join(str(df.loc[current_index, c]) for c in use_cols if c in df.columns)

    label, meta = gemini_classify(text)

    return jsonify({"ok": True, "meta": meta})


@app.route("/download")
def download_labeled():
    if df is None:
        flash("No data to download", "error")
        return redirect(url_for("index"))

    # Save alongside original name
    base = os.path.basename(filepath or "labeled.csv")
    stem, _ = os.path.splitext(base)
    out_name = f"{stem}__labeled.csv"
    out_path = os.path.join(app.config["UPLOAD_FOLDER"], out_name)
    df.to_csv(out_path, index=False)
    return send_from_directory(app.config["UPLOAD_FOLDER"], out_name, as_attachment=True)



# ── Entry ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
