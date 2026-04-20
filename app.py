from flask import Flask, render_template, request, url_for, redirect
import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from werkzeug.utils import secure_filename
import time

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = load_model("model/final_model.h5", compile=False)


def model_predict(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((256, 256))

    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    mask = preds[0]

    if mask.ndim == 3:
        mask = mask.squeeze()

    mask_bin = (mask >= 0.3).astype(np.uint8)

    mask_img = Image.fromarray((mask_bin * 255).astype(np.uint8))

    # Overlay
    orig = img.convert("RGBA")
    overlay = Image.new("RGBA", orig.size, (255, 0, 0, 0))

    for y in range(mask_bin.shape[0]):
        for x in range(mask_bin.shape[1]):
            if mask_bin[y, x] == 1:
                overlay.putpixel((x, y), (255, 0, 0, 120))

    overlay_img = Image.alpha_composite(orig, overlay)

    # Analysis
    oil_pixels = np.sum(mask_bin)
    total_pixels = mask_bin.size
    oil_percent = (oil_pixels / total_pixels) * 100
    confidence = float(np.mean(mask))

    oil_present = oil_percent > 1

    analysis = {
        "oil_percent": round(oil_percent, 2),
        "confidence": round(confidence, 2),
        "status": "Oil Spill Detected" if oil_present else "No Oil Spill"
    }

    return mask_img, overlay_img, analysis


@app.route("/", methods=["GET", "POST"])
def index():

    input_image = None
    mask_image = None
    overlay_image = None
    analysis = None

    if request.method == "POST":
        file = request.files["file"]

        if file and file.filename != "":
            filename = secure_filename(file.filename)

            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            t = time.time()

            input_image = url_for("static", filename=f"uploads/{filename}", t=t)

            mask_img, overlay_img, analysis = model_predict(filepath)

            mask_name = f"mask_{filename}"
            overlay_name = f"overlay_{filename}"

            mask_path = os.path.join(app.config["UPLOAD_FOLDER"], mask_name)
            overlay_path = os.path.join(app.config["UPLOAD_FOLDER"], overlay_name)

            mask_img.save(mask_path)
            overlay_img.convert("RGB").save(overlay_path)

            mask_image = url_for("static", filename=f"uploads/{mask_name}", t=t)
            overlay_image = url_for("static", filename=f"uploads/{overlay_name}", t=t)

    return render_template(
        "index.html",
        input_image=input_image,
        mask_image=mask_image,
        overlay_image=overlay_image,
        analysis=analysis
    )


# ✅ CLEAR ROUTE
@app.route("/clear")
def clear():
    return redirect(url_for("index"))


if __name__ == "__main__":
   app.run(host="0.0.0.0", port=5000, debug=True)