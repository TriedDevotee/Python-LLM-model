from flask import Flask, render_template, jsonify
import threading
import time

app = Flask(__name__)

# Shared data
latest_update = {"epoch": 0, "loss": 0.0, "time": "0s"}

@app.route('/')
def index():
    return render_template("template1.html")

@app.route('/latest')
def latest():
    return jsonify(latest_update)

# Background thread to simulate training updates
def training_loop():
    global latest_update
    for epoch in range(1, 11):
        latest_update = {"epoch": epoch, "loss": round(0.1*(10-epoch),3), "time": f"{epoch*5}s"}
        print("Server update:", latest_update)
        time.sleep(1)

if __name__ == "__main__":
    threading.Thread(target=training_loop, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, debug=True)
