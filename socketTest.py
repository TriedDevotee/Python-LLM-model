import socketio
import time

sio = socketio.Client(logger=True, engineio_logger=True)

@sio.event
def connect():
    print("Connected to server")

@sio.event
def disconnect():
    print("Disconnected from server")

sio.connect("http://127.0.0.1:5000")

for epoch in range(10):
    data = {
        "epoch": epoch+1,
        "loss": round(0.1 * (10 - epoch), 3),
        "time": f"{epoch*5}s"
    }
    print("Sending update:", data)
    sio.emit("training update", data)
    time.sleep(1)
