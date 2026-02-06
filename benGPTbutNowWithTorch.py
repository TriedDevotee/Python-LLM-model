import torch
import torch.nn as nn
import torch.nn.functional as func
import json
import time
import socketio
from flask import Flask, render_template, jsonify
import threading
import time
#from web_server import sendUpdate, testConnection

app = Flask(__name__)
latest_update = {"epoch": 0, "loss": 0.0, "time": "0s"}

@app.route('/')
def index():
    return render_template("template1.html")

@app.route('/latest')
def latest():
    return jsonify(latest_update)

class RNNmodel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h=None):
        x = self.embed(x)
        out, h = self.rnn(x, h)
        out = self.fc(out)

        return out, h

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)


vocab_size = 27
embed_dim = 48
hidden_size = 128
batch_size = 64
seq_length = 16

def loadTokenizer():
    with open("Tokenizers/tokenizer.json") as f:
        tokenizer = json.load(f)
    return tokenizer

def tokenizeData(in_data, tokenizer):
    out_data = []
    for i in range(len(in_data)):
        out_data.append(tokenizer[in_data[i]])
    return out_data

def batch_creator(tokenized_text, batch_size, seq_length):
    for i in range(0, len(tokenized_text) - seq_length - batch_size, batch_size):
        x_batch = []
        y_batch = []
        for j in range(batch_size):
            start = i + j
            x_seq = tokenized_text[start : start + seq_length]
            y_seq = tokenized_text[start + 1 : start + seq_length + 1]
            x_batch.append(x_seq)
            y_batch.append(y_seq)

        x_return_batch = torch.tensor(x_batch, dtype=torch.long, device=device)
        y_return_batch = torch.tensor(y_batch, dtype=torch.long, device=device)
        yield x_return_batch, y_return_batch

def extract_data(path):
    with open(path) as f:
        in_data = f.readline()
    tokenizer = loadTokenizer()
    in_data = tokenizeData(in_data, tokenizer)
    return in_data

model = RNNmodel(vocab_size, embed_dim, hidden_size).to(device)

model.load_state_dict(torch.load("Models/modelFinal.pth", map_location=device))
model.to(device)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

criterion = nn.CrossEntropyLoss()

epoch_count = 5000
batch_count = 0

base_data = extract_data("TestData/testDataMovies.txt")
fun_data = extract_data("TestData/StupidDataCleaned.txt")

start_time = time.time()


def training_loop():
    global latest_update, batch_count, batch_size, vocab_size, hidden_size, embed_dim, seq_length

    for epoch in range(epoch_count):
        if epoch % 100 == 0:
            data = base_data
        else:
            data = fun_data

        for x_batch, y_batch in batch_creator(data, batch_size, seq_length):
            optimizer.zero_grad()
            logits, _ = model(x_batch)

            loss = criterion(logits.view(-1, vocab_size), y_batch.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_count += 1
            elapsed = time.time() - start_time

            if batch_count % 1000 == 0:
                print(f"Completed batch number {batch_count} Epoch number: {epoch}")
                latest_update = {
                "epoch" : epoch+1,
                "loss" : round(loss.item(), 3),
                "time" : f"{elapsed / 60 : 1f} minutes"
                }
        
        if epoch % 10 == 0:
            print(f"Epoch number: {epoch}, Loss {loss.item():.4f}")
            print(f"Elapsed time: {elapsed/60:.2f} minutes")

            if epoch % 50 == 0:
                torch.save(model.state_dict(), f"Models/CheckpointSave.pth")

def generate(model, start_tokens, num_tokens, device = "cuda"):
    model.eval()

    input = torch.tensor(start_tokens, dtype=torch.long, device=device).unsqueeze(0)

    h = None
    output = []
    embedded = model.embed(input)
    out, h = model.rnn(embedded, h)
    logits = model.fc(out[:, -1, :])
    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).item()
    output.append(next_token)

    with torch.no_grad():
        for i in range(num_tokens-1):
            input_ids = torch.tensor([[next_token]], dtype=torch.long, device=device)
            embedded = model.embed(input_ids)
            out, h = model.rnn(embedded, h)
            logits = model.fc(out[:, -1, :])
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            output.append(next_token)
        
    return output

test_data = "hello"
test_data = tokenizeData(test_data, tokenizer=loadTokenizer())

def decode(in_data):
    with open("tokenizers/detokenizer.json") as f:
        detokenizer = json.load(f)
    
    out_data = ""
    
    for character in in_data:
        detoken_char = detokenizer.get(str(character))
        out_data += str(detoken_char)
    return out_data


if __name__ == "__main__":
    threading.Thread(target=training_loop, daemon=True).start()
    app.run(host="192.168.5.17", port=5000, debug=True)

print(decode(generate(model, test_data, 200, device=device)))

torch.save(model.state_dict(), f"Models/modelFinal.pth")