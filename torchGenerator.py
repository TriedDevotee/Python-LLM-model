import torch
import torch.nn as nn
import torch.nn.functional as func
import json

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

#print(device)


vocab_size = 27
embed_dim = 48
hidden_size = 128
batch_size = 128
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

model.load_state_dict(torch.load("Models/ModelFinal.pth", map_location=device))
model.to(device)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

criterion = nn.CrossEntropyLoss()

def generate(model, start_tokens, num_tokens, device = "cuda", temperature = 1.0):
    model.eval()

    input = torch.tensor(start_tokens, dtype=torch.long, device=device).unsqueeze(0)

    h = None
    output = []
    embedded = model.embed(input)
    out, h = model.rnn(embedded, h)
    logits = model.fc(out[:, -1, :]) / temperature
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


def decode(in_data):
    with open("Tokenizers/detokenizer.json") as f:
        detokenizer = json.load(f)
    
    out_data = ""
    
    for character in in_data:
        detoken_char = detokenizer.get(str(character))
        out_data += str(detoken_char)
    return out_data

token_num = 600

while True:
    temp = 0.9

    run_data = input(">>>  ")
    run_data = tokenizeData(run_data, loadTokenizer())

    output_data = generate(model, run_data, token_num, device, temp)
    output_data = decode(output_data)

    print(output_data)
