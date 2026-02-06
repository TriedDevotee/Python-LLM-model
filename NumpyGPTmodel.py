import cupy as numpy
import json

seq_length = 32
batch_sizes = 64
vocab_amount = 27
learning_rate = 0.01
input_size = vocab_amount
hidden_size = 128
output_size = vocab_amount

def loadTokenizer():
    with open("tokenizer.json") as f:
        tokenizer = json.load(f)
    return tokenizer

embedding = numpy.random.uniform(-0.1, 0.1, size=(vocab_amount, hidden_size))
def embed(tokens):
    return embedding[tokens]

def tokenizeData(in_data, tokenizer):
    out_data = []
    for i in range(len(in_data)):
        out_data.append(tokenizer[in_data[i]])
    return out_data

def one_hot_data(tokenizer, sequence):
    sequence_map = []
    for character in sequence:
        character_map = []
        for i in range(vocab_amount):
            if i == character:
                character_map.append(1)
            else:
                character_map.append(0)

        sequence_map.append(character_map)
    return sequence_map

#rebuilding batch maker to save on vram so my pc doesnt explode
def batch_creator(tokenized_text, batch_size, seq_length, tokenizer):
    for i in range(0, len(tokenized_text) - seq_length - batch_size, batch_size):
        x_batch = []
        y_batch = []
        for j in range(batch_size):
            start = i + j
            x_seq = tokenized_text[start : start + seq_length]
            y_seq = tokenized_text[start + 1 : start + seq_length + 1]
            x_batch.append(x_seq)
            y_batch.append(y_seq)
        yield numpy.array(x_batch, dtype=numpy.int32), numpy.array(y_batch, dtype=numpy.int32)



def matrixInitialiser(rows, cols):
    return numpy.random.uniform(low=-0.1, high=0.1, size=(rows, cols))

def softmax(z):
    z_shift = z - numpy.max(z, axis=0, keepdims=True)
    expZ = numpy.exp(z_shift)
    return expZ / numpy.sum(expZ, axis=0, keepdims=True)

def feedForward(x_batch, W_hh, W_hy, W_xh, b_y, b_h):
    hidden_states = []
    probabilities = []

    h_prev = numpy.zeros((hidden_size, batch_sizes))
    
    for t in range(seq_length): 

        token_ids = x_batch[:, t]

        x_t = embedding[token_ids].T

        #print(f"{W_xh.shape} @ {x_t.shape} + {W_hh.shape} @ {h_prev.shape} + {b_h.shape}")


        h_t = numpy.tanh(W_xh @ x_t + W_hh @ h_prev + b_h)

        y_t = W_hy @ h_t + b_y

        p_t = softmax(y_t)

        hidden_states.append(h_t)
        probabilities.append(p_t)

        h_prev = h_t
    return hidden_states, probabilities

def back_propagation(x_batch, y_batch, hidden_states, probabilities, W_hh, W_hy, W_xh, b_y, b_h):
    batch_size = x_batch.shape[0]

    dW_xh = numpy.zeros_like(W_xh)
    dW_hh = numpy.zeros_like(W_hh)
    dW_hy = numpy.zeros_like(W_hy)
    db_h = numpy.zeros_like(b_h)
    db_y = numpy.zeros_like(b_y)

    dh_next = numpy.zeros((hidden_size, batch_size))

    for t in reversed(range(seq_length)):
        token_ids = y_batch[:, t].astype(numpy.int32)   # (batch_size,)
        y_one = numpy.zeros((vocab_amount, batch_size), dtype=numpy.float32)
        y_one[token_ids, numpy.arange(batch_size)] = 1.0

        dy = probabilities[t] - y_one   # (vocab_amount, batch_size)

        dW_hy += dy @ hidden_states[t].T    # (vocab_amount, batch_size) @ (batch_size, hidden_size) -> (vocab_amount, hidden_size)
        db_y += numpy.sum(dy, axis=1, keepdims=True)

        dh = W_hy.T @ dy + dh_next                      # (hidden_size, batch_size)
        dh_raw = dh * (1 - hidden_states[t]**2)         # elementwise

        token_ids_x = x_batch[:, t].astype(numpy.int32)     # (batch_size,)
        x_t_emb = embed(token_ids_x).T                      # (embedding_dim, batch_size)
        dW_xh += dh_raw @ x_t_emb.T                         # (hidden_size, batch_size) @ (batch_size, embedding_dim) -> (hidden_size, embedding_dim)

        if t == 0:
            h_prev = numpy.zeros_like(hidden_states[t])
        else:
            h_prev = hidden_states[t - 1]

        dW_hh += dh_raw @ h_prev.T
        db_h += numpy.sum(dh_raw, axis=1, keepdims=True)

        dh_next = W_hh.T @ dh_raw


    dW_xh /= batch_size
    dW_hh /= batch_size
    dW_hy /= batch_size
    db_h  /= batch_size
    db_y  /= batch_size

    W_xh -= learning_rate * dW_xh
    W_hh -= learning_rate * dW_hh
    W_hy -= learning_rate * dW_hy
    b_h  -= learning_rate * db_h
    b_y  -= learning_rate * db_y
    

    batch_idx = numpy.arange(batch_size)[:, None]
    seq_idx = numpy.arange(seq_length)[None, :]

    probabilities = numpy.stack(probabilities, axis=0)
    probabilities = probabilities.transpose(2, 0 ,1)

    correct_token_probs = probabilities[batch_idx, seq_idx, y_batch]

    #loss = -numpy.mean(numpy.log(correct_token_probs + 1e-9))
    #print(f"Batch loss: {loss:.4f}")

    return W_xh, W_hh, W_hy, b_h, b_y


def prepare_data():


    tokenizer = loadTokenizer()

    with open("testData.txt") as f:
        data = f.readline()

    tokenized_data = tokenizeData(data, tokenizer)

    print(f"Length of tokenised data: {len(tokenized_data)}")

    embedding_dim = hidden_size

    W_xh = matrixInitialiser(hidden_size, embedding_dim)
    W_hh = matrixInitialiser(hidden_size, hidden_size)
    b_h = matrixInitialiser(hidden_size, 1)
    W_hy = matrixInitialiser(output_size, hidden_size)
    b_y = matrixInitialiser(output_size, 1)

    return W_xh, W_hh, b_h, b_y, W_hy, tokenized_data

W_xh, W_hh, b_h, b_y, W_hy, tokenized_data = prepare_data()

with open("tokenizer.json") as f:
    tokenizer = json.load(f)

epochs = 100
batch_number = 0

for epoch in range(epochs):
    print(f"Epoch: {epoch+1}")
    for x_batch, y_batch in batch_creator(tokenized_data, batch_sizes, seq_length, tokenizer):
        hidden_states, probabilities = feedForward(x_batch, W_hh, W_hy, W_xh, b_y, b_h)
        W_xh, W_hh, W_hy, b_h, b_y = back_propagation(x_batch, y_batch, hidden_states, probabilities, W_hh, W_hy, W_xh, b_y, b_h)
        batch_number += 1

        if batch_number % 500 == 0:
            numpy.savez("ShakespeareCheckpoint.npz", W_xh=W_xh.get(), W_hh=W_hh.get(),
            W_hy=W_hy.get(), b_h=b_h.get(), b_y=b_y.get())
            print(f"Checkpoint saved at batch {batch_number}")

    numpy.get_default_memory_pool().free_all_blocks()

def generate(prompt_tokens, num_tokens_in_answer, W_xh, W_hh, W_hy, b_y, b_h):
    h_prev = numpy.zeros((hidden_size, 1))

    output_tokens = []

    for token in prompt_tokens:
        token_list = []
        token_list.append(token)


        x = numpy.array(embed(token_list)).reshape(hidden_size, 1)

        h_prev = numpy.tanh(W_xh @ x + W_hh @ h_prev + b_h)

    for i in range(num_tokens_in_answer):
        y_t = W_hy @ h_prev + b_y
        p_t = softmax(y_t)

        next_token = numpy.random.choice(range(vocab_amount), p=p_t.ravel())
        output_tokens.append(next_token)

        token_list = []
        token_list.append(next_token)

        x = numpy.array(embed(token_list)).reshape(hidden_size, 1)
        h_prev =  numpy.tanh(W_xh @ x + W_hh @ h_prev + b_h)
    
    return output_tokens

        

test_data = "hello"


run_data = tokenizeData(test_data, tokenizer)

generate_response = generate(run_data, 100, W_xh, W_hh, W_hy, b_y, b_h)

def decode(answer):
    response = ""
    
    with open("detokenizer.json") as f:
        detokenizer = json.load(f)
    
    for i in range(len(answer)):
        letter = str(detokenizer[str(answer[i])])
        response += letter

    return response

print(decode(generate_response))

numpy.savez(
    "movieModel.npz",
    W_xh = W_xh.get(),
    W_hy = W_hy.get(),
    W_hh = W_hh.get(),
    b_h = b_h.get(),
    b_y = b_y.get()
)
