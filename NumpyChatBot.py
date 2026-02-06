import numpy
import json

data = numpy.load("shakespeareCheckpoint.npz")

W_xh = data["W_xh"]
W_hh = data["W_hh"]
W_hy = data["W_hy"]
b_h = data["b_h"]
b_y = data["b_y"]

hidden_size = 128
vocab_amount = 27

embedding = numpy.random.uniform(-0.1, 0.1, size=(vocab_amount, hidden_size))

with open("tokenizer.json") as f1:
    tokenizer = json.load(f1)

with open("detokenizer.json") as f2:
    detokenizer = json.load(f2)

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


def generate(prompt_tokens, num_tokens_in_answer, W_xh, W_hh, W_hy, b_y, b_h):
    h_prev = numpy.zeros((hidden_size, 1))

    output_tokens = []

    for token in prompt_tokens:
        token_list = []
        token_list.append(token)


        x = numpy.array(embedding[token_list]).reshape(hidden_size, 1)

        h_prev = numpy.tanh(W_xh @ x + W_hh @ h_prev + b_h)

    for i in range(num_tokens_in_answer):
        y_t = W_hy @ h_prev + b_y
        p_t = softmax(y_t)

        next_token = numpy.random.choice(range(vocab_amount), p=p_t.ravel())
        output_tokens.append(next_token)

        token_list = []
        token_list.append(next_token)

        x = numpy.array(embedding[token_list]).reshape(hidden_size, 1)
        h_prev =  numpy.tanh(W_xh @ x + W_hh @ h_prev + b_h)
    
    return output_tokens

def softmax(z):
    z_shift = z - numpy.max(z, axis=0, keepdims=True)
    expZ = numpy.exp(z_shift)
    return expZ / numpy.sum(expZ, axis=0, keepdims=True)

def tokenizeData(in_data, tokenizer):
    out_data = []
    for i in range(len(in_data)):
        out_data.append(tokenizer[in_data[i]])
    return out_data


def decode(answer):
    response = ""
    
    with open("detokenizer.json") as f:
        detokenizer = json.load(f)
    
    for i in range(len(answer)):
        letter = str(detokenizer[str(answer[i])])
        response += letter

    return response

while True:
    user_input = input(">>>  ")
    run_data = tokenizeData(user_input, tokenizer)

    generate_response = generate(run_data, 20, W_xh, W_hh, W_hy, b_y, b_h)
    print(decode(generate_response))
