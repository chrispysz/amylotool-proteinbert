from flask import Flask, request, jsonify
import uuid

from flask_cors import CORS

import numpy as np
import pickle
import os
import tensorflow as tf
import time


ALL_AAS = 'ACDEFGHIKLMNPQRSTUVWXY'
ADDITIONAL_TOKENS = ['<OTHER>', '<START>', '<END>', '<PAD>']

ADDED_TOKENS_PER_SEQ = 2

n_aas = len(ALL_AAS)
aa_to_token_index = {aa: i for i, aa in enumerate(ALL_AAS)}
additional_token_to_index = {token: i + n_aas for i, token in enumerate(ADDITIONAL_TOKENS)}
token_to_index = {**aa_to_token_index, **additional_token_to_index}
index_to_token = {index: token for token, index in token_to_index.items()}
n_tokens = len(token_to_index)


app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'KHRl4KwU6KkD3AdMaG2jkANKvJLxPAQ8'
app.config['TIMEOUT'] = None

model = tf.keras.models.load_model('./models/ProteinBERT')



def tokenize_seq(seq):
    other_token_index = additional_token_to_index['<OTHER>']
    return [additional_token_to_index['<START>']] + [aa_to_token_index.get(aa, other_token_index) for aa in parse_seq(seq)] + \
            [additional_token_to_index['<END>']]
            
def parse_seq(seq):
    if isinstance(seq, str):
        return seq
    elif isinstance(seq, bytes):
        return seq.decode('utf8')
    else:
        raise TypeError('Unexpected sequence type: %s' % type(seq))
def tokenize_seqs(seqs, seq_len):
    return np.array([seq_tokens + (seq_len - len(seq_tokens)) * [additional_token_to_index['<PAD>']] for seq_tokens in map(tokenize_seq, seqs)], dtype = np.int32)

def predict_window(seq, seq_cutoff = 39):
    global model

    seqqs = [seq[i:seq_cutoff+i+1] for i in range(len(seq)-seq_cutoff+1)]
    preds = model.predict([tokenize_seqs(seqqs, 512), np.zeros((len(seqqs), 8943), dtype=np.int8)])
    seq_dicts = [{"startIndex": i, "endIndex": seq_cutoff+i, "prediction": f"{pred[0]}"}
                 for i, pred in enumerate(preds)]
    return seq_dicts

@app.route('/predict/full', methods=['POST'])
def predictFull():

    try:
        sequence = request.json['sequence']
        if (sequence == "ping"):
            return jsonify(results = "Service reached")
        result = predict_window(sequence)
        return jsonify(
            results=result
        )
    except Exception as e:
        return f"An Error Occurred: {e}"

if __name__ == '__main__':
    app.run(debug = True)
