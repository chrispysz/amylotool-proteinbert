from flask import Flask, request, jsonify
import uuid

from flask_cors import CORS

import numpy as np
import pickle
import os
import tensorflow as tf


ALL_AAS = 'ACDEFGHIKLMNPQRSTUVWXY'
ADDITIONAL_TOKENS = ['<OTHER>', '<START>', '<END>', '<PAD>']

# Each sequence is added <START> and <END> tokens
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
    # Note that tokenize_seq already adds <START> and <END> tokens.
    return np.array([seq_tokens + (seq_len - len(seq_tokens)) * [additional_token_to_index['<PAD>']] for seq_tokens in map(tokenize_seq, seqs)], dtype = np.int32)

def predict_window(seq):
    global model

    seq_cutoff = 39

    current_label = 0

    sequences = []
    spec_preds = []
    seq_dicts = []

    seq_proper = ''
    for aa in seq:
        seq_proper += aa+' '

    if len(seq) > seq_cutoff:
        splits = len(seq)-seq_cutoff
        for i in range(splits):
            subseq = seq[i:seq_cutoff+i+1]
            pred = model.predict([tokenize_seqs([subseq], 512),
                np.zeros((1, 8943), dtype = np.int8)])

            seq_dict = {
                "startIndex": i,
                "endIndex": seq_cutoff+i,
                "prediction": str(pred[0])
            }
            seq_dicts.append(seq_dict)


    return (seq_dicts)

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
