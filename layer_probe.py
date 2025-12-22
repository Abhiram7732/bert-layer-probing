import torch
import numpy as np
import matplotlib.pyplot as plt
import spacy

from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# =========================
# 1. Load LLM
# =========================
MODEL_NAME = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(
    MODEL_NAME,
    output_hidden_states=True
)
model.eval()

# =========================
# 2. Generate labeled data
# =========================
nlp = spacy.load("en_core_web_sm")

sentences = [
    "The cat sat on the mat",
    "She reads a book",
    "He plays football",
    "The dog chased the ball",
    "Students study machine learning"
]

sent_texts = []
labels = []

for sent in sentences:
    doc = nlp(sent)
    sent_tokens = []
    sent_labels = []

    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:
            sent_tokens.append(token.text)
            sent_labels.append(0)
        elif token.pos_ in ["VERB", "AUX"]:
            sent_tokens.append(token.text)
            sent_labels.append(1)

    if len(sent_tokens) > 1:
        sent_texts.append(" ".join(sent_tokens))
        labels.append(sent_labels)

# =========================
# 3. Extract hidden states
# =========================
layer_reps = []
layer_labs = []

for sent, lab in zip(sent_texts, labels):
    enc = tokenizer(sent, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)

    for i, layer in enumerate(out.hidden_states):
        vecs = layer[0][1:-1].numpy()
        if len(vecs) == len(lab):
            if len(layer_reps) <= i:
                layer_reps.append([])
                layer_labs.append([])
            layer_reps[i].append(vecs)
            layer_labs[i].append(lab)

# =========================
# 4. Probing
# =========================
accuracies = []

for i in range(len(layer_reps)):
    X = np.vstack(layer_reps[i])
    y = np.hstack(layer_labs[i])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, ytr)

    acc = accuracy_score(yte, clf.predict(Xte))
    accuracies.append(acc)

# =========================
# 5. Plot
# =========================
plt.plot(accuracies, marker="o")
plt.xlabel("Layer")
plt.ylabel("Probe Accuracy")
plt.title("Layer-wise Syntax Localization in BERT")
plt.grid()
plt.show()
