import torch
import numpy as np
import matplotlib.pyplot as plt
import spacy
import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ========================= 
# Page Configuration
# ========================= 
st.set_page_config(
    page_title="BERT Structural Probing",
    page_icon="🔬",
    layout="wide"
)

# ========================= 
# Cached Model Loading
# ========================= 
@st.cache_resource
def load_model(model_name):
    """Load and cache the BERT model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name,
        output_hidden_states=True
    )
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_spacy():
    """Load and cache spaCy model"""
    return spacy.load("en_core_web_sm")

# ========================= 
# Core Analysis Functions
# ========================= 
def generate_labeled_data(sentences, nlp):
    """Extract nouns and verbs from sentences with labels"""
    sent_texts = []
    labels = []
    
    for sent in sentences:
        doc = nlp(sent)
        sent_tokens = []
        sent_labels = []
        
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"]:
                sent_tokens.append(token.text)
                sent_labels.append(0)  # Noun
            elif token.pos_ in ["VERB", "AUX"]:
                sent_tokens.append(token.text)
                sent_labels.append(1)  # Verb
        
        if len(sent_tokens) > 1:
            sent_texts.append(" ".join(sent_tokens))
            labels.append(sent_labels)
    
    return sent_texts, labels

def extract_hidden_states(sent_texts, labels, tokenizer, model):
    """Extract hidden states from all BERT layers"""
    layer_reps = []
    layer_labs = []
    
    for sent, lab in zip(sent_texts, labels):
        enc = tokenizer(sent, return_tensors="pt")
        
        with torch.no_grad():
            out = model(**enc)
        
        for i, layer in enumerate(out.hidden_states):
            vecs = layer[0][1:-1].numpy()  # Exclude [CLS] and [SEP]
            
            if len(vecs) == len(lab):
                if len(layer_reps) <= i:
                    layer_reps.append([])
                    layer_labs.append([])
                
                layer_reps[i].append(vecs)
                layer_labs[i].append(lab)
    
    return layer_reps, layer_labs

def probe_layers(layer_reps, layer_labs, progress_bar):
    """Train probes for each layer and return accuracies"""
    accuracies = []
    num_layers = len(layer_reps)
    
    for i in range(num_layers):
        X = np.vstack(layer_reps[i])
        y = np.hstack(layer_labs[i])
        
        if len(np.unique(y)) < 2:
            accuracies.append(0.0)
            progress_bar.progress((i + 1) / num_layers)
            continue
        
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
        
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(Xtr, ytr)
        acc = accuracy_score(yte, clf.predict(Xte))
        accuracies.append(acc)
        
        progress_bar.progress((i + 1) / num_layers)
    
    return accuracies

def create_plot(accuracies):
    """Create accuracy plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(accuracies, marker="o", linewidth=2, markersize=8)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Probe Accuracy", fontsize=12)
    ax.set_title("Layer-wise Syntax Localization in BERT", fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    return fig

# ========================= 
# Main App UI
# ========================= 
st.title("🔬 BERT Structural Probing")
st.markdown("""
This application performs **structural probing** on BERT models to analyze how syntactic information 
(nouns vs. verbs) is encoded across different layers.
""")

# ========================= 
# Sidebar Configuration
# ========================= 
with st.sidebar:
    st.header("⚙️ Configuration")
    
    model_options = [
        "bert-base-uncased",
        "bert-base-cased",
        "bert-large-uncased",
        "distilbert-base-uncased"
    ]
    
    selected_model = st.selectbox(
        "Select Model",
        model_options,
        index=0
    )
    
    st.markdown("---")
    
    run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **Structural Probing** trains simple classifiers on hidden representations 
    to determine what linguistic information is captured at each layer.
    
    - **Label 0**: Nouns
    - **Label 1**: Verbs
    """)

# ========================= 
# Input Section
# ========================= 
st.header("📝 Input Data")

default_sentences = """The cat sat on the mat
She reads a book
He plays football
The dog chased the ball
Students study machine learning
Birds fly in the sky
Scientists discover new planets
Children learn mathematics
The teacher explains the concept
Programmers write efficient code"""

input_text = st.text_area(
    "Enter sentences (one per line):",
    value=default_sentences,
    height=200,
    help="Each sentence should be on a new line. The app will extract nouns and verbs for probing."
)

# ========================= 
# Analysis Execution
# ========================= 
if run_button:
    sentences = [s.strip() for s in input_text.split('\n') if s.strip()]
    
    if len(sentences) < 3:
        st.error("❌ Please provide at least 3 sentences for meaningful analysis.")
    else:
        try:
            with st.spinner("Loading models..."):
                tokenizer, model = load_model(selected_model)
                nlp = load_spacy()
            
            st.success("✅ Models loaded successfully!")
            
            # Step 1: Generate labeled data
            with st.spinner("Extracting nouns and verbs..."):
                sent_texts, labels = generate_labeled_data(sentences, nlp)
            
            if len(sent_texts) == 0:
                st.error("❌ No valid noun-verb pairs found. Please provide more diverse sentences.")
            else:
                st.info(f"📊 Found {len(sent_texts)} valid sentence fragments with {sum(len(l) for l in labels)} tokens")
                
                # Step 2: Extract hidden states
                with st.spinner("Extracting hidden states from BERT layers..."):
                    layer_reps, layer_labs = extract_hidden_states(sent_texts, labels, tokenizer, model)
                
                st.info(f"🔍 Analyzing {len(layer_reps)} layers...")
                
                # Step 3: Probe layers
                progress_bar = st.progress(0)
                with st.spinner("Training probes for each layer..."):
                    accuracies = probe_layers(layer_reps, layer_labs, progress_bar)
                
                progress_bar.empty()
                st.success("✅ Analysis complete!")
                
                # ========================= 
                # Results Display
                # ========================= 
                st.header("📊 Results")
                
                # Create two columns
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("Accuracy by Layer")
                    fig = create_plot(accuracies)
                    st.pyplot(fig)
                
                with col2:
                    st.subheader("Layer Statistics")
                    
                    # Create DataFrame
                    df = pd.DataFrame({
                        'Layer': range(len(accuracies)),
                        'Accuracy': [f"{acc:.3f}" for acc in accuracies]
                    })
                    
                    st.dataframe(df, use_container_width=True, height=400)
                    
                    # Summary stats
                    st.markdown("### Summary")
                    st.metric("Best Layer", np.argmax(accuracies))
                    st.metric("Best Accuracy", f"{max(accuracies):.3f}")
                    st.metric("Average Accuracy", f"{np.mean(accuracies):.3f}")
                
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.exception(e)

else:
    st.info("👈 Configure your settings in the sidebar and click 'Run Analysis' to begin!")

# ========================= 
# Footer
# ========================= 
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    Built with Streamlit | BERT Structural Probing Analysis
</div>
""", unsafe_allow_html=True)