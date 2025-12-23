<h1>📌Overview:</h1>
<p>
This project focuses on structural probing of BERT to analyze how knowledge is localized across different layers of large language models (LLMs). Using PyTorch, Hugging Face Transformers, and Streamlit, the app provides reproducible workflows and interactive visualizations for exploring layer‑wise representations.
    
In addition to layer‑wise probing, our approach emphasizes sentence-based analysis:

  1. Parsing sentences into syntactic structures (dependency trees, constituency parses).

  2. Mapping tokens and spans to BERT’s hidden states for fine-grained probing.

  3. Evaluating linguistic properties (syntax, semantics, factual relations) by testing how well linear classifiers recover sentence-level features.

  4. Comparing across layers to see where BERT best encodes sentence structure versus meaning.
     </p>

<h1>🎯 Objectives:</h1>

  1. Identify which layers of BERT encode specific types of knowledge (syntax, semantics, factual).
  2. Compare probing performance across layers using linear classifiers.
  3. Provide an interactive Streamlit app for real‑time exploration.

<h1>⚡ Key Features:</h1>

   1. Bidirectional Training: Reads text in both directions for deeper context.
   2. Masked Language Modeling (MLM): Randomly masks words during training and predicts them, teaching the model contextual inference.
   3. Next Sentence Prediction (NSP): Learns relationships between sentences, useful for tasks like QA and dialogue.
   4. Pre-trained + Fine-tuned: Trained on massive corpora (Wikipedia + BookCorpus) and adaptable to downstream tasks.
   5. Open Source: Widely available with thousands of variants (e.g., DistilBERT, RoBERTa).

<h1>🛠️ Technologies Used:</h1>

   1. Transformers Architecture: Self-attention mechanism for contextual encoding.
   2. PyTorch: Framework commonly used to implement and fine-tune BERT.
   3. Hugging Face Transformers:Popular library offering pre-trained BERT models and utilities.
   4. Scikit-learn: Often used for probing tasks with linear classifiers.
   5. Streamlit: For building interactive apps to visualize probing results (as in your project).

<h1>BERT Architecture</h1>
<img width="960" height="358" alt="BERT-Embedding" src="https://github.com/user-attachments/assets/4f439838-c1ae-4502-8526-cf789c0bf186" />


<h1>📊 Results:</h1>

   1. Shallow layers: Capture surface features (syntax, word shape).
   2. Middle layers:Encode semantic relations and contextual meaning.
   3. Deeper layers: Specialize in task‑specific or factual knowledge.

<h1>demo screenshots:</h1>

<img width="1910" height="705" alt="image" src="https://github.com/user-attachments/assets/97038ceb-6b26-4bb4-a9b8-6425bdc2c14e" />
<img width="1903" height="837" alt="image" src="https://github.com/user-attachments/assets/1845e5d0-78aa-4f0f-90b1-ee7aedbb93c1" />
<img width="1910" height="869" alt="image" src="https://github.com/user-attachments/assets/f59ba407-f26f-4da6-9f8a-cd4c0de1aacf" />
<img width="1871" height="867" alt="image" src="https://github.com/user-attachments/assets/65080eea-8f28-4705-9e10-6eb0b823cb38" />
<img width="1885" height="883" alt="image" src="https://github.com/user-attachments/assets/349262b9-829d-4db6-9c37-404b76b55f70" />




