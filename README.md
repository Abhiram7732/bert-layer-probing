📌 Overview

    This project focuses on structural probing of BERT to analyze how knowledge is localized across different layers of large language models (LLMs). Using PyTorch, Hugging Face Transformers, and Streamlit, the app provides reproducible workflows and interactive visualizations for exploring layer‑wise representations.

🎯 Objectives

    Identify which layers of BERT encode specific types of knowledge (syntax, semantics, factual).
    Compare probing performance across layers using linear classifiers.
    Provide an interactive Streamlit app for real‑time exploration.

⚡ Key Features:

    Bidirectional Training: Reads text in both directions for deeper context.
    Masked Language Modeling (MLM): Randomly masks words during training and predicts them, teaching the model contextual inference.
    Next Sentence Prediction (NSP): Learns relationships between sentences, useful for tasks like QA and dialogue.
    Pre-trained + Fine-tuned: Trained on massive corpora (Wikipedia + BookCorpus) and adaptable to downstream tasks.
    Open Source: Widely available with thousands of variants (e.g., DistilBERT, RoBERTa).

🛠️ Technologies Used: 

    Transformers Architecture: Self-attention mechanism for contextual encoding.
    PyTorch: Framework commonly used to implement and fine-tune BERT.
    Hugging Face Transformers:Popular library offering pre-trained BERT models and utilities.
    Scikit-learn: Often used for probing tasks with linear classifiers.
    Streamlit: For building interactive apps to visualize probing results (as in your project).

📊 Results

    Shallow layers: Capture surface features (syntax, word shape).
    Middle layers:Encode semantic relations and contextual meaning.
    Deeper layers: Specialize in task‑specific or factual knowledge.

demo screenshots : 

<img width="1910" height="705" alt="image" src="https://github.com/user-attachments/assets/97038ceb-6b26-4bb4-a9b8-6425bdc2c14e" />
<img width="1903" height="837" alt="image" src="https://github.com/user-attachments/assets/1845e5d0-78aa-4f0f-90b1-ee7aedbb93c1" />
<img width="1910" height="869" alt="image" src="https://github.com/user-attachments/assets/f59ba407-f26f-4da6-9f8a-cd4c0de1aacf" />
<img width="1871" height="867" alt="image" src="https://github.com/user-attachments/assets/65080eea-8f28-4705-9e10-6eb0b823cb38" />
<img width="1885" height="883" alt="image" src="https://github.com/user-attachments/assets/349262b9-829d-4db6-9c37-404b76b55f70" />




