# Sentiment Analysis with BERT

This repository contains a sentiment analysis pipeline built on top of the IMDB movie reviews dataset. The model is fine-tuned using BERT (`bert-base-uncased`) for binary sentiment classification — distinguishing between positive and negative reviews.

The project also includes a Streamlit web app for running live sentiment predictions on custom input.

---

## Features

- Fine-tuned BERT model for binary sentiment classification  
- Preprocessing pipeline using Hugging Face datasets and tokenizers  
- CUDA-enabled training support for faster performance  
- Streamlit interface for real-time prediction  

---

## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kinshookk/sentiment.git
   cd sentiment
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Enable GPU acceleration**:  
   Make sure you have an NVIDIA GPU with the appropriate CUDA toolkit installed. Verify with:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

---

## Training

To fine-tune the BERT model on the IMDB dataset:

```bash
python main.py
```

Model weights will be downloaded automatically. Training results and logs will be saved to the appropriate directories.

---

## Streamlit Web App

The Streamlit interface allows users to input any text and instantly get sentiment predictions.

To launch the web app:

```bash
streamlit run app.py
```

This opens the application in your default browser, providing an intuitive UI for testing custom inputs.

---

## Project Structure

```
.
├── main.py                  # Training and evaluation script
├── app.py                   # Streamlit frontend for live predictions
├── model/                   # Directory for saving trained model files
├── data/                    # Directory for dataset or preprocessing files
├── requirements.txt         # List of dependencies
└── README.md                # Project documentation
```

---

## Dataset

- **Source**: [IMDB Movie Reviews](https://huggingface.co/datasets/imdb)  
- **Samples**: 25,000 labeled reviews  
- **Task**: Binary classification (`positive` or `negative`)  

---

## License

This project is licensed under the [MIT License](LICENSE).