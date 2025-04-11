import streamlit as st
import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast, BertForSequenceClassification
import plotly.graph_objects as go

# Streamlit UI
st.set_page_config(page_title="Sentiment Analyzer", layout="wide", page_icon="🎭")
# Load model and tokenizer
@st.cache_resource
def load_model():
    model_path = "./model"
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model()

# Prediction function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()
        neg_confidence = probs[0][0].item()
        pos_confidence = probs[0][1].item()

    label = "Positive 😊" if predicted_class == 1 else "Negative 😞"
    return label, confidence, neg_confidence, pos_confidence



# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.2rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .positive {
        background-color: rgba(76, 175, 80, 0.2);
        border: 1px solid #4CAF50;
    }
    .negative {
        background-color: rgba(244, 67, 54, 0.2);
        border: 1px solid #F44336;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown("<h1 class='main-header'>🎬 IMDB Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Enter a movie review below and get its sentiment prediction using BERT!</p>", unsafe_allow_html=True)

# Create two columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area("Enter a movie review:", height=200, 
                             placeholder="Type your movie review here...")

    analyze_button = st.button("Analyze Sentiment", use_container_width=True)

if analyze_button:
    if user_input.strip():
        with st.spinner("Analyzing sentiment..."):
            label, confidence, neg_conf, pos_conf = predict_sentiment(user_input)
            
            # Display results
            result_class = "positive" if "Positive" in label else "negative"
            
            with col2:
                st.markdown(f"<div class='result-box {result_class}'>", unsafe_allow_html=True)
                st.markdown(f"### Prediction: {label}")
                st.markdown(f"**Confidence:** {confidence:.2%}")
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Create gauge chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = pos_conf if "Positive" in label else neg_conf,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': f"Sentiment Score"},
                    gauge = {
                        'axis': {'range': [0, 1]},
                        'bar': {'color': "#1E88E5"},
                        'steps': [
                            {'range': [0, 0.33], 'color': "rgba(244, 67, 54, 0.3)"},
                            {'range': [0.33, 0.67], 'color': "rgba(255, 152, 0, 0.3)"},
                            {'range': [0.67, 1], 'color': "rgba(76, 175, 80, 0.3)"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.5
                        }
                    }
                ))
                
                fig.update_layout(height=250, margin=dict(t=50, b=0, l=25, r=25))
                st.plotly_chart(fig, use_container_width=True)
                
                # Sentiment distribution
                labels = ['Negative', 'Positive']
                values = [neg_conf, pos_conf]
                colors = ['rgba(244, 67, 54, 0.7)', 'rgba(76, 175, 80, 0.7)']
                
                fig = go.Figure(data=[go.Bar(
                    x=labels,
                    y=values,
                    marker_color=colors
                )])
                
                fig.update_layout(
                    title="Sentiment Distribution",
                    yaxis=dict(
                        title="Probability",
                        range=[0, 1]
                    ),
                    height=250,
                    margin=dict(t=50, b=0, l=25, r=25)
                )
                
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please enter a review to analyze.")

# Add footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Built with Streamlit and BERT | Movie review sentiment analysis</p>
</div>
""", unsafe_allow_html=True)