import gradio
import tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# --- 1. Load the Model & Tokenizer ---
# Ensure these files are in a folder named 'models' in your project directory
print("Loading model and tokenizer...")
model = tensorflow.keras.models.load_model('models/fake_news_model.h5')

with open('models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# --- 2. Define the Prediction Function ---
def predict_news(news_text):
    if not news_text:
        return "Please enter some text."

    # Preprocessing: Convert text to sequences and pad them
    # Ensure maxlen matches the length used during training (100)
    sequences = tokenizer.texts_to_sequences([news_text])
    padded = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')
    
    # Prediction: Returns a probability between 0 and 1
    prediction = model.predict(padded)[0][0]
    
    # Classification Logic
    if prediction > 0.5:
        return "REAL NEWS"
    else:
        return "FAKE NEWS"

# --- 3. Create the Gradio Interface ---
# fn: the function to call
# inputs: multi-line textbox for pasting articles
# outputs: simple text result
app = gradio.Interface(
    fn=predict_news,                                    
    inputs=gradio.Textbox(lines=5, placeholder="Paste the news article here..."), 
    outputs="text",                                     
    title="📰 Fake News Detector",
    description="Enter a news article below to check if it is Real or Fake using AI.",
    theme="default"
)

# --- 4. Launch the App ---
if __name__ == "__main__":
    app.launch()