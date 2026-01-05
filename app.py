import gradio
import tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

print("Loading model and tokenizer...")
model = tensorflow.keras.models.load_model('models/fake_news_model.h5')

with open('models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


def predict_news(news_text):
    if not news_text:
        return "Please enter some text."

    
    sequences = tokenizer.texts_to_sequences([news_text])
    padded = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')
    
    prediction = model.predict(padded)[0][0]
    
    # Classification Logic
    if prediction > 0.5:
        return "REAL NEWS"
    else:
        return "FAKE NEWS"

app = gradio.Interface(
    fn=predict_news,                                    
    inputs=gradio.Textbox(lines=5, placeholder="Paste the news article here..."), 
    outputs="text",                                     
    title=" Fake News Detector",
    description="Enter a news CHECK if it is Real or Fake .",
    theme="default"
)


if __name__ == "__main__":
    app.launch()
