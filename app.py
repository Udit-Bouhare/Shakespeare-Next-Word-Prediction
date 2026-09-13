import re
import torch
import torch.nn as nn
import streamlit as st


# Same model class used during training
class LSTMModel(nn.Module): 

    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 100)
        self.lstm = nn.LSTM(100, 150, batch_first=True)
        self.fc = nn.Linear(150, vocab_size)

    def forward(self, x): 
        embedded = self.embedding(x)
        lstm_output , (final_hidden_state, final_cell_state) = self.lstm(embedded)
        output = self.fc(final_hidden_state.squeeze(0))
        return output


# Load saved PyTorch model once
@st.cache_resource
def load_lstm_model():
    checkpoint = torch.load("lstm_checkpoint.pth",map_location="cpu",weights_only=True)
    model = LSTMModel(vocab_size=checkpoint["vocab_size"]) 
    model.load_state_dict(checkpoint["model_state_dict"]) 
    model.eval()

    return model, checkpoint["vocab"], checkpoint["input_sequence_length"]
    

# Load everything
model, vocab, max_sequence_len = load_lstm_model()


# 3. REVERSE VOCABULARY
idx_to_word = {
    index: word
    for word, index in vocab.items()
}

# 4. STREAMLIT UI
st.title("LSTM Next Word Prediction")

st.write(
    "Enter some text and the model will predict the next word."
)


text = st.text_input(
    "Enter your text:",
    placeholder="Example: to be or not"
)



# 5. PREDICTION
if st.button("Predict Next Word"):

    if not text.strip():

        st.warning("Please enter some text.")

    else:

        # Tokenization

        words = re.findall(
            r"\b[a-z]+\b",
            text.lower()
        )


        # Convert words → IDs
        token_ids = [
            vocab.get(
                word,
                vocab["<UNK>"]
            )
            for word in words
        ]


        # Model input length
        input_length = max_sequence_len


        # keep only the last 13
        token_ids = token_ids[-input_length:]


        # Padding
        # Your training code uses 0 for padding.
        # <UNK> is also 0 in your vocabulary.

        if len(token_ids) < input_length:

            padding_length = (
                input_length - len(token_ids)
            )

            token_ids = (
                [0] * padding_length
                + token_ids
            )


        # Convert to tensor
        input_tensor = torch.tensor(
            [token_ids],
            dtype=torch.long
        )


        # Prediction
        with torch.no_grad():

            output = model(input_tensor)

            predicted_id = torch.argmax(
                output,
                dim=1
            ).item()


        # Convert predicted ID → word
        predicted_word = idx_to_word.get(
            predicted_id,
            "<UNK>"
        )


        # Display
        st.success(
            f"Predicted next word: **{predicted_word}**"
        )