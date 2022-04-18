# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import librosa
from transformers import HubertForCTC, Wav2Vec2Processor

# loading model and tokenizer
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")

file = open("female.wav")

# importing wav file
speech, rate = librosa.load(file, sr=16000)

# tokenize
input_values = processor(speech, return_tensors="pt", padding="longest", sampling_rate=rate).input_values

# retrieve logits
logits = model(input_values).logits

# take argmax and decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)

# Print transcribed text

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(transcription)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
