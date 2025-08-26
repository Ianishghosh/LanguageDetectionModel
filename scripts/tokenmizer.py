from transformers import Wav2Vec2Processor

# Path to your local models folder
model_path = "C:/Users/anish/OneDrive/Desktop/Language_Transition_Model/models"

# Download processor for wav2vec2-large-xlsr-53
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")

# Save processor locally
processor.save_pretrained(model_path)

print("Processor (with tokenizer + feature extractor) saved in:", model_path)
