# from bark.bark.api import generate_audio
# from bark.bark.generation import SAMPLE_RATE, preload_models
# from scipy.io.wavfile import write as write_wav
# from IPython.display import Audio 

# preload_models()

# # generate audio from text
# text_prompt = """
#      Hello, my name is Suno. And, uh — and I like pizza. [laughs] 
#      But I also have other interests such as playing tic tac toe.
# """
# audio_array = generate_audio(text_prompt)

# # save audio to disk
# write_wav("bark_generation.wav", SAMPLE_RATE, audio_array)
  
# # play text in notebook
# Audio(audio_array, rate=SAMPLE_RATE)

from bark.bark.api import semantic_to_waveform, generate_text_semantic
from bark.bark.generation import SAMPLE_RATE
import scipy.io.wavfile

# Keep everything on CPU
device = "cpu"

# Step 1: Generate semantic tokens from text
text_prompt = "Hello, this is a test using Bark in a lightweight way."
semantic_tokens = generate_text_semantic(text_prompt, history_prompt=None, temp=0.7, top_k=50)

# Step 2: Convert to audio waveform
audio_array = semantic_to_waveform(semantic_tokens, history_prompt=None)

# Step 3: Save to file
scipy.io.wavfile.write("output.wav", SAMPLE_RATE, audio_array)
print("✅ Audio generated and saved to output.wav")
