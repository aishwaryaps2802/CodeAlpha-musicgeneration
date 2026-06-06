# AI Music Generation with LSTM

An AI that composes original music by learning 
patterns from classical MIDI files using a deep 
learning LSTM neural network.

## Listen to Generated Music
Download generated.mid from the output folder 
and play it at https://www.midiplayer.eu

## How It Works
1. Classical MIDI files parsed using music21
2. Notes extracted and converted to numbers
3. LSTM model trained on note sequences
4. Model generates new note sequence
5. Sequence converted back to MIDI file

## Tech Stack
- Python
- TensorFlow and Keras
- LSTM Neural Network
- music21
- Google Colab GPU

## Project Structure
data/midi_files     → Input MIDI training files
prepare_data.py     → Extract and prepare notes
train_model.py      → Build and train LSTM model
generate_music.py   → Generate new music
output/generated.mid→ AI generated music output

## How to Run
pip install music21 tensorflow numpy flask
python prepare_data.py
python train_model.py
python generate_music.py

## Result
The model generates 200 notes of original music
after training on classical piano compositions.

