# generate_music.py
# Load trained model and generate new music

import numpy as np
import pickle
import random
from music21              import stream, note, chord, instrument
from tensorflow.keras.models import load_model
from prepare_data         import get_notes, prepare_sequences

def generate(num_notes=200):
    print("Loading notes and model...")

    # Load notes
    notes               = get_notes()
    X, _, n_vocab, pitchnames, note_to_int = prepare_sequences(notes)

    # Reverse map — number to note
    int_to_note = {num: note for num, note in enumerate(pitchnames)}

    # Load trained model
    model = load_model('model/music_model.h5')

    # Pick a random starting sequence
    start        = random.randint(0, len(X) - 1)
    pattern      = list(X[start].flatten() * n_vocab)
    pattern      = [int(p) for p in pattern]

    generated_notes = []
    print(f"Generating {num_notes} notes...")

    for i in range(num_notes):
        # Prepare input
        input_seq = np.reshape(pattern, (1, len(pattern), 1))
        input_seq = input_seq / float(n_vocab)

        # Predict next note
        prediction  = model.predict(input_seq, verbose=0)
        index       = np.argmax(prediction)
        result      = int_to_note[index]

        generated_notes.append(result)
        pattern.append(index)
        pattern = pattern[1:]

        if (i + 1) % 50 == 0:
            print(f"Generated {i + 1}/{num_notes} notes")

    # Convert to MIDI
    save_midi(generated_notes)

def save_midi(notes, output_path='output/generated.mid'):
    import os
    os.makedirs('output', exist_ok=True)

    output_notes = []
    offset       = 0

    for pattern in notes:
        # Chord
        if '.' in pattern or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            chord_notes    = []
            for n in notes_in_chord:
                try:
                    new_note = note.Note(int(n))
                    new_note.storedInstrument = instrument.Piano()
                    chord_notes.append(new_note)
                except:
                    pass
            if chord_notes:
                new_chord = chord.Chord(chord_notes)
                new_chord.offset = offset
                output_notes.append(new_chord)
        # Single note
        else:
            try:
                new_note = note.Note(pattern)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)
            except:
                pass

        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_path)
    print(f"\nMusic saved to {output_path}")

if __name__ == '__main__':
    generate(num_notes=200)