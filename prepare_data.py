# prepare_data.py
# Parse MIDI files and extract notes and chords

import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord

def get_notes():
    notes = []

    # Loop through all MIDI files in data folder
    for file in glob.glob('data/midi_files/*.mid'):
        print(f"Parsing: {file}")
        try:
            midi   = converter.parse(file)
            parts  = instrument.partitionByInstrument(midi)

            # Get notes from first instrument part
            if parts:
                notes_to_parse = parts.parts[0].recurse()
            else:
                notes_to_parse = midi.flat.notes

            for element in notes_to_parse:
                # Single note
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                # Chord — multiple notes together
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))

        except Exception as e:
            print(f"Skipping {file}: {e}")
            continue

    # Save notes to file
    with open('data/notes.pkl', 'wb') as f:
        pickle.dump(notes, f)

    print(f"\nTotal notes extracted: {len(notes)}")
    print(f"Unique notes: {len(set(notes))}")
    return notes

def prepare_sequences(notes, sequence_length=100):
    # Get all unique notes
    pitchnames = sorted(set(notes))
    n_vocab    = len(pitchnames)

    # Map notes to numbers
    note_to_int = {note: num for num, note in enumerate(pitchnames)}

    # Create input and output sequences
    network_input  = []
    network_output = []

    for i in range(len(notes) - sequence_length):
        seq_in  = notes[i:i + sequence_length]
        seq_out = notes[i + sequence_length]
        network_input.append([note_to_int[n] for n in seq_in])
        network_output.append(note_to_int[seq_out])

    n_patterns = len(network_input)

    # Reshape and normalize input
    X = np.reshape(network_input, (n_patterns, sequence_length, 1))
    X = X / float(n_vocab)

    # One-hot encode output
    from tensorflow.keras.utils import to_categorical
    y = to_categorical(network_output, num_classes=n_vocab)

    print(f"Training sequences: {n_patterns}")
    print(f"Vocabulary size: {n_vocab}")

    return X, y, n_vocab, pitchnames, note_to_int

if __name__ == '__main__':
    notes = get_notes()
    X, y, n_vocab, pitchnames, note_to_int = prepare_sequences(notes)
    print("Data preparation complete!")