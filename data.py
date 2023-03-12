import requests
import shutil

url = 'https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip'
r = requests.get(url, allow_redirects=True)
open('./groove-v1.0.0-midionly.zip', 'wb').write(r.content)

shutil.unpack_archive("./groove-v1.0.0-midionly.zip","./")

from miditok import REMI
from miditok.utils import get_midi_programs
from miditoolkit import MidiFile

import glob
import random
from tqdm import tqdm 
import numpy as np
import os

tokenizer = REMI()

midi_path = './groove'
midi_list = glob.glob(os.path.join(midi_path,'**/*.midi'), recursive=True)
midi_list.extend(glob.glob(os.path.join(midi_path,'**/*.mid'), recursive=True))
print(midi_list)

print("number of midi files :", len(midi_list))

random.shuffle(midi_list)

t=[]
#zero_padding = [0 for i in range(256)]
#t.extend(zero_padding)

for file in tqdm(midi_list):
    midi = MidiFile(file)
    tmp = tokenizer(midi)[0]
    if len(tmp) >= 32:
        t.append(tmp[:32])
    #t.extend(zero_padding)

train_data = np.array(t)

np.savez('./train_data.npz',train_data=train_data)