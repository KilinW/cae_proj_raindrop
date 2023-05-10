# This script generate two overlapped droplet signal
# The force of each drop last for a fixed value of 0.0005s
# Overlapped difference is ranged from 0.001s to 0.1s with the resolution of 0.001s
# The force of droplet is ranged from 0.1 to 3 with the resolution of 0.1

from core.cantilever_piezo_voltage import piezo_film
import numpy as np
import scipy.io.wavfile as wavf
import shutil
import os

seed = 123456789
np.random.seed(seed)

base_folder = f"data_{seed}"
subfolders = ["train", "test", "validation"]
inner_folders = ["mix", "s1", "s2"]

# Create the folder structure
for subfolder in subfolders:
    for inner_folder in inner_folders:
        # Combine folder names
        folder_path = os.path.join(base_folder, subfolder, inner_folder)
        
        # Check if the directory already exists, and create it if it doesn't
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Directory '{folder_path}' created.")
        else:
            print(f"Directory '{folder_path}' already exists.")

model = piezo_film()
model.time_span(time_span=0.2, step=16000)

def choose_folder():
    values = ['train', 'valid', 'test']
    probabilities = [0.7, 0.2, 0.1]
    choice = np.random.choice(a=values, p=probabilities)
    return choice

for time_diffs in np.arange(0.001, 0.1, 0.001):
    for force_a in np.arange(0.1, 3.1, 0.1):
        for force_b in np.arange(0.1, 3.1, 0.1):
            folder = choose_folder()
            
            model.set_force(lambda t: force_a if t < time_diffs else force_a + force_b)
            r = model.voltage()
            wavf.write(f'./data_{seed}/{folder}/mixture/{force_a:01.1f}_{force_b:01.1f}_{time_diffs:01.3f}.wav', 16000, r.y[ 2 ].astype(np.float32)/5)
            
            model.set_force(lambda t: force_a)
            r = model.voltage()
            wavf.write(f'./data_{seed}/{folder}/source1/{force_a:01.1f}_{force_b:01.1f}_{time_diffs:01.3f}.wav', 16000, r.y[ 2 ].astype(np.float32)/5)
            
            model.set_force(lambda t: force_b if t > time_diffs else 0)
            r = model.voltage()
            wavf.write(f'./data_{seed}/{folder}/source2/{force_a:01.1f}_{force_b:01.1f}_{time_diffs:01.3f}.wav', 16000, r.y[ 2 ].astype(np.float32)/5)

## Make a copy of current setting to data folder
shutil.copy('data_gen.py', f'data_{seed}/data_gen.py')
shutil.copytree('core', f'data_{seed}/core')
            