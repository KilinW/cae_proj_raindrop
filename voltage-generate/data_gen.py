# This script generate two overlapped droplet signal
# The force of each drop last for a fixed value of 0.0005s
# Overlapped difference is ranged from 0.001s to 0.1s with the resolution of 0.001s
# The force of droplet is ranged from 0.1 to 3 with the resolution of 0.1

from core.cantilever_piezo_voltage import piezo_film
import numpy as np
import scipy.io.wavfile as wavf
import matplotlib.pyplot as plt

model = piezo_film()
model.time_span(time_span=0.2, step=16000)

id = 0
for time_diffs in np.arange(0.001, 0.1, 0.001):
    for force_a in np.arange(2, 3, 0.1):
        for force_b in np.arange(2, 3, 0.1):
            #model.set_force(lambda t: force_a if t < time_diffs else force_a + force_b)
            #r = model.voltage()
            #wavf.write(f'./data/mix/{force_a:01.1f}_{force_b:01.1f}_{time_diffs:01.3f}.wav', 16000, r.y[ 2 ].astype(np.float32))
            
            #model.set_force(lambda t: force_a)
            #r = model.voltage()
            #wavf.write(f'./data/s1/{force_a:01.1f}_{force_b:01.1f}_{time_diffs:01.3f}.wav', 16000, r.y[ 2 ].astype(np.float32))
            
            #model.set_force(lambda t: force_b if t > time_diffs else 0)
            #r = model.voltage()
            #wavf.write(f'./data/s2/{force_a:01.1f}_{force_b:01.1f}_{time_diffs:01.3f}.wav', 16000, r.y[ 2 ].astype(np.float32))
            with open(f'./data/piezo_voltage_tr.csv', 'a') as f:
                f.write(f'{id},1.0,/data/train/mix/{force_a:01.1f}_{force_b:01.1f}_{time_diffs:01.3f},wav,,/data/train/s1/{force_a:01.1f}_{force_b:01.1f}_{time_diffs:01.3f},wav,,/data/train/s2/{force_a:01.1f}_{force_b:01.1f}_{time_diffs:01.3f},wav,\n')
            break
        break
    id += 1