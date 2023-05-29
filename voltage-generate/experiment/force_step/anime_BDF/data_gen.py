from core.cantilever_piezo_voltage import piezo_film
import matplotlib.pyplot as plt
import matplotlib.animation as anime
import numpy as np

figure, ax = plt.subplots(2, 1)
model = piezo_film()
writer = anime.PillowWriter(fps=24) 

model.time_span(time_span=0.3, step=100000)

# This script generate two overlapped droplet signal
# The force of each drop last for a fixed value of 0.0005s
# Overlapped difference is ranged from 0.001s to 0.1s with the resolution of 0.001s
# The force of droplet is ranged from 0.1 to 3 with the resolution of 0.1

frame = []
            
def update(time_diff, model: piezo_film, ax):
    diff = time_diff / 5000
    force_a = 2
    force_b = 2
    model.set_force(lambda t: force_a if t < diff  else force_a + force_b)
    r = model.voltage(method='BDF')

    ax[0].clear()
    ax[1].clear()

    ax[0].plot(r.t, r.y[0] * model.phi(model.L1) * 1e3)
    ax[0].set_ylabel('Displacement')
    ax[0].set_xlabel('time')
    ax[0].set_ylim([-0.4, 0.4])

    ax[1].plot(r.t, r.y[2])
    ax[1].set_ylabel('Voltage')
    ax[1].set_xlabel('time')
    ax[1].set_ylim([-2, 2])

def animate(i):
    update(i + 1, model, ax)
    return ax

ani = anime.FuncAnimation(figure, animate, frames=500, blit=False)
ani.save('Cantilever_mass_fcn_python.gif', writer=writer)

