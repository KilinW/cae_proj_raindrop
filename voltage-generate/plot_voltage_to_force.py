from core.cantilever_piezo_voltage import piezo_film
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

model = piezo_film()
model.time_span(time_span=0.2, step=10000)
model.set_force(lambda t: 1 if t<0.1 else 0)
model.set_thickness(substrate=0.001)
#model.set_youngs(substrate=69*1e9)

youngs_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5,
          4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
          10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40,
          45, 50, 55, 60, 65, 70]

# Function to update each frame
def animate_A(youngs):
    model.set_youngs(substrate=youngs*1e9)
    r = model.voltage(method='RK23')
    voltages = r.y[2]
    d_voltages = np.gradient(voltages, r.t[1] - r.t[0])
    dd_voltages = np.gradient(d_voltages, r.t[1] - r.t[0])
    i_voltages = np.cumsum(voltages) * (r.t[1] - r.t[0])
    force_A, A, B, C, D = model.voltage_to_force_A(voltages, r.t[1] - r.t[0])
    for ax in axs:
        ax.clear()
    axs[0].plot(r.t, C)
    axs[1].plot(r.t, B)
    axs[2].plot(r.t, A)
    axs[3].plot(r.t, D)
    axs[4].plot(r.t, force_A)
    axs[0].set_title('voltage')
    axs[1].set_title('d_voltage')
    axs[2].set_title('dd_voltage')
    axs[3].set_title('i_voltage')
    axs[4].set_title('force')
    fig.suptitle(f'Youngs Modulus: {youngs}e9')
    fig.tight_layout()

def animate_B(youngs):
    model.set_youngs(substrate=youngs*1e9)
    r = model.voltage(method='RK23')
    voltages = r.y[2]
    d_voltages = np.gradient(voltages, r.t[1] - r.t[0])
    dd_voltages = np.gradient(d_voltages, r.t[1] - r.t[0])
    i_voltages = np.cumsum(voltages) * (r.t[1] - r.t[0])
    force_A, A, B, C = model.voltage_to_force_B(voltages, r.t[1] - r.t[0])
    for ax in axs:
        ax.clear()
    axs[0].plot(r.t, B)
    axs[1].plot(r.t, A)
    axs[2].plot(r.t, C)
    axs[3].plot(r.t, force_A)
    axs[0].set_title('voltage')
    axs[1].set_title('d_voltage')
    axs[2].set_title('dd_voltage')
    axs[3].set_title('force')
    fig.suptitle(f'Youngs Modulus: {youngs}e9')
    fig.tight_layout()

# Create the figure and the axes
fig, axs = plt.subplots(4, 1, figsize=(10, 10))

# Create the animation
ani = animation.FuncAnimation(fig, animate_B, frames=youngs_list, interval=200)

# Save the animation
ani.save('force_dif_youngs_B.gif', writer='pillow')

fig, axs = plt.subplots(5, 1, figsize=(10, 10))

# Create the animation
ani = animation.FuncAnimation(fig, animate_A, frames=youngs_list, interval=200)

# Save the animation
ani.save('force_dif_youngs_A.gif', writer='pillow')




#### Plot the eta, and d_eta calculated from the voltage
#eta, d_eta, dd_eta = model.voltage_to_eta(voltages, r.t[ 1 ]-r.t[ 0 ])
#fig, axs = plt.subplots(4, 1, figsize=(10, 10))
#axs[ 0 ].plot(r.t, eta)
#axs[ 1 ].plot(r.t, d_eta)
#axs[ 2 ].plot(r.t, dd_eta)
#axs[ 3 ].plot(r.t, voltages)
## Set title for each subplot and fig
#axs[ 0 ].set_title('eta')
#axs[ 1 ].set_title('d_eta')
#axs[ 2 ].set_title('dd_eta')
#axs[ 3 ].set_title('voltage')
#fig.suptitle('Voltage to eta')
#fig.tight_layout()
#plt.savefig('eta_from_voltage.png')

#### Plot the eta, d_eta and voltage
#fig, axs = plt.subplots(3, 1, figsize=(10, 10))
#axs[ 0 ].plot(r.t, r.y[0])
#axs[ 1 ].plot(r.t, r.y[1])
#axs[ 2 ].plot(r.t, r.y[2])
## Set title for each subplot
#axs[ 0 ].set_title('eta')
#axs[ 1 ].set_title('d_eta')
#axs[ 2 ].set_title('voltage')
#fig.suptitle('Force to voltage and eta')
#fig.tight_layout()
#plt.savefig('derived_eta.png')