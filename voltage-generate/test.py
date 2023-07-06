from core.cantilever_piezo_voltage import piezo_film
import matplotlib.pyplot as plt
import numpy as np

model = piezo_film()
model.time_span(time_span=0.2, step=10000)
model.set_force(lambda t: np.sin(2*np.pi*40*t) if t <0.025 else 0)

r = model.voltage(method='Radau')
print(r)
voltages = r.y[ 2 ]
# Differential of voltage
d_voltages = np.gradient(voltages, r.t[ 1 ]-r.t[ 0 ])
# Integral of voltage
i_voltages = np.cumsum(voltages)*(r.t[ 1 ]-r.t[ 0 ])


### Plot the force, voltage, differential of voltage and integral of voltage in different subplots
force_B = model.voltage_to_force_B(voltages, r.t[ 1 ]-r.t[ 0 ])
fig, axs = plt.subplots(4, 1, figsize=(10, 10))
axs[ 0 ].plot(r.t, voltages)
axs[ 1 ].plot(r.t, d_voltages)
axs[ 2 ].plot(r.t, i_voltages)
axs[ 3 ].plot(r.t, force_B)
# Set title for each subplot
axs[ 0 ].set_title('voltage')
axs[ 1 ].set_title('d_voltage')
axs[ 2 ].set_title('i_voltage')
axs[ 3 ].set_title('force')
fig.suptitle('Voltage to force B')
fig.tight_layout()
plt.savefig('force_from_voltage_B.png')

### Plot the force, voltage, differential of voltage and integral of voltage in different subplots
force_A = model.voltage_to_force_A(voltages, r.t[ 1 ]-r.t[ 0 ])
fig, axs = plt.subplots(4, 1, figsize=(10, 10))
axs[ 0 ].plot(r.t, voltages)
axs[ 1 ].plot(r.t, d_voltages)
axs[ 2 ].plot(r.t, i_voltages)
axs[ 3 ].plot(r.t, force_A)
# Set title for each subplot
axs[ 0 ].set_title('voltage')
axs[ 1 ].set_title('d_voltage')
axs[ 2 ].set_title('i_voltage')
axs[ 3 ].set_title('force')
fig.suptitle('Voltage to force A')
fig.tight_layout()
plt.savefig('force_from_voltage_A.png')

### Plot the eta, and d_eta calculated from the voltage
eta, d_eta, dd_eta = model.voltage_to_eta(voltages, r.t[ 1 ]-r.t[ 0 ])
fig, axs = plt.subplots(4, 1, figsize=(10, 10))
axs[ 0 ].plot(r.t, eta)
axs[ 1 ].plot(r.t, d_eta)
axs[ 2 ].plot(r.t, dd_eta)
axs[ 3 ].plot(r.t, voltages)
# Set title for each subplot and fig
axs[ 0 ].set_title('eta')
axs[ 1 ].set_title('d_eta')
axs[ 2 ].set_title('dd_eta')
axs[ 3 ].set_title('voltage')
fig.suptitle('Voltage to eta')
fig.tight_layout()
plt.savefig('eta_from_voltage.png')

### Plot the eta, d_eta and voltage
fig, axs = plt.subplots(3, 1, figsize=(10, 10))
axs[ 0 ].plot(r.t, r.y[0])
axs[ 1 ].plot(r.t, r.y[1])
axs[ 2 ].plot(r.t, r.y[2])
# Set title for each subplot
axs[ 0 ].set_title('eta')
axs[ 1 ].set_title('d_eta')
axs[ 2 ].set_title('voltage')
fig.suptitle('Force to voltage and eta')
fig.tight_layout()
plt.savefig('derived_eta.png')