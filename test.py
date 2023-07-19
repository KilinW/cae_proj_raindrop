from tools.keysightBin.importAgilentBin import readfile
import matplotlib.pyplot as plt

binary = "scope_9"
bin_data_path = f"realsignal/{binary}.bin"
pic_data_path = f"realsignal/{binary}.png"

time, data1 = readfile(bin_data_path, 0)
time, data2 = readfile(bin_data_path, 1)
time, data3 = readfile(bin_data_path, 2)
time, data4 = readfile(bin_data_path, 3)

#Normalize all data to -1 to 1 using numpy
#data1 = data1/max(data1)
#data2 = data2/max(data2)
#data3 = data3/max(data3)
#data4 = data4/max(data4)

#Plot the data
plt.plot(time, data1, label="Channel 1", linewidth=0.3)
plt.plot(time, data2, label="Channel 2", linewidth=0.3)
plt.plot(time, data3, label="Channel 3", linewidth=0.3)
plt.plot(time, data4, label="Channel 4", linewidth=0.3)
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.legend()
plt.savefig(pic_data_path, dpi=300)
