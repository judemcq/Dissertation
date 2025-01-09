import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.optimize
import matplotlib.animation as animation
from mpl_toolkits import mplot3d

#Function
def lorenz(xyz, *, a=0.15, b=0.2, c=10):

    x, y, z = xyz
    x_dot = -y-z
    y_dot = x+a*y
    z_dot = b+z*(x-c)
    return np.array([x_dot, y_dot, z_dot])

def bandFunc(kx,E,t):
    inside_sqrt = kx**3 + t*kx - E
    ky_pos = np.sqrt(np.maximum(inside_sqrt, 0))  # Positive branch
    ky_neg = -np.sqrt(np.maximum(inside_sqrt, 0)) # Negative branch
    return ky_pos, ky_neg
    return epsilon

energy_levels = np.linspace(0, 10, 20)
kx = np.linspace(-1,1,1000)
ky = np.linspace(-1,1,1000)
plt.figure()
for E in energy_levels:
    ky_pos, ky_neg = bandFunc(kx, E, t=-0.1)
    plt.plot(kx, ky_pos)  # Positive branch
    plt.plot(kx, ky_neg)  # Negative branch

plt.xlabel(r'$k_x$')
plt.ylabel(r'$k_y$')
plt.title("Constant Energy Contours for $\epsilon = k_x^3 - k_y^2 + t \cdot k_x$")
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.show()


kx = np.linspace(-1,1,1000)
ky = np.linspace(-1,1,1000)
KX, KY = np.meshgrid(kx, ky)
def simpleVOS(KX,KY,t):
    epsilon = []
    epsilon = KX**3 - KY**2 + t*KX
    return epsilon

simpleEpsilon = simpleVOS(KX,KY,0.2)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(KX, KY, simpleEpsilon, cmap="summer", linewidth=0, antialiased=False, alpha = 0.95)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()



dt = 0.01
num_steps = 100000
t = np.linspace(0,num_steps*dt,num_steps+1)

xyz = np.empty((num_steps + 1, 3))

for i in range(num_steps):
    xyz[i + 1] = xyz[i] + lorenz(xyz[i]) * dt
"""
# Plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(*xyz.T, lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Rossler Function")
plt.show()

fig = plt.figure()
bx = fig.add_subplot()
bx.plot(t, xyz[:,0])
ax.set_xlabel("X Axis")
ax.set_title("Rossler Attractor x vs t")
plt.show() 
"""