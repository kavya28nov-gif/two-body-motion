import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle # noqa
from collections import deque

# ── Simulation parameters ──────────────────────────────────────────────
G = 1.0          # gravitational constant (normalised)
m1 = 1.0         # mass of body 1
m2 = 1.0         # mass of body 2  (change ratio here, e.g. 2.0)
dt = 0.001       # time step (smaller = more accurate)
SPEED = 5        # physics steps per animation frame

# ── Initial conditions (circular orbit) ────────────────────────────────
sep = 2.0        # initial separation

# Centre of mass stays fixed; each body orbits it
r1 = sep * m2 / (m1 + m2)   # distance of body 1 from CoM
r2 = sep * m1 / (m1 + m2)   # distance of body 2 from CoM

v_circ = np.sqrt(G * (m1 + m2) / sep)
v1 = v_circ * m2 / (m1 + m2)
v2 = v_circ * m1 / (m1 + m2)

# Positions and velocities as numpy arrays [x, y]
pos1 = np.array([-r1, 0.0])
pos2 = np.array([ r2, 0.0])
vel1 = np.array([0.0, -v1])
vel2 = np.array([0.0,  v2])

# ── Trail length = 0.3% of orbital circumference (in time steps) ───────
orbit_circumference = 2 * np.pi * ((r1 + r2) / 2)   # rough mean radius
trail_time          = 0.003 * orbit_circumference     # 0.3 % of circumference
trail_steps         = max(30, int(trail_time / dt))   # convert to steps

trail1 = deque(maxlen=trail_steps)
trail2 = deque(maxlen=trail_steps)

# ── Runge-Kutta 4 integrator ───────────────────────────────────────────
def acceleration(p1, p2, m_other):
    diff = p2 - p1
    dist = np.linalg.norm(diff)
    dist = max(dist, 0.05)          # softening to avoid singularity
    return G * m_other / dist**2 * diff / dist

def rk4_step(p1, v1, p2, v2, dt):
    def derivs(p1, v1, p2, v2):
        a1 = acceleration(p1, p2, m2)
        a2 = acceleration(p2, p1, m1)
        return v1, a1, v2, a2

    dp1k1, dv1k1, dp2k1, dv2k1 = derivs(p1,             v1,             p2,             v2)
    dp1k2, dv1k2, dp2k2, dv2k2 = derivs(p1+dp1k1*dt/2,  v1+dv1k1*dt/2,  p2+dp2k1*dt/2,  v2+dv2k1*dt/2)
    dp1k3, dv1k3, dp2k3, dv2k3 = derivs(p1+dp1k2*dt/2,  v1+dv1k2*dt/2,  p2+dp2k2*dt/2,  v2+dv2k2*dt/2)
    dp1k4, dv1k4, dp2k4, dv2k4 = derivs(p1+dp1k3*dt,    v1+dv1k3*dt,    p2+dp2k3*dt,    v2+dv2k3*dt)

    new_p1 = p1 + dt/6 * (dp1k1 + 2*dp1k2 + 2*dp1k3 + dp1k4)
    new_v1 = v1 + dt/6 * (dv1k1 + 2*dv1k2 + 2*dv1k3 + dv1k4)
    new_p2 = p2 + dt/6 * (dp2k1 + 2*dp2k2 + 2*dp2k3 + dp2k4)
    new_v2 = v2 + dt/6 * (dv2k1 + 2*dv2k2 + 2*dv2k3 + dv2k4)
    return new_p1, new_v1, new_p2, new_v2

# ── Matplotlib figure ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8), facecolor='#0a0a12')
ax.set_facecolor('#0a0a12')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Two-Body Gravitational Motion', color='white', fontsize=13, pad=10)

# Stars (static background)
rng = np.random.default_rng(42)
star_x = rng.uniform(-3, 3, 120)
star_y = rng.uniform(-3, 3, 120)
star_a = rng.uniform(0.1, 0.5, 120)
ax.scatter(star_x, star_y, s=0.8, color='white', alpha=0.4, zorder=0)

# Trail lines
trail_line1, = ax.plot([], [], color='#64b4ff', linewidth=1.2, alpha=0.7, zorder=1)
trail_line2, = ax.plot([], [], color='#ffa050', linewidth=1.2, alpha=0.7, zorder=1)

# Body markers
size1 = 80 + m1 * 60
size2 = 80 + m2 * 60
body1_scatter = ax.scatter([], [], s=size1, color='#64b4ff', zorder=3,
                            edgecolors='#a0d4ff', linewidths=1.5)
body2_scatter = ax.scatter([], [], s=size2, color='#ffa050', zorder=3,
                            edgecolors='#ffd0a0', linewidths=1.5)

label1 = ax.text(0, 0, 'm₁', color='#a0d4ff', fontsize=10, zorder=4,
                 ha='left', va='bottom')
label2 = ax.text(0, 0, 'm₂', color='#ffd0a0', fontsize=10, zorder=4,
                 ha='left', va='bottom')

# ── Animation update ──────────────────────────────────────────────────
def update(frame):
    global pos1, vel1, pos2, vel2

    for _ in range(SPEED):
        pos1, vel1, pos2, vel2 = rk4_step(pos1, vel1, pos2, vel2, dt)
        trail1.append(pos1.copy())
        trail2.append(pos2.copy())

    if len(trail1) > 1:
        t1 = np.array(trail1)
        t2 = np.array(trail2)
        trail_line1.set_data(t1[:, 0], t1[:, 1])
        trail_line2.set_data(t2[:, 0], t2[:, 1])

    body1_scatter.set_offsets([pos1])
    body2_scatter.set_offsets([pos2])

    offset = 0.12
    label1.set_position((pos1[0] + offset, pos1[1] + offset))
    label2.set_position((pos2[0] + offset, pos2[1] + offset))

    return trail_line1, trail_line2, body1_scatter, body2_scatter, label1, label2

ani = animation.FuncAnimation(fig, update, interval=16, blit=True, cache_frame_data=False)

plt.tight_layout()
plt.show()