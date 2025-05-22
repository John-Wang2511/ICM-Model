import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os

# 创建figures文件夹
if not os.path.exists('figures'):
    os.makedirs('figures')

# 定义模型参数
r = 0.1  # 作物的内在增长率
K = 2000  # 环境的承载能力
alpha = 0.01  # 作物被草食动物消耗的速率
beta = 0.005  # 作物被间接影响的速率
gamma = 0.02  # 草食动物被捕食者消耗的速率
delta = 0.05  # 草食动物的自然死亡率
epsilon = 0.03  # 捕食者的自然死亡率
Delta_delta = 0.02  # 除草剂去除后草食动物死亡率的减少量

# 定义微分方程组
def model(y, t):
    P, H, I = y
    # 除草剂去除后草食动物的死亡率调整
    delta_prime = delta - Delta_delta
    # 作物的动态
    dPdt = r * P * (1 - P / K) - alpha * P * H - beta * P * I
    # 草食动物的动态
    dHdt = alpha * P * H - gamma * H * I - delta_prime * H
    # 捕食者的动态
    dIdt = beta * P * I + gamma * H * I - epsilon * I
    return [dPdt, dHdt, dIdt]

# 初始条件
y0 = [1000, 100, 50]  # P(0) = 1000, H(0) = 100, I(0) = 50

# 时间点
t = np.linspace(0, 100, 1000)

# 求解微分方程
solution = odeint(model, y0, t)

# 提取结果
P = solution[:, 0]  # 作物种群
H = solution[:, 1]  # 草食动物种群
I = solution[:, 2]  # 捕食者种群

# 绘制图形1：作物、草食动物和捕食者种群随时间的变化
plt.figure(figsize=(10, 6))
plt.plot(t, P, label='作物种群 (P)', color='green', linestyle='-', linewidth=2)
plt.plot(t, H, label='草食动物种群 (H)', color='blue', linestyle='--', linewidth=2)
plt.plot(t, I, label='捕食者种群 (I)', color='red', linestyle='-.', linewidth=2)
plt.xlabel('时间 (t)')
plt.ylabel('种群数量')
plt.title('作物、草食动物和捕食者种群随时间的变化')
plt.legend()
plt.grid(True)
plt.savefig('figures/q3_作物草食动物捕食者种群变化.png')
plt.close()

# 绘制图形2：作物和草食动物种群的相位图
plt.figure(figsize=(10, 6))
plt.plot(P, H, label='作物 vs 草食动物', color='purple', linestyle='-', linewidth=2)
plt.xlabel('作物种群 (P)')
plt.ylabel('草食动物种群 (H)')
plt.title('作物和草食动物种群的相位图')
plt.legend()
plt.grid(True)
plt.savefig('figures/q3_作物草食动物相位图.png')
plt.close()

# 绘制图形3：作物、草食动物和捕食者种群的三维关系图
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(P, H, I, label='作物-草食动物-捕食者', color='orange')
ax.set_xlabel('作物种群 (P)')
ax.set_ylabel('草食动物种群 (H)')
ax.set_zlabel('捕食者种群 (I)')
ax.set_title('作物、草食动物和捕食者种群的三维关系')
ax.legend()
plt.savefig('figures/q3_作物草食动物捕食者三维关系.png')
plt.close()