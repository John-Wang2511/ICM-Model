import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os

# 创建figures文件夹
if not os.path.exists('figures'):
    os.makedirs('figures')

# 定义模型参数
r_c = 0.5  # 作物的自然增长率
K_c = 1000  # 作物的承载能力
alpha_c = 0.01  # 害虫对作物的捕食率
r_p = 0.3  # 害虫的自然增长率
K_p = 500  # 害虫的承载能力
beta_p = 0.02  # 天敌对害虫的捕食率
gamma_p = 0.05  # 农药对害虫的死亡率
r_b = 0.2  # 天敌的自然增长率
K_b = 200  # 天敌的承载能力
beta_b = 0.1  # 天敌对害虫的捕食效率
delta_b = 0.03  # 农药对天敌的死亡率
lambda_ = 0.1  # 农药的自然降解率

# 定义微分方程组
def model(y, t):
    C, P, B, H = y
    # 农药使用策略 (假设农药使用随时间线性增加)
    u = 0.1 * t  # 农药使用量随时间线性增加
    # 作物的动态
    dCdt = r_c * C * (1 - C / K_c) - alpha_c * C * P
    # 害虫的动态
    dPdt = r_p * P * (1 - P / K_p) - beta_p * P * B - gamma_p * P * H
    # 天敌的动态
    dBdt = r_b * B * (1 - B / K_b) + beta_b * P * B - delta_b * B * H
    # 农药使用的动态
    dHdt = u - lambda_ * H
    return [dCdt, dPdt, dBdt, dHdt]

# 初始条件
y0 = [500, 100, 50, 0]  # C(0) = 500, P(0) = 100, B(0) = 50, H(0) = 0

# 时间点
t = np.linspace(0, 100, 1000)

# 求解微分方程
solution = odeint(model, y0, t)

# 提取结果
C = solution[:, 0]  # 作物种群
P = solution[:, 1]  # 害虫种群
B = solution[:, 2]  # 天敌种群
H = solution[:, 3]  # 农药使用量

# 绘制图形1：作物、害虫和天敌种群随时间的变化
plt.figure(figsize=(10, 6))
plt.plot(t, C, label='作物种群 (C)', color='green', linestyle='-', linewidth=2)
plt.plot(t, P, label='害虫种群 (P)', color='red', linestyle='--', linewidth=2)
plt.plot(t, B, label='天敌种群 (B)', color='blue', linestyle='-.', linewidth=2)
plt.xlabel('时间 (t)')
plt.ylabel('种群数量')
plt.title('作物、害虫和天敌种群随时间的变化')
plt.legend()
plt.grid(True)
plt.savefig('figures/q4_作物害虫天敌种群变化.png')
plt.close()

# 绘制图形2：农药使用量随时间的变化
plt.figure(figsize=(10, 6))
plt.plot(t, H, label='农药使用量 (H)', color='purple', linestyle='-', linewidth=2)
plt.xlabel('时间 (t)')
plt.ylabel('农药使用量')
plt.title('农药使用量随时间的变化')
plt.legend()
plt.grid(True)
plt.savefig('figures/q4_农药使用量变化.png')
plt.close()

# 绘制图形3：作物、害虫和天敌种群的三维关系图
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(C, P, B, label='作物-害虫-天敌', color='orange')
ax.set_xlabel('作物种群 (C)')
ax.set_ylabel('害虫种群 (P)')
ax.set_zlabel('天敌种群 (B)')
ax.set_title('作物、害虫和天敌种群的三维关系')
ax.legend()
plt.savefig('figures/q4_作物害虫天敌三维关系.png')
plt.close()