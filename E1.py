import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting # noqa
from scipy.integrate import solve_ivp
# 1. 定义模型微分方程
def seasonal_factors(t, base_value, amplitude=0.3, period=365):
    """
    Generate seasonal variation using sine function
    t: time in days
    base_value: mean value
    amplitude: relative amplitude of oscillation (0-1)
    period: days per cycle (default 365 for annual cycle)
    """
    return base_value * (1 + amplitude * np.sin(2 * np.pi * t / period))

def eco_system(t, Y, params):
    P, I, B, G = Y
    # Unpack parameters in correct order
    base_r_P = params[0]
    K_P = params[1]
    epsilon_PG = params[2]
    alpha_PI = params[3]
    phi_P = params[4]
    base_alpha_I = params[5]
    mu_I = params[6]
    phi_I = params[7]
    base_beta_BI = params[8]
    a_B = params[9]
    b_I = params[10]
    kappa_B = params[11]
    delta_B = params[12]
    phi_B = params[13]
    base_beta_GI = params[14]
    a_G = params[15]
    b_Ip = params[16]
    kappa_G = params[17]
    delta_G = params[18]
    phi_G = params[19]
    c = params[20]
    base_poll_coeff = params[21]
    sigma_P = params[22]
    sigma_I = params[23]
    sigma_B = params[24]
    sigma_G = params[25]

    noise_P = sigma_P * np.random.normal(0, 1)
    noise_I = sigma_I * np.random.normal(0, 1)
    noise_B = sigma_B * np.random.normal(0, 1)
    noise_G = sigma_G * np.random.normal(0, 1)
    
    # Seasonal variations
    r_P = seasonal_factors(t, base_r_P)
    alpha_I = seasonal_factors(t, base_alpha_I)
    beta_BI = seasonal_factors(t, base_beta_BI)
    beta_GI = seasonal_factors(t, base_beta_GI)
    f_poll = seasonal_factors(t, base_poll_coeff)

    # Functional responses
    pred_BI = beta_BI * B * I / (1 + a_B*B + b_I*I + 1e-12)
    pred_GI = beta_GI * G * I / (1 + a_G*G + b_Ip*I + 1e-12)

    # Differential equations
    dPdt = r_P * P * (1 - P/K_P) - alpha_PI * P * I - epsilon_PG * P * G - phi_P * f_poll * P + P * noise_P
    dIdt = alpha_I * I * P - pred_BI - pred_GI - mu_I * I - phi_I * f_poll * I + I * noise_I
    dBdt = kappa_B * pred_BI - delta_B * B - phi_B * f_poll * B + B * noise_B
    dGdt = kappa_G * pred_GI - delta_G * G - phi_G * f_poll * G + G * noise_G

    return [dPdt, dIdt, dBdt, dGdt]
# 2. 设置参数与初始条件(构造数据)
# 这里数值仅为示例
param_dict = {
    # 作物相关 - 保持现有设置
    'r_P': 0.2,            
    'K_P': 100.0,         
    'epsilon_PG': 0.001,   
    'alpha_PI': 0.008,     
    'phi_P': 0.001,        
    
    # 害虫相关 - 增加数量以支持鸟类生存
    'base_alpha_I': 0.08 ,      # 增加繁殖效应
    'mu_I': 0.08,          # 降低死亡率
    'phi_I': 0.001,       
    
    # 鸟类相关 - 显著改善生存条件
    'base_beta_BI': 0.25,      # 显著增加捕食效率（高于蝙蝠）
    'a_B': 0.05,          # 降低干扰系数
    'b_I': 0.03,          # 降低干扰系数
    'kappa_B': 0.3,       # 增加转化效率
    'delta_B': 0.05,       # 降低死亡率
    'phi_B': 0.0001,       # 降低农药影响
    
    # 蝙蝠相关 - 适当降低优势
    'base_beta_GI': 0.3,      # Increased from original value
    'a_G': 0.02,              # Decreased interference
    'b_Ip': 0.04,             # Decreased prey interference
    'kappa_G': 0.3,           # Increased conversion efficiency
    'delta_G': 0.05,          # Decreased mortality rate
    'phi_G': 0.0001, 
    
    'c': 0.01,            # 保持低农药使用
    'base_poll_coeff': 0.2,

    'sigma_P': 0.05,  # Crop noise intensity
    'sigma_I': 0.05,  # Insect noise intensity
    'sigma_B': 0.03,  # Bird noise intensity
    'sigma_G': 0.03,  # Bat noise intensity     
}
# 初始种群 [P0, I0, B0, G0]
Y0 = [20.0, 10.0, 5.0, 2.0]  # 增加鸟类初始数量，保持其他相对稳定
# 3. 数值求解
t_span = (0, 50)          # 延长时间以观察长期行为
t_eval = np.linspace(t_span[0], t_span[1], 500)
# 使用更严格的求解设置
sol = solve_ivp(fun=lambda t, y: eco_system(t, y, tuple(param_dict.values())),
                t_span=t_span, 
                y0=Y0, 
                t_eval=t_eval,
                method='LSODA',    
                rtol=1e-6,         # 提高精度要求
                atol=1e-6,
                max_step=0.01)     # 使用更小的步长

if not sol.success:
    print("Warning: Solver did not converge!")
    print(f"Message: {sol.message}")
    # 如果求解失败，尝试减少时间范围或调整参数
    t_span = (0, 100)  # 减少时间范围
    t_eval = np.linspace(t_span[0], t_span[1], 500)
    sol = solve_ivp(fun=lambda t, y: eco_system(t, y, tuple(param_dict.values())),
                    t_span=t_span, 
                    y0=Y0, 
                    t_eval=t_eval,
                    method='BDF',
                    rtol=1e-6,
                    atol=1e-6,
                    max_step=0.5)

P_sol, I_sol, B_sol, G_sol = sol.y
# 4. 绘制随时间演化的曲线
plt.figure(figsize=(10,6))
plt.plot(sol.t, P_sol, label='P(t) - Crops', lw=2)
plt.plot(sol.t, I_sol, label='I(t) - Insects', lw=2)
plt.plot(sol.t, B_sol, label='B(t) - Birds', lw=2)
plt.plot(sol.t, G_sol, label='G(t) - Bats', lw=2)
plt.xlabel('Time')
plt.ylabel('Population size')
plt.title('Dynamics of the Agricultural Ecosystem')
plt.legend()
plt.grid(True)
plt.show()
# 5. 灵敏度分析：对关键参数做扫描（举例：化学药剂强度 c 与 害虫捕食率 alpha_PI）
c_values = np.linspace(0, 1.0, 6) # 在0~1范围内取6个值
alphaPI_values = np.linspace(0.005, 0.02, 6) # 在0.005~0.02范围取6个值
final_P = np.zeros((len(c_values), len(alphaPI_values)))
final_I = np.zeros((len(c_values), len(alphaPI_values)))
for i, cval in enumerate(c_values):
    for j, alphaPI in enumerate(alphaPI_values):
        # 修改参数
        local_params = dict(param_dict)
        local_params['c'] = cval
        local_params['alpha_PI'] = alphaPI
        sol_ij = solve_ivp(fun=lambda t, y: eco_system(t, y, tuple(local_params.values())),
                          t_span=t_span, y0=Y0, t_eval=[t_span[1]])  # 只取终末时刻
        # 终末时刻的P, I
        P_end, I_end, B_end, G_end = sol_ij.y[:, -1]
        final_P[i,j] = P_end
        final_I[i,j] = I_end
# 6. 绘制灵敏度分析的2D子图
fig, ax = plt.subplots(1,2, figsize=(12,5))
c_grid, alphaPI_grid = np.meshgrid(alphaPI_values, c_values)
# 左图: 终末P
cp1 = ax[0].contourf(alphaPI_grid, c_grid, final_P, 20, cmap='viridis')
fig.colorbar(cp1, ax=ax[0])
ax[0].set_xlabel('alpha_PI (Insect feeding rate on crops)')
ax[0].set_ylabel('c (Chemical intensity)')
ax[0].set_title('Final P (Crop) distribution')
# 右图: 终末I
cp2 = ax[1].contourf(alphaPI_grid, c_grid, final_I, 20, cmap='plasma')
fig.colorbar(cp2, ax=ax[1])
ax[1].set_xlabel('alpha_PI (Insect feeding rate on crops)')
ax[1].set_ylabel('c (Chemical intensity)')
ax[1].set_title('Final I (Insects) distribution')
plt.tight_layout()
plt.show()
# 7. 绘制3D图(示例: final P 随 c 与 alpha_PI 的变化)
fig = plt.figure(figsize=(8,6))
ax3d = fig.add_subplot(111, projection='3d')
X = alphaPI_grid
Y = c_grid
Z = final_P
ax3d.plot_surface(X, Y, Z, cmap='coolwarm', edgecolor='none')
ax3d.set_xlabel('alpha_PI (Insect feeding rate on crops)')
ax3d.set_ylabel('c (Chemical intensity)')
ax3d.set_zlabel('Final P (Crop) distribution')
ax3d.set_title('3D Surface of Final Crop Population')
plt.show()