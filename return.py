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
    # 确保所有输入值非负
    Y = [max(0, y) for y in Y]
    P, I, B, G, R1, R2 = Y
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
    lambda_R1 = params[26]
    lambda_R2 = params[27]
    base_r_R1 = params[28]
    base_r_R2 = params[29]
    base_beta_R1B = params[30]  # R1对鸟类的捕食率
    base_beta_R1G = params[31]  # R1对蝙蝠的捕食率
    base_beta_GR2 = params[32]  # 蝙蝠对R2的捕食率
    a_R1 = params[33]      # R1捕食的干扰系数
    a_R2 = params[34]      # R2的资源竞争系数
    phi_cR1 = params[35]
    phi_cR2 = params[36]
    delta_R1 = params[37]
    delta_R2 = params[38]
    sigma_R1 = params[39]
    sigma_R2 = params[40]

    noise_P = sigma_P * np.random.normal(0, 0.1)
    noise_I = sigma_I * np.random.normal(0, 0.1)
    noise_B = sigma_B * np.random.normal(0, 0.1)
    noise_G = sigma_G * np.random.normal(0, 0.1)
    noise_R1 = sigma_R1 * np.random.normal(0, 0.1)
    noise_R2 = sigma_R2 * np.random.normal(0, 0.1)
    
    # Seasonal variations
    r_P = seasonal_factors(t, base_r_P)
    alpha_I = seasonal_factors(t, base_alpha_I)
    beta_BI = seasonal_factors(t, base_beta_BI)
    beta_GI = seasonal_factors(t, base_beta_GI)
    r_R1 = seasonal_factors(t, base_r_R1)
    r_R2 = seasonal_factors(t, base_r_R2)
    beta_R1B = seasonal_factors(t, base_beta_R1B)
    beta_R1G = seasonal_factors(t, base_beta_R1G)
    beta_GR2 = seasonal_factors(t, base_beta_GR2)
    f_poll = seasonal_factors(t, base_poll_coeff)
    imm_rate_R1 = lambda_R1 * (t/(t+10)) if t > 0 else 0  # 渐进的迁入率
    imm_rate_R2 = lambda_R2 * (t/(t+10)) if t > 0 else 0

    # 改进功能响应函数的数值稳定性
    def safe_divide(x, y, default=0):
        try:
            with np.errstate(divide='raise', invalid='raise'):
                return x / y if abs(y) > 1e-10 else default
        except:
            return default
    
    # 使用安全除法重写捕食函数
    pred_BI = safe_divide(beta_BI * B * I, 1 + a_B*B + b_I*I) if (B > 0 and I > 0) else 0
    pred_GI = safe_divide(beta_GI * G * I, 1 + a_G*G + b_Ip*I) if (G > 0 and I > 0) else 0
    pred_R1 = safe_divide(beta_R1B * R1 * B + beta_R1G * R1 * G, 1 + a_R1*R1) if (R1 > 0) else 0
    pred_GR2 = safe_divide(beta_GR2 * G * R2, 1 + a_R2*R2) if (G > 0 and R2 > 0) else 0

    # 限制增长率的范围
    def bounded_rate(rate, min_rate=-1.0, max_rate=1.0):
        return max(min_rate, min(max_rate, rate))
    
    # 计算并限制变化率
    dPdt = bounded_rate(r_P * P * (1 - P/K_P) - alpha_PI * P * I - epsilon_PG * P * G - phi_P * f_poll * P + P * noise_P)
    dIdt = bounded_rate(alpha_I * I * P - pred_BI - pred_GI - mu_I * I - phi_I * f_poll * I + I * noise_I)
    dBdt = bounded_rate(kappa_B * pred_BI - delta_B * B - phi_B * f_poll * B + B * noise_B)
    dGdt = bounded_rate(kappa_G * pred_GI - delta_G * G - phi_G * f_poll * G + G * noise_G)
    dR1dt = bounded_rate(imm_rate_R1 + R1*(r_R1 + pred_R1) - phi_cR1*c*R1 - delta_R1*R1 + R1*noise_R1)
    dR2dt = bounded_rate(imm_rate_R2 + R2*(r_R2 - a_R2*R2 - pred_GR2) - phi_cR2*c*R2 - delta_R2*R2 + R2*noise_R2)

    return [dPdt, dIdt, dBdt, dGdt, dR1dt, dR2dt]

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
    'base_alpha_I': 0.05 ,      # 降低繁殖效应
    'mu_I': 0.05,          # 调整死亡率
    'phi_I': 0.001,       
    
    # 鸟类相关 - 显著改善生存条件
    'base_beta_BI': 0.15,      # 调整捕食效率
    'a_B': 0.05,          # 降低干扰系数
    'b_I': 0.03,          # 降低干扰系数
    'kappa_B': 0.3,       # 增加转化效率
    'delta_B': 0.03,       # 降低死亡率
    'phi_B': 0.0001,       # 降低农药影响
    
    # 蝙蝠相关 - 适当降低优势
    'base_beta_GI': 0.15,      # 调整捕食效率
    'a_G': 0.02,              # Decreased interference
    'b_Ip': 0.04,             # Decreased prey interference
    'kappa_G': 0.3,           # Increased conversion efficiency
    'delta_G': 0.03,          # 降低死亡率
    'phi_G': 0.0001, 

    'lambda_R1': 0.05,     # R1迁入率
    'lambda_R2': 0.1,      # R2迁入率
    'base_r_R1': 0.02,          # R1自身增长率
    'base_r_R2': 0.05,          # R2自身增长率
    'base_beta_R1B': 0.08,      # R1捕食鸟类效率
    'base_beta_R1G': 0.05,      # R1捕食蝙蝠效率
    'base_beta_GR2': 0.07,      # 蝙蝠捕食R2效率
    'a_R1': 0.02,          # R1捕食干扰
    'a_R2': 0.01,          # R2资源竞争
    'phi_cR1': 0.001,      # R1农药敏感度
    'phi_cR2': 0.002,      # R2农药敏感度
    'delta_R1': 0.03,      # R1自然死亡率
    'delta_R2': 0.04,      # R2自然死亡率
    'sigma_R1': 0.01,      # R1噪声强度
    'sigma_R2': 0.01,     # R2噪声强度

    'c': 0.01,            # 保持低农药使用
    'base_poll_coeff': 0.2,

    'sigma_P': 0.01,  # Crop noise intensity
    'sigma_I': 0.01,  # Insect noise intensity
    'sigma_B': 0.01,  # Bird noise intensity
    'sigma_G': 0.01,  # Bat noise intensity     
}
# 初始种群 [P0, I0, B0, G0, R1_0, R2_0]
Y0 = [20.0, 10.0, 5.0, 2.0, 1.0, 3.0]  # 增加鸟类初始数量，保持其他相对稳定
# 3. 数值求解
t_span = (0, 100)
sol = solve_ivp(fun=lambda t, y: eco_system(t, y, tuple(param_dict.values())),
                t_span=t_span, 
                y0=Y0, 
                t_eval=np.linspace(t_span[0], t_span[1], 500),
                method='LSODA',    # 使用更稳定的求解器
                rtol=1e-3,         # 放宽容差
                atol=1e-3,
                max_step=0.5,      # 增加最大步长
                first_step=0.01    # 指定初始步长
                )

if sol is not None and sol.success:
    # 绘图代码
    plt.figure(figsize=(12,7))
    plt.plot(sol.t, sol.y[0], label='Crops')
    plt.plot(sol.t, sol.y[1], label='Insects')
    plt.plot(sol.t, sol.y[2], label='Birds')
    plt.plot(sol.t, sol.y[3], label='Bats')
    plt.plot(sol.t, sol.y[4], '--', label='Raptors (R1)')
    plt.plot(sol.t, sol.y[5], '--', label='Herbivores (R2)')
    plt.legend()
    plt.title('Extended Ecosystem Dynamics with Returning Species')
    plt.show()
else:
    print("Failed to solve the system")
# 5. 灵敏度分析：对关键参数做扫描（举例：化学药剂强度 c 与 害虫捕食率 alpha_PI）
c_values = np.linspace(0, 1.0, 6)
alphaPI_values = np.linspace(0.005, 0.02, 6)
final_P = np.zeros((len(c_values), len(alphaPI_values)))
final_I = np.zeros((len(c_values), len(alphaPI_values)))

for i, cval in enumerate(c_values):
    for j, alphaPI in enumerate(alphaPI_values):
        # 修改参数
        local_params = dict(param_dict)
        local_params['c'] = cval
        local_params['alpha_PI'] = alphaPI
        
        # 修改求解器调用方式
        sol_ij = solve_ivp(
            fun=lambda t, y: eco_system(t, y, tuple(local_params.values())),
            t_span=t_span,
            y0=Y0,
            t_eval=np.array([t_span[1]]),  # 确保t_eval是一个数组
            method='BDF',
            rtol=1e-3,
            atol=1e-3
        )
        
        if sol_ij.success:
            final_P[i,j] = sol_ij.y[0,-1]  # 取第一个物种(P)的最后一个时间点
            final_I[i,j] = sol_ij.y[1,-1]  # 取第二个物种(I)的最后一个时间点
        else:
            print(f"Warning: Solution failed for c={cval}, alpha_PI={alphaPI}")
            final_P[i,j] = np.nan
            final_I[i,j] = np.nan

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