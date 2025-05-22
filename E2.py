import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp

# -----------------------
# 1. 定义模型微分方程
# -----------------------
def seasonal_factors(t, base_value, amplitude=0.3, period=365):
    """Generate seasonal variation using sine function"""
    return base_value * (1 + amplitude * np.sin(2 * np.pi * t / period))

def eco_system_decisions(t, Y, params):
    """Modified system with seasonality and noise"""
    # 确保所有状态变量非负
    Y = np.maximum(Y, np.zeros_like(Y))
    W, P, I, B, G, X = Y
    
    # Update parameter unpacking to match exactly with params_dict order
    (r_W, K_W, alpha_WI, phi_W,
     r_P, K_P, alpha_PI, phi_P,
     beta_IW, beta_IP, mu_I, phi_I,
     beta_B, a_B, b_I, kappa_B, delta_B, phi_B,
     beta_G, a_G, kappa_G, delta_G, phi_G,
     beta_X, a_X, kappa_X, delta_X, phi_X,
     epsilon_PB, epsilon_PG,
     u_func, c_func,
     sigma_W, sigma_P, sigma_I, sigma_B, sigma_G, sigma_X) = params
    
   
    
    # Seasonal variations
    r_W_t = seasonal_factors(t, r_W)
    r_P_t = seasonal_factors(t, r_P)
    alpha_WI_t = seasonal_factors(t, alpha_WI)  # Seasonal herbivory
    beta_IP_t = seasonal_factors(t, beta_IP)    # Seasonal crop consumption
    beta_B_t = seasonal_factors(t, beta_B)      # Seasonal bird predation
    beta_G_t = seasonal_factors(t, beta_G)      # Seasonal bat predation
    beta_X_t = seasonal_factors(t, beta_X)      # Seasonal predator X predation
    poll_effect = seasonal_factors(t, 1.0)      # Pollination seasonality
    
    # White noise terms
    noise_W = sigma_W * np.random.normal(0, 1)
    noise_P = sigma_P * np.random.normal(0, 1)
    noise_I = sigma_I * np.random.normal(0, 1)
    noise_B = sigma_B * np.random.normal(0, 1)
    noise_G = sigma_G * np.random.normal(0, 1)
    noise_X = sigma_X * np.random.normal(0, 1)
    
    # Chemical inputs
    u = u_func(t)
    c = c_func(t)
    
    # 添加小量以防止除零
    eps = 1e-10
    
    # 修改捕食项计算方式，移除未定义的b_I
    pred_BI = beta_B_t * B * I / (1 + a_B * B + a_B * I + eps)
    pred_GI = beta_G_t * G * I / (1 + a_G * G + a_G * I + eps)
    pred_XI = beta_X_t * X * I / (1 + a_X * X + a_X * I + eps)
    
    # 使用np.clip限制增长率在合理范围内
    dWdt = np.clip(r_W_t * W * (1 - W/K_W) - alpha_WI_t * I * W - phi_W * u * W + W * noise_W, -W, 1e3)
    
    dPdt = np.clip(r_P_t * P * (1 - P/K_P) - alpha_PI * I * P - phi_P * c * P + \
           poll_effect * (epsilon_PB * P * B + epsilon_PG * P * G) + P * noise_P, -P, 1e3)
    
    dIdt = np.clip(I * (beta_IP_t * P + beta_IW * W - mu_I) - phi_I * c * I - \
           pred_BI - pred_GI - pred_XI + I * noise_I, -I, 1e3)
    
    dBdt = np.clip(B * (kappa_B * pred_BI - delta_B) - phi_B * c * B + B * noise_B, -B, 1e3)
    
    dGdt = np.clip(G * (kappa_G * pred_GI - delta_G) - phi_G * c * G + G * noise_G, -G, 1e3)
    
    dXdt = np.clip(X * (kappa_X * pred_XI - delta_X) - phi_X * c * X + X * noise_X, -X, 1e3)
    
    return [dWdt, dPdt, dIdt, dBdt, dGdt, dXdt]


# -----------------------
# 2. 参数与初始条件
# -----------------------
def const_func(value):
    """返回一个对时间t不变的常函数"""
    return lambda t: value

# 示例：我们先设置默认的 u, c 为常量
u_default = 1.0  # 较高除草剂
c_default = 0.5  # 中等杀虫剂

params_dict = {
    # Growth parameters
    'r_W': 0.8,
    'K_W': 50.0,
    'r_P': 1.0,
    'K_P': 80.0,
    
    # Interaction parameters
    'alpha_WI': 0.002,
    'phi_W': 0.1,
    'alpha_PI': 0.01,
    'phi_P': 0.001,
    
    # Insect parameters
    'beta_IW': 0.05,
    'beta_IP': 0.05,
    'mu_I': 0.1,
    'phi_I': 0.2,
    
    # Bird parameters
    'beta_B': 0.3,
    'a_B': 0.1,
    'b_I': 0.05,
    'kappa_B': 0.2,
    'delta_B': 0.05,
    'phi_B': 0.01,
    
    # Bat parameters
    'beta_G': 0.3,
    'a_G': 0.1,
    'kappa_G': 0.25,
    'delta_G': 0.05,
    'phi_G': 0.01,
    
    # Species X parameters
    'beta_X': 0.3,
    'a_X': 0.1,
    'kappa_X': 0.22,
    'delta_X': 0.05,
    'phi_X': 0.01,
    
    # Pollination parameters
    'epsilon_PB': 0.001,
    'epsilon_PG': 0.001,
    
    # Chemical control functions
    'u_func': const_func(1.0),
    'c_func': const_func(0.5),
    
    # Noise parameters
    'sigma_W': 0.05,
    'sigma_P': 0.05,
    'sigma_I': 0.05,
    'sigma_B': 0.03,
    'sigma_G': 0.03,
    'sigma_X': 0.03
}

# 初始条件: W, P, I, B, G, X
Y0 = [10.0, 20.0, 8.0, 5.0, 5.0, 1.0]  # More balanced initial populations


# -----------------------
# 3. 数值求解与绘图
# -----------------------
def run_simulation(params, Y0):
    param_tuple = tuple(params.values())
    t_span = (0, 365)  # 改为365天以观察完整的季节性变化
    t_eval = np.linspace(0, 365, 3650)  # 增加采样点
    
    # 添加事件检测以防止负值
    def event_negative(t, y):
        return min(y)
    event_negative.terminal = True
    event_negative.direction = -1
    
    sol = solve_ivp(
        fun=lambda t, y: eco_system_decisions(t, y, param_tuple),
        t_span=t_span,
        y0=Y0,
        t_eval=t_eval,
        method='LSODA',
        rtol=1e-6,
        atol=1e-6,
        max_step=0.1
    )
    
    return sol.t, sol.y


# 3.1 先做一个场景模拟： 高除草剂(u=1.0) vs. 无除草剂(u=0)
def scenario_comparison():
    fig, axs = plt.subplots(2,1, figsize=(10,8), sharex=True)
    
    # 场景A: 高除草剂(u=1), 中等杀虫剂(c=0.5)
    paramsA = dict(params_dict)
    paramsA['u_func'] = const_func(1.0)
    paramsA['c_func'] = const_func(0.5)
    tA, solA = run_simulation(paramsA, Y0)
    
    # 场景B: 无除草剂(u=0), 同样杀虫剂(c=0.5)
    paramsB = dict(params_dict)
    paramsB['u_func'] = const_func(0.0)
    paramsB['c_func'] = const_func(0.5)
    tB, solB = run_simulation(paramsB, Y0)
    
    # solA, solB分别是6行(W,P,I,B,G,X)
    labels = ['W (weeds)', 'P (crops)', 'I (insects)', 'B (birds)', 'G (bats)', 'X (species X)']
    
    # 上图: 场景A
    for i in range(6):
        axs[0].plot(tA, solA[i], label=labels[i])
    axs[0].set_title('Scenario A: High herbicide (u=1.0), medium pesticide (c=0.5)')
    axs[0].legend(loc='best')
    axs[0].grid(True)
    
    # 下图: 场景B
    for i in range(6):
        axs[1].plot(tB, solB[i], label=labels[i])
    axs[1].set_title('Scenario B: No herbicide (u=0.0), medium pesticide (c=0.5)')
    axs[1].legend(loc='best')
    axs[1].set_xlabel('Time')
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.show()


# 3.2 灵敏度分析: 扫描 [u, c] 两个方向, 观察最终(或平均)作物P和害虫I
def sensitivity_analysis():
    u_values = np.linspace(0, 1.0, 5)   # 5个除草剂强度取值
    c_values = np.linspace(0, 1.0, 5)   # 5个杀虫剂强度取值
    
    final_P = np.zeros((len(u_values), len(c_values)))
    final_I = np.zeros((len(u_values), len(c_values)))
    
    # 扫描
    for i, uv in enumerate(u_values):
        for j, cv in enumerate(c_values):
            local_params = dict(params_dict)
            local_params['u_func'] = const_func(uv)
            local_params['c_func'] = const_func(cv)
            
            t_sim, sol_sim = run_simulation(local_params, Y0)
            # sol_sim是 shape (6, len(t_sim)), 取最后一列
            W_end, P_end, I_end, B_end, G_end, X_end = sol_sim[:, -1]
            final_P[i,j] = P_end
            final_I[i,j] = I_end
    
    # 绘制2D热力图
    fig, ax = plt.subplots(1,2, figsize=(12,5))
    U_grid, C_grid = np.meshgrid(c_values, u_values)  # 注意 x= c, y= u
    
    # Final P
    cp1 = ax[0].contourf(C_grid, U_grid, final_P, 20, cmap='viridis')
    fig.colorbar(cp1, ax=ax[0])
    ax[0].set_xlabel('Pesticide c')
    ax[0].set_ylabel('Herbicide u')
    ax[0].set_title('Final P (crops) distribution')
    
    # Final I
    cp2 = ax[1].contourf(C_grid, U_grid, final_I, 20, cmap='plasma')
    fig.colorbar(cp2, ax=ax[1])
    ax[1].set_xlabel('Pesticide c')
    ax[1].set_ylabel('Herbicide u')
    ax[1].set_title('Final I (insects) distribution')
    
    plt.tight_layout()
    plt.show()
    
    # 也可作3D图, 以 final P 随 (u, c) 变化为例
    fig = plt.figure(figsize=(8,6))
    ax3d = fig.add_subplot(111, projection='3d')
    
    ax3d.plot_surface(C_grid, U_grid, final_P, cmap='coolwarm', edgecolor='none')
    ax3d.set_xlabel('Pesticide c')
    ax3d.set_ylabel('Herbicide u')
    ax3d.set_zlabel('Final P')
    ax3d.set_title('3D Surface of Final Crop Population')
    plt.show()

# -----------------------
# 4. 主程序演示
# -----------------------
if __name__ == "__main__":
    # 场景对比
    scenario_comparison()
    
    # 灵敏度分析
    sensitivity_analysis()