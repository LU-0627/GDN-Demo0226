"""
Case Study：证明 Condition Gate 抑制误报且不漏报

本脚本在当前目录读取 `scoring_components.npz` 并绘制用于向评委展示的时序折线图。
运行前请确认 `scoring_components.npz` 位于此目录，或修改下面代码中的 `npz_path`。

提示：根据实际数据调整 `start` 与 `end` 索引以选取包含"正常工况大切换"与"真实攻击"的代表窗口。
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# 1. 读取打分组件（将 npz_path 指向你的 scoring_components.npz）
npz_path = r'F:\GDN\GDN-Demo0226\GDN\logs\wadi\20260302_202109\scoring_components.npz'
if not os.path.exists(npz_path):
    raise FileNotFoundError(f"{npz_path} 未在当前目录找到。请把文件放在此目录或修改 npz_path。")

data = np.load(npz_path)
print('Loaded keys:', list(data.keys()))
final_scores = data['final_scores']
residual_scores = data['residual_scores']
struct_scores = data['struct_scores']
gate_scores = data['gate_scores']
labels = data['labels']

# 2. 截取一段极具代表性的时序窗口（请根据实际数据调整 start/end）
start = 1000  # 替换为你的真实 start 索引
end = 3000    # 替换为你的真实 end 索引
# 保证索引合法
end = min(end, final_scores.shape[0])
start = max(0, min(start, end-1))

t = np.arange(start, end)
f_s = final_scores[start:end]
r_s = residual_scores[start:end]
s_s = struct_scores[start:end]
g_s = gate_scores[start:end]
l_s = labels[start:end]

# 3. 开始绘制 4 联子图
fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
plt.subplots_adjust(hspace=0.3)

# 定义异常高亮区间绘制函数
def highlight_anomalies(ax):
    in_anomaly = False
    start_idx = 0
    for i, label in enumerate(l_s):
        if label == 1 and not in_anomaly:
            in_anomaly = True
            start_idx = i
        elif label == 0 and in_anomaly:
            in_anomaly = False
            ax.axvspan(t[start_idx], t[i], color='red', alpha=0.2)
    if in_anomaly:
        ax.axvspan(t[start_idx], t[-1], color='red', alpha=0.2)

# --- Subplot 1: 基础残差 (Residual) ---
axs[0].plot(t, r_s, color='steelblue', linewidth=1.5)
axs[0].set_title('Component 1: Base Residual Scores (Prone to False Alarms)', fontsize=12, pad=10)
axs[0].set_ylabel('Residual L1')
highlight_anomalies(axs[0])

# --- Subplot 2: 工况一致性门控 (Condition Gate) ---
axs[1].plot(t, g_s, color='darkorange', linewidth=1.5)
axs[1].set_title('Component 2: Condition Consistency Gate (Suppressing Benign Switches)', fontsize=12, pad=10)
axs[1].set_ylabel('Gate Value (0~1)')
axs[1].set_ylim(-0.1, 1.1)
highlight_anomalies(axs[1])

# --- Subplot 3: 局部结构漂移 (Structural Drift) ---
axs[3].plot(t, s_s, color='purple', linewidth=1.5)
axs[3].set_title('Component 3: Top-q% Local Structural Drift (Capturing Attack Decoupling)', fontsize=12, pad=10)
axs[3].set_ylabel('Drift')
highlight_anomalies(axs[3])

# --- Subplot 4: 最终异常分数 (Final Score) ---
axs[2].plot(t, f_s, color='crimson', linewidth=2)
axs[2].set_title('Final Anomaly Score (s_t = Residual + Gate × Drift)', fontsize=12, pad=10)
axs[2].set_ylabel('Score')

# 画一条动态选取的阈值线（这里用简单的 99% 分位数做展示，实际可填入 best_threshold）
threshold = np.percentile(final_scores, 99)
axs[2].axhline(y=threshold, color='black', linestyle='--', label=f'Threshold ({threshold:.2f})')
axs[2].legend(loc='upper right')
highlight_anomalies(axs[2])

# 优化排版
for ax in axs:
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axs[-1].set_xlabel('Time Steps', fontsize=12)
# 保存到与 scoring_components.npz 相同的目录
save_dir = os.path.dirname(npz_path)
save_path = os.path.join(save_dir, 'case_study_suppression.pdf')
plt.savefig(save_path, bbox_inches='tight')
print(f'图表已保存到: {save_path}')
plt.show()
