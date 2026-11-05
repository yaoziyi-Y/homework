import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("3 Digital_Twin_Functional_Garments_Dataset.csv", encoding="utf-8-sig")

print("===== 数据集基础信息 =====")
print(f"数据集形状（行,列）：{df.shape}")
print(f"总样本量：{df.shape[0]}")
print(f"总特征数：{df.shape[1]}")
print(f"缺失值总数：{df.isnull().sum().sum()}")
print(f"重复样本数量：{df.duplicated().sum()}")

# 看刚才打印的列顺序：
# 0织物类型，1织物组织类型，2织物厚度(mm)，3织物面密度(g/m²)，4纤维直径，5孔隙率，6透气性，7吸湿性，8导热系数，9初始拉伸强度
# 目标6个指标：索引 [2,3,6,7,5,9]
idx_list = [2,3,6,7,5,9]
df_sub = df.iloc[:, idx_list].copy()
# 手动给这6列重命名，方便画图和输出
df_sub.columns = ["织物厚度 (mm)", "织物面密度 (g/m²)", "透气性 (cm3/s)", "吸湿性 (%)", "孔隙率_百分比 (%)", "初始拉伸强度 (MPa)"]

# ========== 2. 高级统计量：均值、标准差、偏度、峰度 ==========
stat_result = pd.DataFrame()
stat_result["均值"] = df_sub.mean()
stat_result["标准差"] = df_sub.std()
stat_result["偏度"] = df_sub.skew()
stat_result["峰度"] = df_sub.kurt()
print("\n===== 6项指标统计诊断表 =====")
print(stat_result)

# 保存txt
if not os.path.exists("output"):
    os.makedirs("output")
with open("output/hw3_stat_result.txt", "w", encoding="utf-8") as f:
    f.write("6项织物指标统计诊断(均值、标准差、偏度、峰度)\n")
    f.write(stat_result.to_string())

# ========== 3. 直方图+KDE ==========
fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=120)
axes = axes.flatten()
cols = df_sub.columns.tolist()

for i, col in enumerate(cols):
    ax = axes[i]
    sns.histplot(df_sub[col], kde=True, ax=ax, color="#c82423", bins=30)
    ax.set_title(f"{col} 分布直方图+KDE密度曲线", fontsize=10)
    ax.set_xlabel(col)
    ax.set_ylabel("频数")
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("output/hw3_多特征KDE分布图.png", bbox_inches="tight")
plt.show()