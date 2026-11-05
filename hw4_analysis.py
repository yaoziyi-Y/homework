import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# ---------------------- 1. 路径设置 ----------------------
BASE_DIR = os.getcwd()
output_dir = os.path.join(BASE_DIR, "output")
os.makedirs(output_dir, exist_ok=True)
txt_path = os.path.join(output_dir, "hw3_清洗与交叉分析结果.txt")
heatmap_path = os.path.join(output_dir, "hw3_双因素均值热力图.png")
bar_path = os.path.join(output_dir, "hw3_分组条形图.png")

# 读取数据集
csv_path = os.path.join(BASE_DIR, "3 Digital_Twin_Functional_Garments_Dataset.csv")
df = pd.read_csv(csv_path, encoding="utf-8-sig")

# 打印列和序号，方便核对
print("=== 列序号：列名 ===")
for idx, name in enumerate(df.columns):
    print(idx, ":", repr(name))

# ========== 用索引选取！根据刚才打印的序号 ==========
# 从你之前输出：
# 0:织物类型，1:织物组织类型，2:织物厚度，3:织物面密度，4:纤维直径，5:透气性，6:吸湿率，7:导热系数，8:初始拉伸强度
idx_numeric = [2,3,4,5,6,7]   # 6个指标：厚度、面密度、纤维直径、透气、吸湿、导热
idx_type = 0
idx_struct = 1
idx_strength = 8

numeric_cols = df.columns[idx_numeric].tolist()
col_type = df.columns[idx_type]
col_structure = df.columns[idx_struct]
col_strength = df.columns[idx_strength]

# ---------------------- 2. 定义异常值检测函数 ----------------------
def iqr_clean(df_in, cols):
    df_iqr = df_in.copy()
    mask_all = pd.Series([True]*len(df_iqr))
    for c in cols:
        Q1 = df_iqr[c].quantile(0.25)
        Q3 = df_iqr[c].quantile(0.75)
        iqr = Q3 - Q1
        low = Q1 - 1.5 * iqr
        high = Q3 + 1.5 * iqr
        mask = (df_iqr[c] >= low) & (df_iqr[c] <= high)
        mask_all = mask_all & mask
    df_clean = df_iqr[mask_all]
    return df_clean, mask_all.sum()

def sigma3_clean(df_in, cols):
    df_sigma = df_in.copy()
    mask_all = pd.Series([True]*len(df_sigma))
    for c in cols:
        mu = df_sigma[c].mean()
        std = df_sigma[c].std()
        low = mu - 3 * std
        high = mu + 3 * std
        mask = (df_sigma[c] >= low) & (df_sigma[c] <= high)
        mask_all = mask_all & mask
    df_clean = df_sigma[mask_all]
    return df_clean, mask_all.sum()

# ---------------------- 3. 执行清洗、计算保留率 ----------------------
n_origin = len(df)
df_iqr, n_iqr = iqr_clean(df, numeric_cols)
df_sigma, n_sigma = sigma3_clean(df, numeric_cols)
rate_iqr = n_iqr / n_origin * 100
rate_sigma = n_sigma / n_origin * 100

# ---------------------- 4. 双因素交叉分析（IQR清洗后的数据） ----------------------
df_used = df_iqr
stat_type = df_used.groupby(col_type)[col_strength].agg(["mean","std","count"])
stat_struct = df_used.groupby(col_structure)[col_strength].agg(["mean","std","count"])
pivot_mean = pd.pivot_table(
    df_used,
    values=col_strength,
    index=col_type,
    columns=col_structure,
    aggfunc="mean"
)

# ---------------------- 5. 输出文本结果，保存txt ----------------------
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("===== 第三次作业：异常值清洗与双因素交叉分析 =====\n")
    f.write(f"原始样本总数：{n_origin}\n")
    f.write(f"IQR法清洗后样本量:{n_iqr}，样本保留率：{rate_iqr:.2f} %\n")
    f.write(f"3σ法清洗后样本量:{n_sigma}，样本保留率：{rate_sigma:.2f} %\n\n")
    f.write("----- 单维度统计：织物类型 -----\n")
    f.write(stat_type.to_string())
    f.write("\n\n----- 单维度统计：织物组织类型 -----\n")
    f.write(stat_struct.to_string())
    f.write("\n\n----- 双因素交叉透视表（初始拉伸强度均值） -----\n")
    f.write(pivot_mean.to_string())

# ---------------------- 6. 绘图1：分组条形图 ----------------------
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
fig1, ax1 = plt.subplots(figsize=(10,6))
pivot_mean.plot(kind="bar", ax=ax1)
ax1.set_title("织物类型-组织类型 初始拉伸强度均值分组条形图")
ax1.set_ylabel("初始拉伸强度(MPa)")
ax1.set_xlabel("织物类型")
ax1.legend(title="织物组织类型")
plt.tight_layout()
plt.savefig(bar_path, dpi=300)
plt.close()

# ---------------------- 7. 绘图2：交叉均值热力图 ----------------------
fig2, ax2 = plt.subplots(figsize=(9,6))
sns.heatmap(pivot_mean, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax2)
ax2.set_title("双因素交叉均值热力图（初始拉伸强度）")
plt.tight_layout()
plt.savefig(heatmap_path, dpi=300)
plt.close()

print("✅ 运行完成！")
print(f"清洗结果txt:{txt_path}")
print(f"分组条形图：{bar_path}")
print(f"均值热力图：{heatmap_path}")