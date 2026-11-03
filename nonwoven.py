import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("2 非织造织物.csv", encoding="gbk")

print("====全部列名====")
print(df.columns.tolist())

print("\n===== 缺失值统计 =====")
print(df.isnull().sum())
print("\n===== 重复行数量 =====")
print("重复样本数：", df.duplicated().sum())

# 用序号选列：1水刺压力、2喷孔直径、5面密度、6厚度、9机器方向断裂力
df_sub = df.iloc[:, [1,2,5,6,9]].copy()
df_sub.columns = ["水刺压力", "喷孔直径", "面密度", "厚度", "MD断裂力"]

# 描述统计
print("\n===== 5项指标描述统计 =====")
desc_result = df_sub.describe()
print(desc_result)
with open("描述统计结果.txt","w",encoding="utf-8") as f:
    f.write(desc_result.to_string())

# 相关性热力图
corr = df_sub.corr()
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("指标相关性热力图：水刺压力、喷孔直径、面密度、厚度、机器方向断裂力")
plt.tight_layout()
plt.savefig("相关性热力图.png", dpi=300)
plt.show()

print("\n✅ 运行完成！生成：描述统计结果.txt、相关性热力图.png")