import pandas as pd
import matplotlib.pyplot as plt

# 解决中文乱码
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ========== 数据集1：纤维基础属性与分类统计 ==========
# (1) 读取csv，预览前5行
df = pd.read_csv("1 fibers数据集.csv")
print("===== 数据前5行 =====")
print(df.head())

# (2) 检查缺失值和重复记录
print("\n===== 缺失值统计 =====")
print(df.isnull().sum())
print("\n===== 重复记录数量 =====")
print("重复行数：", df.duplicated().sum())

# (3) 按纤维大类统计数量，绘制柱状图、饼图
cate_count = df["纤维大类"].value_counts()
print("\n===== 各纤维大类样本数量 =====")
print(cate_count)

# 柱状图
plt.figure(figsize=(8, 4))
cate_count.plot(kind="bar", color="#4472C4")
plt.title("纤维类别数量柱状图")
plt.xlabel("纤维大类")
plt.ylabel("样本数量")
plt.tight_layout()
plt.show()

# 饼图
plt.figure(figsize=(6, 6))
cate_count.plot(kind="pie", autopct="%.1f%%")
plt.title("纤维类别占比饼图")
plt.ylabel("")
plt.tight_layout()
plt.show()

# (4) 12项属性分组求均值
attr_cols = ["透气性","导湿性","耐用性","起球","缩水","抗皱性",
             "悬垂性","柔软度","保暖性","染色性","防紫外线","抗静电"]
group_avg = df.groupby("纤维大类")[attr_cols].mean()
print("\n===== 不同纤维12项属性均值 =====")
print(group_avg)