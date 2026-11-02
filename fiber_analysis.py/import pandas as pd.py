import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)
txt_path = os.path.join(output_dir, '1_纤维数据集分析结果.txt')

def save_txt(title, content):
    with open(txt_path, 'a', encoding='utf-8') as f:
        f.write(f"{'='*25} {title} {'='*25}\n")
        if isinstance(content, (pd.DataFrame, pd.Series)):
            f.write(content.to_string() + "\n\n")
        else:
            f.write(str(content) + "\n\n")

pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.width', 1200)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("1 fibers数据集.csv", encoding="gbk")

print("【全部列名清单】")
print(df.columns.tolist())
save_txt("全部列名清单", df.columns.tolist())

# (1)预览前5行
print("\n===== (1) 前5行数据 =====")
print(df.head())
save_txt("前5行数据", df.head())

# (2)缺失值、重复记录
print("\n===== (2) 缺失值统计 =====")
miss_info = df.isnull().sum()
print(miss_info)
save_txt("缺失值统计", miss_info)

print("\n===== 重复行数量 =====")
dup_num = df.duplicated().sum()
print(f"重复记录条数：{dup_num}")
save_txt("重复记录条数", f"重复记录条数：{dup_num}")

# (3)按【类别】统计、柱状图+饼图
group_col = "类别"
print(f"\n===== (3) {group_col} 数量统计 =====")
fiber_count = df[group_col].value_counts()
print(fiber_count)
save_txt(f"{group_col}数量统计", fiber_count)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13,5))
fiber_count.plot(kind="bar", ax=ax1)
ax1.set_title("各类别纤维数量柱状图")
ax1.set_ylabel("样本数量")
ax1.set_xlabel("纤维类别")

fiber_count.plot(kind="pie", autopct="%.1f%%", ax=ax2)
ax2.set_title("各类别纤维占比饼图")
ax2.set_ylabel("")
plt.tight_layout()
plt.savefig(os.path.join(output_dir,"纤维类别_柱状饼图.png"), dpi=300)
plt.show()

# (4)12项核心属性分组求均值
cols_12 = [
    "透气性","导湿性","耐用性","起球","缩水",
    "抗皱性","悬垂性","柔软度","保暖性","染色性",
    "防紫外线","抗静电"
]
group_mean = df.groupby(group_col)[cols_12].mean()
print("\n===== (4) 各类纤维12项属性均值 =====")
print(group_mean)
save_txt("各类纤维12项属性均值", group_mean)