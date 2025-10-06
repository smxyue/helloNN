from collections import Counter

# 1. 原始文本作为程序数据
raw_text = """\
Label=4, Predict=9 x
Label=5, Predict=8 x
Label=8, Predict=3 x
Label=9, Predict=4 x
Label=1, Predict=3 x
Label=8, Predict=9 x
Label=5, Predict=9 x
Label=6, Predict=5 x
Label=8, Predict=1 x
Label=4, Predict=6 x
Label=4, Predict=7 x
Label=7, Predict=2 x
Label=9, Predict=4 x
Label=9, Predict=5 x
Label=7, Predict=1 x
Label=5, Predict=3 x
Label=9, Predict=4 x
Label=7, Predict=9 x
Label=5, Predict=6 x
Label=5, Predict=3 x
Label=9, Predict=7 x
Label=8, Predict=4 x
Label=9, Predict=4 x
Label=8, Predict=7 x
Label=4, Predict=6 x
Label=9, Predict=8 x
Label=9, Predict=3 x
Label=9, Predict=7 x
Label=0, Predict=6 x
Label=9, Predict=4 x
Label=5, Predict=1 x
Label=2, Predict=3 x
Label=8, Predict=3 x
Label=9, Predict=4 x
Label=7, Predict=2 x
Label=5, Predict=3 x
Label=7, Predict=4 x
Label=2, Predict=0 x
Label=9, Predict=7 x
Label=6, Predict=1 x
Label=9, Predict=4 x
Label=4, Predict=7 x
Label=9, Predict=4 x
Label=3, Predict=2 x
Label=9, Predict=4 x
Label=2, Predict=0 x
Label=2, Predict=4 x
Label=5, Predict=3 x
Label=6, Predict=1 x
Label=9, Predict=4 x
Label=5, Predict=8 x
Label=8, Predict=5 x
Label=3, Predict=8 x
Label=3, Predict=2 x
Label=9, Predict=7 x
Label=3, Predict=5 x
Label=6, Predict=8 x
Label=9, Predict=7 x
Label=8, Predict=3 x
Label=5, Predict=9 x
Label=7, Predict=9 x
Label=4, Predict=9 x
Label=6, Predict=4 x
Label=4, Predict=8 x
Label=3, Predict=2 x
Label=8, Predict=5 x
Label=8, Predict=2 x
Label=8, Predict=9 x
Label=7, Predict=2 x
Label=4, Predict=6 x
Label=0, Predict=4 x
Label=9, Predict=4 x
Label=9, Predict=4 x
Label=9, Predict=4 x
Label=5, Predict=8 x
Label=4, Predict=6 x
Label=7, Predict=1 x
Label=9, Predict=4 x
Label=9, Predict=3 x
Label=9, Predict=7 x
Label=2, Predict=7 x
Label=9, Predict=7 x
Label=2, Predict=1 x
Label=9, Predict=4 x
Label=5, Predict=3 x
Label=5, Predict=3 x
Label=9, Predict=4 x
Label=3, Predict=2 x
Label=3, Predict=2 x
Label=8, Predict=3 x
Label=6, Predict=2 x
Label=3, Predict=0 x
Label=1, Predict=2 x
Label=2, Predict=3 x
Label=8, Predict=7 x
Label=3, Predict=5 x
Label=1, Predict=3 x
Label=9, Predict=4 x
Label=9, Predict=0 x
Label=0, Predict=8 x
Label=8, Predict=4 x
Label=5, Predict=4 x
Label=8, Predict=9 x
Label=8, Predict=2 x
"""

# 2. 提取 label 数字
labels = []
for line in raw_text.splitlines():
    if line.startswith("Label="):
        # 取 "Label=" 后面到第一个非数字前的部分
        digit_part = line.split('=')[1].split(',')[0]
        labels.append(int(digit_part))

# 3. 统计
counter = Counter(labels)

# 4. 按 label 升序输出
for label in sorted(counter):
    print(f"{label} {counter[label]}")

print(f"Total error: {counter.total()}")

# 5. 图形化输出
import matplotlib.pyplot as plt

labels_sorted = sorted(counter)
counts = [counter[l] for l in labels_sorted]

plt.figure(figsize=(6, 4))
bars = plt.bar(labels_sorted, counts, color='steelblue')
plt.xticks(labels_sorted)
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Error lable & count')

# 在柱顶部标数字
for b in bars:
    h = b.get_height()
    plt.text(b.get_x() + b.get_width()/2, h+0.3,
             str(int(h)), ha='center', va='bottom')

plt.tight_layout()
plt.show()