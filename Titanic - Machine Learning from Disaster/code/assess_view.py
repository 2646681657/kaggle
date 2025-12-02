# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os


for dirname, _, filenames in os.walk('..'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Any results you write to the current directory are saved as output.


train_data = pd.read_csv(r'/Titanic - Machine Learning from Disaster/train.csv')
print(train_data.head())


test_data = pd.read_csv(r'/Titanic - Machine Learning from Disaster/train.csv')
print(test_data.head())

women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)
print("% of women who survived:", rate_women)

men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)
print("% of men who survived:", rate_men)

# 新增：可视化对比submission和gender_submission的吻合度
print("\n=== 对比submission和gender_submission的吻合度 ===")

# 加载两个提交文件
submission = pd.read_csv(r'/Titanic - Machine Learning from Disaster/submission.csv')
gender_submission = pd.read_csv(r'/Titanic - Machine Learning from Disaster/gender_submission.csv')

print("Submission文件形状:", submission.shape)
print("Gender submission文件形状:", gender_submission.shape)

# 检查数据是否对齐
if submission['PassengerId'].equals(gender_submission['PassengerId']):
    print("✓ PassengerId完全对齐")
else:
    print("✗ PassengerId不匹配")

# 比较预测结果
comparison = pd.merge(submission, gender_submission, on='PassengerId', suffixes=('_model', '_gender'))
comparison['match'] = comparison['Survived_model'] == comparison['Survived_gender']
match_rate = comparison['match'].mean()

print(f"预测结果吻合度: {match_rate:.2%}")
print(f"匹配的预测数量: {comparison['match'].sum()}/{len(comparison)}")
print(f"不匹配的预测数量: {(~comparison['match']).sum()}")

# 创建可视化图表
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. 吻合度饼图
axes[0, 0].pie([match_rate, 1-match_rate], 
               labels=[f'匹配\n{match_rate:.1%}', f'不匹配\n{(1-match_rate):.1%}'],
               autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
axes[0, 0].set_title('预测结果吻合度分布')

# 2. 生存预测对比柱状图
survival_comparison = pd.DataFrame({
    'Model': submission['Survived'].value_counts(),
    'Gender Baseline': gender_submission['Survived'].value_counts()
}).fillna(0)

survival_comparison.plot(kind='bar', ax=axes[0, 1], color=['skyblue', 'pink'])
axes[0, 1].set_title('生存预测数量对比')
axes[0, 1].set_xlabel('预测结果 (0=死亡, 1=生存)')
axes[0, 1].set_ylabel('数量')
axes[0, 1].legend()
axes[0, 1].tick_params(axis='x', rotation=0)

# 3. 不匹配情况分析
mismatches = comparison[~comparison['match']]
if len(mismatches) > 0:
    mismatch_analysis = mismatches.groupby(['Survived_model', 'Survived_gender']).size().reset_index(name='count')
    mismatch_labels = [f"模型:{row['Survived_model']}, 性别基线:{row['Survived_gender']}" 
                      for _, row in mismatch_analysis.iterrows()]
    axes[1, 0].pie(mismatch_analysis['count'], labels=mismatch_labels, autopct='%1.1f%%')
    axes[1, 0].set_title('不匹配情况详细分析')
else:
    axes[1, 0].text(0.5, 0.5, '所有预测都匹配!', ha='center', va='center', fontsize=14)
    axes[1, 0].set_title('不匹配情况分析')

# 4. PassengerId分布对比
axes[1, 1].scatter(comparison['PassengerId'], comparison['Survived_model'], 
                    alpha=0.6, label='Model Prediction', color='blue', s=20)
axes[1, 1].scatter(comparison['PassengerId'], comparison['Survived_gender'], 
                    alpha=0.6, label='Gender Baseline', color='red', s=20)
axes[1, 1].set_title('预测结果沿PassengerId分布')
axes[1, 1].set_xlabel('PassengerId')
axes[1, 1].set_ylabel('Survived (0=死亡, 1=生存)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('submission_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 详细统计
print("\n=== 详细统计对比 ===")
print("模型预测生存率:", submission['Survived'].mean())
print("性别基线生存率:", gender_submission['Survived'].mean())

# 生存预测差异
model_survivors = submission['Survived'].sum()
gender_survivors = gender_submission['Survived'].sum()
print(f"模型预测幸存者: {model_survivors}人")
print(f"性别基线预测幸存者: {gender_survivors}人")
print(f"差异: {model_survivors - gender_survivors}人")
