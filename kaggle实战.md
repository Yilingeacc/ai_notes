# California House Prices

> https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview

# Elo Merchant Category Recommendation

> https://www.kaggle.com/competitions/elo-merchant-category-recommendation

## 查看基本信息

```python
train.shape, test.shape
train.info()  # 列信息
```

## 数据质量分析

1. ID列是否有重复（数据正确性校验）
2. 查看缺失值情况
3. 检查label异常值（直方图）

```python
# 1. 数据正确性校验, 检查id是否重复
train['id'].nunique() == train.shape[0]
test['id'].nunique() == test.shape[0]
# 2. 查看缺失值情况
train.isnull().sum()
test.isnull().sum()
# 3. 检查label异常值
statistics = train['target'].describe()
import seaborn as sns
sns.histplot(train['target'], kde=True)
```

## 规律一致性分析

训练集和测试集特征数据的分布是否一致，确定是否采样自同一个总体。（满足i.i.d.）

### 单变量分析

```python
features = ['feature_1', 'feature_2', 'feature_3']

train_count = train.shape[0]
test_count = test.shape[0]

# 单变量分析
for feature in features:
    plt.xlabel(feature)
    plt.ylabel('ratio')
    (train[feature].value_counts().sort_index() / train_count).plot()
    (test[feature].value_counts().sort_index() / test).plot()
    plt.legend(['train', 'test'])
```

### 多变量分析

```python
def combine_feature(df):
    cols = df.columns
    feature1 = df[cols[0]].astype(str).values.tolist()
    feature2 = df[cols[1]].astype(str).values.tolist()
    return pd.Series([f'{feature1[i]}&{feature2[i]}' for i in range(df.shape[0])])

n = len(features)
for f1 in range(n-1):
    for f2 in range(f1 + 1, n):
        cols = [features[f1], features[f2]]
        train_dis = combine_feature(train[cols]).value_counts().sort_index() / train_count
        test_dis = combine_feature(test[cols]).value_counts().sort_index() / test_count
        index_dis = pd.Series(train_dis.index.tolist() + test_dis.index.tolist()).drop_duplicates()
        (index_dis.map(train_dis).fillna(0).fillna(0)).plot()
        (index_dis.map(test_dis).fillna(0).fillna(0)).plot()
        plt.xlabel('&'.join(cols))
        plt.ylabel('ratio')
        plt.legend(['train', 'test'])
        plt.show()
```

- 如果分布非常一致，则说明所有特征均来自统一整体，训练集和测试集规律拥有较高一致性，模型效果上限较高，建模过程中应该更加依靠特征工程方法和模型建模技巧提高最终预测效果
- 如果分布不太一致，则说明训练集和测试集规律不太一致，此时模型预测效果上限会受此影响而被限制，并且模型大概率容易过拟合，在实际建模过程中可以多考虑使用交叉验证等方式防止过拟合，并且需要注重除了通用特征工程和建模方法外的trick使用



