
# coding: utf-8

# ## 探索电影数据集
# 
# 在这个项目中，你将尝试使用所学的知识，使用 `NumPy`、`Pandas`、`matplotlib`、`seaborn` 库中的函数，来对电影数据集进行探索。
# 
# 下载数据集：
# [TMDb电影数据](https://s3.cn-north-1.amazonaws.com.cn/static-documents/nd101/explore+dataset/tmdb-movies.csv)
# 

# 
# 数据集各列名称的含义：
# <table>
# <thead><tr><th>列名称</th><th>id</th><th>imdb_id</th><th>popularity</th><th>budget</th><th>revenue</th><th>original_title</th><th>cast</th><th>homepage</th><th>director</th><th>tagline</th><th>keywords</th><th>overview</th><th>runtime</th><th>genres</th><th>production_companies</th><th>release_date</th><th>vote_count</th><th>vote_average</th><th>release_year</th><th>budget_adj</th><th>revenue_adj</th></tr></thead><tbody>
#  <tr><td>含义</td><td>编号</td><td>IMDB 编号</td><td>知名度</td><td>预算</td><td>票房</td><td>名称</td><td>主演</td><td>网站</td><td>导演</td><td>宣传词</td><td>关键词</td><td>简介</td><td>时常</td><td>类别</td><td>发行公司</td><td>发行日期</td><td>投票总数</td><td>投票均值</td><td>发行年份</td><td>预算（调整后）</td><td>票房（调整后）</td></tr>
# </tbody></table>
# 

# In[1]:


**请注意，你需要提交该报告导出的 `.html`、`.ipynb` 以及 `.py` 文件。**


# 
# 
# ---
# 
# ---
# 
# ## 第一节 数据的导入与处理
# 
# 在这一部分，你需要编写代码，使用 Pandas 读取数据，并进行预处理。

# 
# **任务1.1：** 导入库以及数据
# 
# 1. 载入需要的库 `NumPy`、`Pandas`、`matplotlib`、`seaborn`。
# 2. 利用 `Pandas` 库，读取 `tmdb-movies.csv` 中的数据，保存为 `movie_data`。
# 
# 提示：记得使用 notebook 中的魔法指令 `%matplotlib inline`，否则会导致你接下来无法打印出图像。

# In[276]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')

movie_data = pd.read_csv('./tmdb-movies.csv')


# ---
# 
# **任务1.2: ** 了解数据
# 
# 你会接触到各种各样的数据表，因此在读取之后，我们有必要通过一些简单的方法，来了解我们数据表是什么样子的。
# 
# 1. 获取数据表的行列，并打印。
# 2. 使用 `.head()`、`.tail()`、`.sample()` 方法，观察、了解数据表的情况。
# 3. 使用 `.dtypes` 属性，来查看各列数据的数据类型。
# 4. 使用 `isnull()` 配合 `.any()` 等方法，来查看各列是否存在空值。
# 5. 使用 `.describe()` 方法，看看数据表中数值型的数据是怎么分布的。
# 
# 

# In[277]:


# 改进分别显示
# 1.获取数据表的行列，并打印。(第一行的 original_title 列)
print(movie_data['original_title'][0])


# In[278]:


# 增加display 显示
# 2. 使用 `.head()`、`.tail()`、`.sample()` 方法，观察、了解数据表的情况。
display(movie_data.head())
display(movie_data.tail())
display(movie_data.sample())


# In[279]:


# 3. 使用 `.dtypes` 属性，来查看各列数据的数据类型
movie_data.dtypes


# In[280]:


# 4. 使用 `isnull()` 配合 `.any()` 等方法，来查看各列是否存在空值。
movie_data.isnull().any()


# In[281]:


#5. 使用 `.describe()` 方法，看看数据表中数值型的数据是怎么分布的。
movie_data.describe()


# ---
# 
# **任务1.3: ** 清理数据
# 
# 在真实的工作场景中，数据处理往往是最为费时费力的环节。但是幸运的是，我们提供给大家的 tmdb 数据集非常的「干净」，不需要大家做特别多的数据清洗以及处理工作。在这一步中，你的核心的工作主要是对数据表中的空值进行处理。你可以使用 `.fillna()` 来填补空值，当然也可以使用 `.dropna()` 来丢弃数据表中包含空值的某些行或者列。
# 
# 任务：使用适当的方法来清理空值，并将得到的数据保存。

# In[282]:


# 查看各个列空值情况
display(movie_data.isnull().sum())
# 查看各列数据类型
display(movie_data1.dtypes)
# 处理在下一个cell


# In[283]:


# 清理空值步骤
# 分析 可以看见homepage（网站），tagline（宣传词），keywords（）这几列在后面的数据分析处理用不上，而且空值比较大
# 可将这几列丢弃／忽略
movie_data1 = movie_data.drop(['homepage','tagline','keywords'], axis=1)
# 查看各个列空值情况
display(movie_data1.isnull().sum())
# 分析发现现在出现空值的列数据类型是字符型
# 所以将字符串列imdb_id，production_companies,cast,overview后面分析用不上，空值可以用'Unknown'的字眼来填充 
movie_data1['imdb_id'] = movie_data1['imdb_id'].fillna('Unknown')
movie_data1['production_companies'] = movie_data1['production_companies'].fillna('Unknown')
movie_data1['cast'] = movie_data1[ 'cast'].fillna('Unknown')
movie_data1['overview'] = movie_data1['overview'].fillna('Unknown')
# genres 电影类型后面分析用到，但是填充Unknown 对类型统计分析没有影响，所以也填充
movie_data1['genres'] = movie_data1['genres'].fillna('Unknown')
# 查看处理后的空值情况
display(movie_data1.isnull().sum())
# director 导演这一样这里不能填充Unknown，否则影响统计数量的排行榜，这里可以考虑去掉空值对应的行
movie_data1 = movie_data1.dropna(axis=0) # 实际上到导演分类统计任务中处理更合理
# 查看处理后的空值情况
display(movie_data1.isnull().sum())
# 对比与原始数据行和列
display(movie_data.shape)
display(movie_data1.shape)

# 将处理后的空值保存赋值movie_data
movie_data = movie_data1


# ---
# 
# ---
# 
# ## 第二节 根据指定要求读取数据
# 
# 
# 相比 Excel 等数据分析软件，Pandas 的一大特长在于，能够轻松地基于复杂的逻辑选择合适的数据。因此，如何根据指定的要求，从数据表当获取适当的数据，是使用 Pandas 中非常重要的技能，也是本节重点考察大家的内容。
# 
# 

# ---
# 
# **任务2.1: ** 简单读取
# 
# 1. 读取数据表中名为 `id`、`popularity`、`budget`、`runtime`、`vote_average` 列的数据。
# 2. 读取数据表中前1～20行以及48、49行的数据。
# 3. 读取数据表中第50～60行的 `popularity` 那一列的数据。
# 
# 要求：每一个语句只能用一行代码实现。

# In[284]:


# 1. 读取数据表中名为 `id`、`popularity`、`budget`、`runtime`、`vote_average` 列的数据。
movie_data1 = movie_data[['id', 'popularity', 'budget', 'runtime', 'vote_average']]
# print(movie_data1)

# 读取数据表中前1～20行以及48、49行的数据。
# 创建1-20行索引
index = [x for x in range(20)]
# 增加48、49行索引
for x in range(47,49):
    index.append(x)
#2. 读取数据表中前1～20行以及48、49行的数据。
# 改进 
movie_data2 = movie_data.iloc[index]
# print(movie_data2)
# movie_data2 = movie_data.iloc[0:20].append(movie_data.iloc[47:49])



# 3. 读取数据表中第50～60行的 `popularity` 那一列的数据。
movie_data3 = movie_data.iloc[49:60][['popularity']]
# print(movie_data3)


# ---
# 
# **任务2.2: **逻辑读取（Logical Indexing）
# 
# 1. 读取数据表中 **`popularity` 大于5** 的所有数据。
# 2. 读取数据表中 **`popularity` 大于5** 的所有数据且**发行年份在1996年之后**的所有数据。
# 
# 提示：Pandas 中的逻辑运算符如 `&`、`|`，分别代表`且`以及`或`。
# 
# 要求：请使用 Logical Indexing实现。

# In[285]:


# 读取数据表中 popularity 大于5 的所有数据。
movie_test = movie_data[movie_data['popularity'] > 5]
# print(movie_test)
# 读取数据表中 popularity 大于5 的所有数据且发行年份在1996年之后的所有数据。
movie_test = movie_data[(movie_data['popularity'] > 5) & (movie_data['release_year'] > 1996)]
# print(movie_test)


# ---
# 
# **任务2.3: **分组读取
# 
# 1. 对 `release_year` 进行分组，使用 [`.agg`](http://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.core.groupby.DataFrameGroupBy.agg.html) 获得 `revenue` 的均值。
# 2. 对 `director` 进行分组，使用 [`.agg`](http://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.core.groupby.DataFrameGroupBy.agg.html) 获得 `popularity` 的均值，从高到低排列。
# 
# 要求：使用 `Groupby` 命令实现。

# In[286]:


# 对 release_year 进行分组，使用 .agg 获得 revenue 的均值。
# movie_test = movie_data.groupby('release_year')['revenue'].mean()
movie_test = movie_data.groupby('release_year')['revenue'].agg(['mean'])
# print(movie_test)
# 对 director 进行分组，使用 .agg 获得 popularity 的均值，从高到低排列。
movie_test = movie_data.groupby('director')['popularity'].agg(['mean']).sort_values(by="mean" , ascending=False)
# print(movie_test)


# ---
# 
# ---
# 
# ## 第三节 绘图与可视化
# 
# 接着你要尝试对你的数据进行图像的绘制以及可视化。这一节最重要的是，你能够选择合适的图像，对特定的可视化目标进行可视化。所谓可视化的目标，是你希望从可视化的过程中，观察到怎样的信息以及变化。例如，观察票房随着时间的变化、哪个导演最受欢迎等。
# 
# <table>
# <thead><tr><th>可视化的目标</th><th>可以使用的图像</th></tr></thead><tbody>
#  <tr><td>表示某一属性数据的分布</td><td>饼图、直方图、散点图</td></tr>
#  <tr><td>表示某一属性数据随着某一个变量变化</td><td>条形图、折线图、热力图</td></tr>
#  <tr><td>比较多个属性的数据之间的关系</td><td>散点图、小提琴图、堆积条形图、堆积折线图</td></tr>
# </tbody></table>
# 
# 在这个部分，你需要根据题目中问题，选择适当的可视化图像进行绘制，并进行相应的分析。对于选做题，他们具有一定的难度，你可以尝试挑战一下～

# **任务3.1：**对 `popularity` 最高的20名电影绘制其 `popularity` 值。

# In[287]:


# 按popularity降幂排序，选取其最高的前20数据
movie_popularity = movie_data.sort_values(by='popularity' , ascending=False).head(20)
# movie_popularity = movie_data.groupby('original_title')['popularity'].agg(['mean']).sort_values(by="mean" , ascending=False)
base_color = sb.color_palette()[0]
# print(movie_popularity)
# 绘制电影名称对应的popularity柱状图
sb.barplot(data = movie_popularity,y = 'original_title', x = 'popularity',color = base_color);
# plt.xticks(rotation = 90)


# ---
# **任务3.2：**分析电影净利润（票房-成本）随着年份变化的情况，并简单进行分析。

# In[288]:


# 筛选年份
year_profit = movie_data[[ 'release_year']]
# 插入利润列
year_profit.insert(1,'profit',movie_data[ 'revenue'] - movie_data['budget'])
# 求每一年的平均利润
year_profit = year_profit.groupby((['release_year'])).mean()
# 转成dataframe,索引变成列
year_profit.reset_index(inplace=True)
# print(year_profit)
# year_profit['release_year'] = year_profit.index
plt.figure(figsize=(14, 6));
sb.barplot(data = year_profit,y = 'profit', x = 'release_year',color = base_color);
plt.xticks(rotation = 70);

# 简要分析
# 从下图可以看出随着年份的增长，电影净利润总体也随着增长，每年平均净利也是呈增长但后期变化不大


# ---
# 
# **[选做]任务3.3：**选择最多产的10位导演（电影数量最多的），绘制他们排行前3的三部电影的票房情况，并简要进行分析。

# In[289]:


# 获取最多产的10位导演
directors = movie_data.groupby(['director'])['original_title'].agg(['size']).sort_values(by='size' , ascending=False).head(10)
# print(directors)
#创建一个空的dataframe,用于存放前10导演、电影、票房
top_data = pd.DataFrame(columns = ['director','original_title','revenue'])
# 遍历导演获取对应前三票房 DataFrame
plt.figure(figsize=(12, 4));
for index, row in directors.iterrows():
    # 筛选导演对应票房前三的电影
    top_data = top_data.append(movie_data[(movie_data['director'] == index)].sort_values(by='revenue' , ascending=False).head(3)[['director','original_title','revenue']])

# 整理完数据后生成的图（改进的地方）
sb.barplot(data = top_data,y = 'revenue', x = 'original_title', hue='director', dodge=False, palette="Set2");
    
plt.xticks(rotation = 90);
# print(top_data)
# 简要分析
# 产量高导演的电影票房不一定就高


# ---
# 
# **[选做]任务3.4：**分析1968年~2015年六月电影的数量的变化。

# In[290]:


# 柱状图
# movie_d1 = movie_data
# 发行日期release_year转化成数字方便获取数据年份区间
# movie_d1['release_year'] = pd.to_numeric(movie_d1['release_year'])
# movie_d1 = movie_data['release_year'].between(1968, 2015)
# # 筛选1968年~2015年电影 数据
# movie_d1 = movie_d1[(movie_d1['release_year'] >= 1968) & (movie_d1['release_year'] <= 2015)]
# 筛选1968年~2015年电影 数据(改进)
movie_d1 = movie_data[movie_data['release_year'].between(1968, 2015)]
# 选取1968年~2015年6月份的电影
movie_d1 = movie_d1[movie_data['release_date'].str.startswith('6/')]
# 分组获取
movie_d2 = movie_d1.groupby(['release_year'])['id'].agg(['count'])
# 将时间索引变成列
movie_d2.reset_index(inplace=True)
# movie_d2['release_year'] = movie_d2.index
# print(movie_d2)
# 绘制968年~2015年六月电影的数量的柱状图
plt.figure(figsize=(14, 6));
sb.barplot(data = movie_d2,y = 'count', x = 'release_year',color = base_color);
plt.xticks(rotation = 80);
# 简要分析
# 如下图1968年~2015年六月电影随着年份的增长，电影数量也逐渐的增多,中间1989－2000年6月份电影数量有点下降


# ---
# 
# **[选做]任务3.5：**分析1968年~2015年六月电影 `Comedy` 和 `Drama` 两类电影的数量的变化。

# In[291]:


# 审阅老师的思路代码实现 👍

# 将电影类型进行拆分，重新生单个类型的列
df_genres = movie_data.drop('genres', axis=1).join(movie_data['genres'].str.split('|', expand=True).stack().reset_index(level=1, drop=True).rename('genres'))

# 筛选条件1 年份
sel_year = df_genres['release_year'].between(1968, 2015)

# 筛选条件2 六月
sel_June = pd.to_datetime(df_genres['release_date']).dt.month == 6

# 筛选条件3 类型
sel_genre = df_genres['genres'].isin(['Drama', 'Comedy'])
# print(df_genres['genres'].head(10))
# 筛选数据并作图(参考逻辑读取部分)
plt.figure(figsize=[18, 5])
# 满足前面筛选条件数据集作图
sb.countplot(data=df_genres[sel_year&sel_June&sel_genre], x='release_year', hue='genres')
plt.xticks(rotation=90);

## 简要分析
# 如下图1968年~2015年六月电影随着年份的增长,Comedy（喜剧）和Drama（戏剧）电影数量总体逐年增长


# In[292]:


# 集群条形图

# 思路是对了，但是代码实现有待提高

movie_d1 = movie_data
# 发行日期release_year转化成数字方便获取数据年份区间
movie_d1['release_year'] = pd.to_numeric(movie_d1['release_year'])
# 筛选1968年~2015年电影 数据
movie_d1 = movie_d1[(movie_d1['release_year'] >= 1968) & (movie_d1['release_year'] <= 2015)]
# movie_d1 = movie_data['release_year'].between(1968, 2015)
# 选取1968年~2015年6月份的电影
movie_d2 = movie_d1[movie_data['release_date'].str.startswith('6/')]
# 选取Comedy 和 Drama 两类的电影数据
movie_d2 = movie_d2[(movie_data['genres'].str.find('Comedy') > 0) | (movie_data['genres'].str.find('Drama') > 0)]

## 分类命名Comedy，Drama 组装合并DataFrame, 合并两个子集错误,应该是索引重复问题
# movie_comedy = movie_d2[(movie_data['genres'].str.find('Comedy') > 0)]
# movie_comedy['genres'] = 'Comedy'
# movie_drama = movie_d2[(movie_data['genres'].str.find('Drama') > 0)]
# movie_drama['genres'] = 'Drama'
# movie_d3.append(movie_drama)
# movie_d3 = pd.concat([movie_comedy,movie_drama])
# print(movie_d3)

# 分组统计
ct_counts = movie_d2.groupby(['release_year', 'genres']).size()
# 将序列转成dataframe,实际未生效？？？？？
ct_counts.reset_index(name = 'count')
print(ct_counts)
# 画集群条形图，这里是失败的，所以注释掉了
# sb.countplot(data = ct_counts, x = 'release_year', hue = 'genres')
# plt.xticks(rotation = 15);

## 简要分析
## 😓任务失败所以这里木有分析，请老师指导


# > 注意: 当你写完了所有的代码，并且回答了所有的问题。你就可以把你的 iPython Notebook 导出成 HTML 文件。你可以在菜单栏，这样导出**File -> Download as -> HTML (.html)、Python (.py)** 把导出的 HTML、python文件 和这个 iPython notebook 一起提交给审阅者。
