#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import jieba
import matplotlib
import matplotlib as plt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import collections
from pandas.plotting import scatter_matrix


# In[3]:


txt = open("2020gov.txt", encoding="utf-8").read() 
words = jieba.lcut(txt) 


# In[4]:


count = {}
for word in words:            #  使用 for 循环遍历每个词语并统计个数
    if len(word) < 2:          # 排除单个字的干扰，使得输出结果为词语
        continue
    else:
        count[word] = count.get(word, 0) + 1    #如果字典里键为 word 的值存在，则返回键的值并加一，如果不存在键word，则返回0再加上1


# In[5]:


exclude = ["可以", "一起", "这样"]  # 建立无关词语列表
for key in pd.DataFrame(count.keys()):     # 遍历字典的所有键，即所有word
    if key in exclude:
        del count[key]                  #  删除字典中键为无关词语的键值对
        
df = pd.DataFrame(count.items(),columns = ['Key Words','Frequence'])        # 将字典的所有键值对转化为列表
df = df.sort_values(by='Frequence',ascending=False)
df


# In[6]:


df.describe()


# In[7]:


fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(df['Frequence'], bins = 10, range = (df['Frequence'].min(),df['Frequence'].max()))
plt.title('Frequence Distribution')
plt.xlabel('Frequence')
plt.ylabel('Number of Frequence')
plt.show()


# In[8]:


count = sum(i > 10 for i in df.Frequence)
count


# In[9]:


df=df[df.Frequence > 10]
df['Number'] = np.arange(len(df))
df.set_index('Number',inplace=True)
df.index=df.index+1
df.style.set_properties(**{'text-align': 'right'})


# # Word Cloud

# In[10]:


from wordcloud import WordCloud
from matplotlib.pyplot import imread
from django.urls import path


# In[13]:


path_txt = '2020gov.txt'
path_img = "jerry.jpg"
f = open('2020gov.txt', 'r', encoding='UTF-8').read()
background_image = np.array(imread(path_img))
   # 结巴分词，生成字符串，如果不通过分词，无法直接生成正确的中文词云,感兴趣的朋友可以去查一下，有多种分词模式
   #Python join() 方法用于将序列中的元素以指定的字符连接生成一个新的字符串。
cut_text = " ".join(jieba.cut(f))

wordcloud = WordCloud(font_path="C:/Windows/Fonts/simfang.ttf",width=1000,height=1500,background_color="white", mask=background_image,
    contour_color='black').generate(cut_text)

plt.figure(figsize=(10,6))
plt.imshow(wordcloud.recolor(), interpolation="bilinear")
plt.axis("off")
plt.show()

