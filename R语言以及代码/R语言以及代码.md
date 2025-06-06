<!-- 封面样式 -->
<style>
@page {
    size: A4;
    margin: 20mm;
}
body {
    font-family: Arial, sans-serif;
    font-size: 14pt;
    line-height: 1.5;
}
.cover-page {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    height: 100vh;
    text-align: center;
}
.cover-title {
    font-size: 36pt;
    font-weight: bold;
    margin-bottom: 20px;
}
.cover-subtitle {
    font-size: 24pt;
    margin-bottom: 40px;
}
.cover-author {
    font-size: 18pt;
    margin-bottom: 20px;
}
.cover-date {
    font-size: 16pt;
}
</style>

<!-- 封面内容 -->
<div class="cover-page">
    <div class="cover-title">经济研究的实证方法与R代码</div>
    <div class="cover-subtitle">The Empirical Method of Economic Research and R Code</div>
    <div class="cover-author">作者:Laiqi Song</div>
    <div class="cover-date">日期: 2024年12月5日</div>
    在做任何分析之前都要做协变量平衡分析，防止由于对照组和控制组变量分布造成的误差。
</div>

- [1.OLS](#1ols)
- [9.RDD](#9rdd)
- [实用小代码](#实用小代码)

<div style="page-break-after: always;"></div>

## <div style="font-size:25px;text-align:center;">1.OLS</div>

```R
lm(Y ~ X + C, data = data)#表示进行OLS回归，其中Y为被解释变量，X为解释变量，C为控制变量
#画散点图
    p <- ggplot(data, aes(x = x, y = y))+
    # 添加散点图层
    geom_point() +
    # 添加标题和坐标轴标签（可选）
    labs(title = "Scatter Plot", x = "X Variable", y = "Y Variable") +
    # 选择主题（可选，这里使用默认主题）
    theme_bw()
    # 显示绘制的散点图
    print(p)
```






<div style="page-break-after: always;"></div>

## <div style="font-size:25px;text-align:center;">9.RDD</div>

```R
rdrobust(Y, X, covs = C）#表示进行rdd，其cov为控制变量，x为驱动变量，y为被解释变量，其系数为截距，就是我们要的
summary(rdrobust(Y,X,covs = C, kernel = "uniform"))#表示进行对于数据进行核加权的rdd分析
summary(rdrobust(Y,X,covs = C,  p = 2))# 使用局部二次函数进行RD估计，假设带宽为默认值（可根据需要调整带宽参数h）
summary(rdrobust(Y,X,covs = C,  h = 40))# 使用带宽为40进行RD估计

```

利用RDHonest进行画图和比较更加广泛的RDD分析 [RDDHonest画图](https://github.com/kolesarm/RDHonest/blob/master/doc/RDHonest.pdf)
[RDHonest公式文档，fuzzy or sharp](https://cran.r-project.org/web//packages/RDHonest/RDHonest.pdf)

```R
#参数自己去看文档，这里表示出来所有的参数
RDHonest(formula,data,subset,weights,cutoff = 0,M,kern = "triangular",na.action,opt.criterion = "MSE",h,se.method = "nn",alpha = 0.05,beta = 0.8,J = 3,sclass = "H",
result[["coefficients"]] #看估计的参数
```

<div style="page-break-after: always;"></div>

## <div style="font-size:25px;text-align:center;">10.SCM</div>

```R
library(tidyverse)
library(haven)
library(Synth)
library(devtools)
library(SCtools)

read_data <- function(df)
{
  full_path <- paste("https://github.com/scunning1975/mixtape/raw/master/", 
                     df, sep = "")#读取github的数据，返回数据的路径
  df <- read_dta(full_path) #读取stata数据
  return(df)
}

texas <- read_data("texas.dta") %>%
  as.data.frame(.)#读取stata数据，转化为数据框

dataprep_out <- dataprep(#数据准备
  foo = texas,#需要处理的数据框
  predictors = c("poverty", "income"),#构建预测变量
  predictors.op = "mean",#定义了对预测变量进行的操作，这里指定为 "mean"，意味着会使用这些预测变量在相应时间区间内的均值来参与后续的计算等操作
  time.predictors.prior = 1985:1993,#预测变量计算均值等操作所依据的时间范围，这里是从 1985 年到 1993 年这个时间段。
  special.predictors = list(
    list("bmprison", c(1988, 1990:1992), "mean"),#这里在弄匹配变量的匹配条件（均值是给定的权重）
    list("alcohol", 1990, "mean"),
    list("aidscapita", 1990:1991, "mean"),
    list("black", 1990:1992, "mean"),
    list("perc1519", 1990, "mean")),
  dependent = "bmprison",#因变量的名称
  unit.variable = "statefip",#单位变量的名称
  unit.names.variable = "state",#展示单位名称的变量
  time.variable = "year",
  treatment.identifier = 48,#确定了处理组的标识
  controls.identifier = c(1,2,4:6,8:13,15:42,44:47,49:51,53:56),#给出了对照组的标识集合
  time.optimize.ssr = 1985:1993,#权重的考虑范围
  time.plot = 1985:2000
)

synth_out <- synth(data.prep.obj = dataprep_out)

path.plot(synth_out, dataprep_out)
```


## <div style="font-size:25px;text-align:center;">实用小代码</div>

```R
1 #导入csv数据进入
data <- read.csv("path/to/your/file.csv")
#导入excel数据进入
install.packages("readxl")
library(readxl)
data <- read_excel("path/to/your/file.xlsx")
#导入stata数据进入
install.packages("haven")
library(haven)
data <- read_dta("path/to/your/file.dta")
3.#创建一个新的变量并将其输入数据框，数据集中
my_data <- my_data %>% mutate(firstmonth = agemo_mda==0)
4.#用$取出来数据框中的变量，也可以用其来取出变量值
Y <- my_data$cod_any
5.#删除数据列表中的变量值的行
my_data <- subset(my_data, firstmonth!= 1)
my_data <- my_data[!(my_data$firstmonth == 1), ]
6.#看数据frame中的数据列表,
data[['列表名称']]
data$列表名称[“数据标签”]
7.#数值转换
data$列表名称 <- as.numeric(data$列表名称)#不同的转换就是不同的as.类型
```

<div style="page-break-after: always;"></div>

# 找数据网站

[克雷格列表网](https://hongkong.craigslist.org/)
[权威的大数据竞赛平台 —— 数据泉](https://www.datafountain.cn)
[卡格乐数据集](https://www.kaggle.com/datasets)