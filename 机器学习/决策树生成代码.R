install.packages("rpart")
library(rpart)

library(MASS)
data(Boston)
dim(Boston)
set.seed(1)
train <- sample(506,354)# 70% of the data训练，30%的数据测试
?rpart

# minsplit节点中最少的观测数量，只有当节点中的观测数大于等于这个值时，该节点才有可能被进一步划分（分裂）成子节点
# minbucket表示叶节点中最少需要包含的观测数量，最终生成的树结构中，叶节点所包含的样本数不能小于 
# maxdepth 所构建的决策树从根节点到叶节点的最长路径上的节点数最多为 30 个

set.seed(123)
fit <- rpart(medv~.,data=Boston,subset=train,minsplit = 20, minbucket = 5, maxdepth = 30)  
fit#构建分类或回归树模型

#画图命令
op <- par(no.readonly = TRUE)#功能强大的函数，用于控制 R 中绘图相关的各种参数 no.readonly = TRUE 这个参数，par() 函数就只会返回那些可以被修改的图形参数设置
par(mar=c(1,1,1,1))#把图形的下、左、上、右四边距离图形边框的空白空间都设定为相当于 1 个文本行高度的大小
plot(fit,margin=0.1)#生成树结构
text(fit)#加上树结构的标签
par(op)

#cptable通常是指 “Complexity Parameter Table”（复杂度参数表）
#nsplit列：表示决策树的分裂次数
#CP（Complexity Parameter）列：复杂度参数，它是衡量决策树复杂度的一个指标。CP 值越小，对应的树越复杂
#相对误差，它是相对于一个完整（可能是过度复杂）的树模型的误差度量，根节点为0
#xerror列：交叉验证误差
plotcp(fit)
fit$cptable


min_cp <-  fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"]
min_cp

#会根据这个复杂度参数对初始决策树（fit）进行修剪，得到一个复杂度更合适的决策树（fit_best）
fit_best <- prune(fit, cp = min_cp)#树的减枝操作，据给定的复杂度参数（cp）来简化决策树模型。

install.packages("rpart.plot")
library(rpart.plot)

#以 fit_best 这个决策树模型对象为基础，按照指定的 type=2 这种绘图类型来绘制决策树的图形。
prp(fit_best,type=2)  # plot a rpart model

# Test Error

tree.pred <- predict(fit_best,newdata=Boston[-train,])
y.test <- Boston[-train,"medv"]
mean((tree.pred-y.test)^2)   # MSE

plot(tree.pred,y.test,main="Tree Prediction")
abline(0,1)  



# Comparison with OLS
#线性回归的最小均方误差
ols.fit <- lm(medv~.,Boston,subset=train)  
ols.pred <- predict(ols.fit,newdata=Boston[-train,])
mean((ols.pred-y.test)^2)  

plot(ols.pred,y.test,main="OLS Prediction")
abline(0,1)  







### Classification Trees with Bank Marketing Data
#学斌的例子
setwd("C:/Users/cuixu/Desktop")
bank <- read.csv("bank-additional.csv",header = TRUE,sep=";")
str(bank,vec.len=1)
bank$duration <- NULL
prop.table(table(bank$y))

set.seed(1)
train <- sample(4119,3119)

library(rpart)
set.seed(123)
fit <- rpart(y~.,data=bank,subset=train)
plotcp(fit)


fit$cptable
min_cp <-  fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"]
min_cp

fit_best <- prune(fit, cp = min_cp)
op <- par(no.readonly = TRUE)
par(mar=c(1,1,1,1))
plot(fit_best,uniform=TRUE,margin=0.1)
text(fit_best,cex=1.5)
par(op)


tree.pred <- predict(fit_best,bank[-train,],type="class")
y.test <- bank[-train,"y"]
(table <- table(tree.pred,y.test))
(accuracy <- sum(diag(table))/sum(table))
(sensitivity <- table[2,2]/(table[1,2]+table[2,2]))

tree.prob <- predict(fit_best,bank[-train,],type="prob")
tree.pred <- tree.prob[,2] >= 0.1
(table <- table(tree.pred,y.test))
(accuracy <- sum(diag(table))/sum(table))


# Use information entropy as splitting criterion

set.seed(123)
fit <- rpart(y~.,data=bank,subset=train,parms=list(split="information"))
min_cp <-  fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"]
fit_best <- prune(fit, cp = min_cp)
tree.pred <- predict(fit_best,bank[-train,],type="class")
(table <- table(tree.pred,y.test))
(accuracy <- sum(diag(table))/sum(table))
(sensitivity <- table[2,2]/(table[1,2]+table[2,2]))

# Remove default restrictions

set.seed(123)
fit <- rpart(y~.,data=bank,subset=train,control=rpart.control(minsplit = 0,minbucket = 0))
min_cp <-  fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"]
fit_best <- prune(fit, cp = min_cp)
tree.pred <- predict(fit_best,bank[-train,],type="class")
(table <- table(tree.pred,y.test))
(accuracy <- sum(diag(table))/sum(table))
  