### W8 Bagging and Random Forest 
## Bagging can smooth regression function

library(MASS)#导入数据
data(mcycle)
str(mcycle)

library(rpart)
fit <- rpart(accel~.,data=mcycle) 
library(rpart.plot)
prp(fit,type=2)  # plot a rpart model
pred <- predict(fit)
plot(mcycle$times,mcycle$accel,xlab="Time",ylab="Acceleration",main="Single Tree Estimation")
lines(mcycle$times,pred,col="blue")#画好图，不是平滑的线

install.packages("randomForest")#随机森林的包
library(randomForest)
set.seed(1)

fit <- randomForest(accel~.,data=mcycle,mtry=1,ntree=1000)
#表示在构建每一棵决策树时，从全部的自变量（特征）集合中随机选取的特征数量，较小的 mtry 值会使单棵树的随机性更强，但也可能导致每棵树的预测能力相对较弱；而较大的 mtry 值会让单棵树更接近普通的决策树，可能失去随机森林通过引入随机性来提升泛化能力的优势。
#ntree 指的是要构建的决策树的数量
#mtry 的值来规定每棵决策树构建时随机选取特征的数量
pred <- predict(fit)#默认对原来预测
plot(mcycle$times,mcycle$accel,xlab="Time",ylab="Acceleration",main="Bagging Estimation")
lines(mcycle$times,pred,col="blue")


## Bagging for Regression

library(MASS)
data(Boston)
dim(Boston)
set.seed(1)
train <- sample(506,354)

install.packages("randomForest")
library(randomForest)
set.seed(123)
#要使用 Boston 数据集中名为 train 的子集来训练决策树
bag.fit <- randomForest(medv~.,data=Boston,subset=train,mtry=13,importance=TRUE,replace=T)
#T 表示采用有放回抽样
# Bagging is random forest with m=K=13
# "importance=TRUE" means compute variable importance 
bag.fit
#其中的结果有很多，就有不同树所带来的均方误
plot(bag.fit,main="Bagging OOB Errors")  # plot OOB error against number of trees

# Variable Importance Plot

importance(bag.fit)
varImpPlot(bag.fit,main="Variable Importance Plot")

# The first measure is based on OOB prediction error by excluding one given variable
# The second measure is based training sample
# For classification,node impurity = Gini index
# For regression,node impurity = RSS

# Partial Dependence Plot
#偏数据的图，用训练集，rm要绘制偏依赖图的自变量（特征）的参数
partialPlot(bag.fit,Boston[train,],x.var=rm)
partialPlot(bag.fit,Boston[train,],x.var=lstat)

# Test error
#训练集即装袋误差
bag.pred <- predict(bag.fit,newdata=Boston[-train,])
y.test <- Boston[-train,"medv"]
plot(bag.pred, y.test,main="Bagging Prediciton")
abline(0,1)
mean((bag.pred-y.test)^2)

# Comparison with OLS
#ols2误差，同样的训练集
ols.fit <- lm(medv~.,Boston,subset=train)  
ols.pred <- predict(ols.fit,newdata=Boston[-train,])
mean((ols.pred-y.test)^2)  

# Different tuning parameters
#每次选取13个特征，构建1000棵树，每棵树的最小节点数为10
set.seed(123)
bag.fit <- randomForest(medv~.,data=Boston,subset=train,mtry=13,ntree=1000,nodesize=10)
#控制决策树节点的最小样本数量，也就是规定了叶节点（树的最底层节点）中最少需要包含的观测数量为 10 个样本
# Bagging is random forest with m=K=13
# Grow 1000 trees. The default is ntree=500
# Set minimum size of terminal nodes to be 10. The default is nodesize=5

bag.pred <- predict(bag.fit,newdata=Boston[-train,])
mean((bag.pred-y.test)^2)

## Random Forest for regression
#每次选取4个特征
set.seed(123)
forest.fit <- randomForest(medv~.,data=Boston,subset=train)
# The default is mtry=p/3 for regression problem?? so here mtry=4
forest.fit
forest.pred <- predict(forest.fit,newdata=Boston[-train,])
mean((forest.pred-y.test)^2)

# Random Forest test errors and the number of trees
#树不同数量下的随机森林的误差，每次选取4个特征
MSE.forest <- numeric(100)  # initialize
set.seed(123)
for(i in 1:100){
  fit <- randomForest(medv~.,data=Boston,subset=train,ntree=i)
  pred <- predict(fit,newdata=Boston[-train,])
  y.test <- Boston[-train,"medv"]
  MSE.forest[i] <- mean((pred-y.test)^2)
}

# Bagging test errors and the number of trees
#树不同数量下的随机森林的误差，每次选取13个特征
MSE.bag <- numeric(100)
set.seed(123)
for(i in 1:100){
  fit <- randomForest(medv~.,data=Boston,subset=train,ntree=i,mtry=13)
  pred <- predict(fit,newdata=Boston[-train,])
  y.test <- Boston[-train,"medv"]
  MSE.bag[i] <- mean((pred-y.test)^2)
}

# Test Error for a Single Tree
#以最优决策树进行预测
library(rpart)
set.seed(123)
tree.fit <- rpart(medv~.,Boston,subset=train)  
min_cp <-  tree.fit$cptable[which.min(tree.fit$cptable[,"xerror"]),"CP"]
tree.prune <- prune(tree.fit, cp = min_cp)
tree.pred <- predict(tree.prune,newdata=Boston[-train,])
mse <- mean((tree.pred-y.test)^2)   
MSE.tree <- rep(mse,100)

#画图看误差
plot(1:100,MSE.forest,type="l",col="blue",ylab="MSE",xlab="Number of Trees",main="Test Error",ylim=c(10,55))
lines(1:100,MSE.bag,type="l")
lines(1:100,MSE.tree,type="l",col="black",lty=2)
legend("topright",lty=c(2,1,1),col=c("black","black","blue"),legend=c("Best Single Tree","Bagging","Random Forest"))

# how to choose the optimal mtry?
# OOB Error and Optimal mtry

#最优的特征数量的选择
MSE <- numeric(13)
set.seed(123)
for(i in 1:13){
  fit <- randomForest(medv~.,data=Boston,subset=train,mtry=i)#用训练集训练，并且训练集预测
  MSE[i] <- mean(fit$mse)
}
MSE
min(MSE)
which.min(MSE)
plot(1:13,MSE,type="b",xlab = "mtry",main="OOB Errors")
abline(v=which.min(MSE),lty=2)
# OOB error is too optimistic 



# Test Errors and Optimal mtry

MSE <- numeric(13)
set.seed(123)
for(i in 1:13){
  fit <- randomForest(medv~.,data=Boston,subset=train,mtry=i)
  pred <- predict(fit,newdata=Boston[-train,])#用训练集训练，并且测试集预测
  y.test <- Boston[-train,"medv"]
  MSE[i] <- mean((pred-y.test)^2)
}
min(MSE)
which.min(MSE)
plot(1:13,MSE,type="b",xlab = "mtry",main="Test Error")
abline(v=which.min(MSE),lty=2)

### Random Forest for Classification with Sonar Data

install.packages("mlbench")
library(mlbench)
data(Sonar)
dim(Sonar)
table(Sonar$Class)
edit(Sonar)

set.seed(1)
train <- sample(208,158)

# Single Tree

library(rpart)
set.seed(123)
fit <- rpart(Class~.,data=Sonar,subset=train)  
min_cp <-  fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"]
min_cp
fit_best <- prune(fit, cp = min_cp)
pred <- predict(fit_best,newdata=Sonar[-train,],type="class")
y.test <- Sonar[-train,"Class"]
(table <- table(pred,y.test))
(error_rate <- 1-sum(diag(table))/sum(table))

# Logit

fit <- glm(Class~.,data=Sonar,subset=train,family=binomial)
prob <- predict(fit,newdata=Sonar[-train,],type="response")
pred <- prob >= 0.5
(table <- table(pred,y.test))
(error_rate <- 1-sum(diag(table))/sum(table))

# Random Forest

library(randomForest)
set.seed(123)
fit <- randomForest(Class~.,data=Sonar,subset=train,importance=TRUE)
fit

varImpPlot(fit,main="Variable Importance Plot")
partialPlot(fit,Sonar[train,],x.var=V11)

pred <- predict(fit,newdata=Sonar[-train,])
(table <- table(pred, y.test))
(error_rate <- 1-sum(diag(table))/sum(table))

