# R语言学习

```r
################################ R 参考手册 ###############################
search()。找文件
install.packages()。安装包
????library

######### (I) R  data structure(Four types)##############
R的逻辑结构

###### 1) vectors
创建向量列表等，可以使用角标  
a=c(1,2,3,6,-2,4)。
a=c(1:10)
a[1]
a[c(1,3,5)]
a[2:6]

######data sampling。
数据抽样等
seq(0,100,1) ###start from 0 to 100, interval 2      创建序列
rep(1:5,each=3)####repeat 3 times    重复序列
rnorm(10,mean=0,sd=3) ###draw 10 numbers following the normal distribution with mean=0 and sd=3
正态分布
runif(10,min=0,max=1)
均匀分布
sample(c("A","B","C","D","E"),4,replace=TRUE)
放回抽样
sample(c("A","B","C","D","E"),4,replace=FALSE)
不放回抽样

######2) matrix
创建矩阵
y<-matrix(1:20,nrow=5,ncol=4)
y<-matrix(1:20,nrow = 5,ncol=4,byrow=TRUE)  参数是按行排还是按列
cells<-c(1,2,3,4,5,6)
mymatrix<- matrix(cells,nrow=2,ncol=3,byrow=FALSE)

# rowname columnname
对于矩阵行列重新命名
cells <- c(1,26,24,68)  
rnames  <-c("R1","R2")
cnames <-c("c1","c2")
mymatrix <- matrix(cells,nrow = 2,ncol = 2,byrow=TRUE, dimnames= list(rnames,cnames))
mymatrix

#extract matrix elements
角标抽取
x<-matrix(1:10,nrow=2,ncol=5) 
x
x[,1]
x[1,4]
x[1,c(4,5)]

####### 3) array
创建数组
dim1 <-c("A1","A2")
dim2 <-c("B1","B2","B3")
dim3 <-c("C1","C2","C3","C4")
z <- array(1:24,c(2,3,4), dimnames = list(dim1,dim2,dim3)) # three-dimensional (2*3*4) array
z

#######  4) data.frame 大数据
导入数据
patientID <- c(1,2,3,4)
age <- c(25,34,28,52)
diabetes<- c("Type1","Type2","Type1","Type1")
status <- c("Poor","Improved","Excellent","Poor")
patientdata <- data.frame(patientID,age, diabetes,status)
patientdata
数据抽取
patientdata[1:2]
patientdata[1:4]
patientdata[c("diabetes","status")]
table (patientdata$diabetes, patientdata$status) 

数据美元符号指名
patientdata$status
patientdata$diabetes
patientdata$age
mode(status) #see the variable type("numemric or character")

###########(II) Common data converting################
数据转换
# 1) convert data type (numeric character vector matrix ...)
#####types of data
patientdata$age
#####numeric
数据验证
is.numeric #check the variable type
#####character
as.character  #convert the variable type
数据转换
as.numeric
is.numeric(patientdata$age)
patientdata$age=as.numeric(patientdata$age) 
patientdata$age <- as.character(patientdata$age)
class(patientdata$age) #####check the data type

#####date
c<-as.Date("2012-06-12")  #####date
数据赋值以及转换
class(c)
c<-as.POSIXct("2012-06-12 12:32") #####date and time 
class(c)
c<-as.character("2012-06-12 12:32") #####date and time 
class(c)

###########matrix and dataframe conversion
数据和矩阵转换
y<- as.data.frame(mymatrix)
y<-as.matrix(mymatrix)
is.matrix(y)

数据编辑与删除
y<-edit(y) #edit and look at the data#
rm(y)# #####remove the data y
rm(age, cells)

#2) rename variable
变量重命名
install.packages("reshape")
install.packages("Rcpp")
library(reshape)  # reshape package contains the "rename" function #
patientdata <- rename(patientdata,c(age="age1",status="status1"))
patientdata<- edit(patientdata)

?attach    代码减少误差，可以绑定变量数据

#3) order  
attach(patientdata)
数据绑定
Xnewdata <- patientdata[order(diabetes,age),] 
newdata <- patientdata[order(diabetes,-age),] 
detach(patientdata)

上下等价
Xnewdata <- patientdata[order(patientdata$diabetes,patientdata$age),] 
newdata <- patientdata[order(patientdata$diabetes,-patientdata$age),] 
rm(newdata,Xnewdata)

#4) data merge
数据融合，按照相同数据拼接数据

#####merge by variable
total<- merge(dataframeA,dataframeB, by="ID")
total<- merge(dataframeA,dataframeB, by=c("ID","Country"))
#####add a new variable 
patientdata$gender<-c("Male","Female","Female","Male")
gender<-c("Male","Female","Female","Male")
total<-cbind(patientdata,gender)
######add new observations
total<-rbind(patientdata,patientdata)
####delete variable
删除变量
patientdata1=subset(patientdata, select=-age)
patientdata1=subset(patientdata, select=-c(age,status))
rm(patientdata1)
####delete observation
删除观测值
patientdata2=patientdata[-c(2,3,4),]

#5) subset the data based on conditions
patientdata3=subset(patientdata,age<30&status=="Excellent") 
抽数据按条件放成子集
newdata=patientdata[which(patientdata$age<30&status=="Excellent"),]

#6) input and output data set
#input#
导入数据
df <- read.csv("C:/Users/cuixu/Desktop/Rdataset.csv")
设置链接文件环境
curwd = setwd('C:/Users/cuixu/Desktop/')

header是指第一行为变量名称
df1 = read.table("Rdataset.csv",header=TRUE,sep= ",") #header:weather the first row contains the variable name
df <- edit(df)
df

df <- read.csv("Rdataset.csv")
rm("df")

#output#
设置输出的数据环境
setwd('C:/Users/cuixu/Desktop/')
write.table(df,"df.csv",sep=",")相对环境
write.table(df, file ="C:/Users/cuixu/Desktop/df1.csv", sep =",")
绝对环境

# 7) Loops
循环
x=0
for(i in 1:10)
{
  x=x+1
}
x

x=0
i=1
while(i<=10){ ####  ######condition: if condition is true, then loop until condition become false and then stop.=x+1
   x=x+1
   i=i+1
}
x

###calculate 1^3+2^3+...+100^3 ### example
s=0
for(i in 1:100)
{
  s=s+i^3
}
s

s=0
i=1
while(i<=100){
  s=s+i^3
  i=i+1
}            
s

#8)conditional statement: if else, ifelse
a1=rnorm(100,0,1)
a2=runif(100,min=-1,max=1) 均匀分布
b=ifelse(a1<=a2,1,0)  类正则

if (a1<=a2){b=1}else{b=0} #######cannot run,only get 1 value

b1=rep(0,100)
for(i in 1:100){
if(a1[i]<=a2[i]){
  b1[i]=1
}else{
  b1[i]=0
}
  }
b1
data <- data.frame(b,b1,a1,a2)

ls()  ##list all the dataset

#############(III) Matrix Operations################
# simple operations
矩阵运算
8*6
2^16
sqrt(2)
abs(-65)
?sqrt  ###check the function"sqrt" help 
?abs
?round
?floor

8%%3 ###reminder /m+odulo ????  求余数
20%%7

#######Example: calculate 2+4+6+8+...+20
s=0
for (i in 1:20) {
  if(i%%2==1) next     ####jump out of the loop
  next表示直接下一个循环，不走接下来的
  s=s+i
}
s

help("abs")
round(5.634)
floor(3.824) #the largest integers not greater than x
向下取整
ceiling(6.821)#the smallest integers not less than x
向上取整
log(1)
exp(0)

SquareRoot2=sqrt(2)
SquareRoot2

HoursYear<-365*24
HoursYear

######vector calculation 
向量计算
X1<-c(35, 40, 40, 42, 37, 45, 43, 37, 44, 42, 41, 39)
mean(X1)  
sd(X1)
max(X1)
min(X1)
length(X1)
sum(X1)
max(patientdata$age)

## Matrix Operations
dfmat <-as.matrix(df)
dim(dfmat)
y<-dfmat[,2]

y <- as.numeric(dfmat[,2])
X <- matrix(as.numeric(dfmat[,3:4]),ncol=2)

class(dfmat[,3:4])
is.numeric(dfmat[,3:4])

X*y 这个不行，需要转置
t(X)%*%y #matrix multiplication
矩阵乘法
crossprod(X,y) # t(X) %*%Y
与上等价

mydata<-matrix(rnorm(30),nrow=6).  自动分配
apply(mydata,1,mean) ####row mean。 求数据的方式，1代表行，2代表列
apply(mydata,2,mean) ####column mean
apply(df[,2:4],2,mean) ####column mean
?apply
#########Statistic ##############

install.packages(c('npmc', 'ggm', 'gmodels', 'vcd', 'Hmisc',       
                   'pastecs', 'psych', 'doBy', 'reshape')) 

## 1) describe statistic
描述性统计
summary(df)
summary(df$X1)

## 2) correlation analysis
相关性分析
cor(df[2:4])
cor(df[2],df[4])

## 3) t-test (compare the difference between two groups)
t检验
library(MASS)
UScrime=UScrime
t.test(Prob ~ So,data=UScrime) ###compare crime probability
t.test(Prob ~ So,data=UScrime,var.equal=TRUE) #variance homoscedasticity#

### 4) linear regression###
线性回归
lmout = lm(Y ~ X1 + X2,data=df)
print(lmout)
summary(lmout)
source("printCoefmat.R") ###change the significant star
lmout$coef
fitted(lmout) ####bhat
res1<-residuals(lmout)####res
res1

### test linear regression (multicollinearity problem)
线性检验，多重共线性检验
library(car)
vif(lmout)
sqrt(vif(lmout))>2 #squart vif >2  has multicollinearity problem

### 5) logistic regression###
逻辑回归。 假设分布不一样，logistic分布函数，
install.packages(c('AER', 'robust', 'qcc')) 
data(Affairs, package = "AER")
summary(Affairs)
table (Affairs$affairs)

# create binary outcome variable
创建二进制变量
Affairs$ynaffair[Affairs$affairs > 0] <- 1
Affairs$ynaffair[Affairs$affairs == 0] <- 0
Affairs$ynaffair <- factor(Affairs$ynaffair, levels = c(0, 1), labels = c("No", "Yes"))
table(Affairs$ynaffair)

# fit full model
fit.full <- glm(ynaffair ~ gender + age + yearsmarried + 
                  children + religiousness + education + occupation + rating, 
                family = binomial(),data = Affairs)
summary(fit.full)

# predict probabilities
预测可能性
Affairs$prob<- predict(fit.full,data=Affairs,type="response")

help("glm")

### 5) Probit model ###
Probit模型使用正态分布函数
fit.full1 <- glm(ynaffair ~ gender + age + yearsmarried + 
                   children + religiousness + education + occupation + rating, 
                 family = binomial(link="probit"),data = Affairs)
summary(fit.full1)

###################################plot####################################
画图

## scatter plot
散点图
plot(df$X2,df$Y,xlab="col1", ylab="col2",xlim=c(-4,4),ylim=c(0,11))
############### Elementary Graphics ############
## scatter plot
plot(X[,2],y,xlab="col1", ylab="col2",xlim=c(-4,4),ylim=c(0,8))
title("scatterplot")
abline(c(0,1),lwd=2,lty=2)  #####a, b	the intercept and slope, single values
######lwd=x: specifies the width of lines (1 is default, >1 is thicker
######lty=x specifies type of line (e.g. solid vs dashed)
?abline

a=rnorm(1000)
## Histogram
直方图
hist(rnorm(1000),breaks=50,col="green")
a=mtcars
hist(mtcars$mpg,breaks=12,col="red",xlab="Miles",main="colored histogram with 12 bins")
hist(mtcars$mpg,freq=FALSE,breaks=12,col="red",xlab="Miles",main="colored histogram with 12 bins")
lines(density(mtcars$mpg),col="blue",lwd=2)

## Density
密度图
plot(density(mtcars$mpg),col="blue")  ####kernel density 

```