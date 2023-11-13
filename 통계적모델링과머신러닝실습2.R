setwd("C:/Users/318si/OneDrive/바탕 화면/2023-1/통계적모델링과머신러닝실습/HW/HW2")
library(tidyverse)
library(mice)
library(rms)
library(finalfit)

# Q1 ####
train1 <- read.csv("train.csv")
test1 <- read.csv("test.csv")

# 1.description of data
train1 %>% head()
train1 %>% str()
describe(train1)

## missing value
md.pattern(train1)
missing_pairs(train1, 'Y', c('X1', 'X2', 'X3', 'X4'))
missing.cluster=naclus(train1,method="average")
plot(missing.cluster)

## 2. 전처리 ##
## MICE imputation
set.seed(0)
imp = mice(train1, m = 10 , method = c('pmm', 'pmm', 'pmm','pmm', ''),print=F)
plot(imp, c('X1', 'X2', 'X3', 'X4'))
densityplot(imp, scales=list(relation = 'free'), layout = c(2,2))

## Transformation of Data
# Skewness of Data
library(moments)
imp_train <- complete(imp)
par(mfrow=c(1,4))
for(j in c("X1","X2","X3","X4")){
  hist(imp_train[,j],main=j,xlab=skewness(imp_train[,j]))
}

# 3. prediction model
# linear regression model
fit = with(imp, lm(Y ~ X1 + X2 + X3 + X4))
# model averaging
summary(pool(fit))
# 모든 변수가 유의하다고 나오나 mice 과정에서 y 정보 사용했는지 check 
tr = na.omit(train1)
fit1 <- lm(Y~., data = tr)
summary(fit1)

library("MLmetrics")
M = imp$m
imp_train <- vector(mode = "list", length = M)
for (m in 1:M){
  imp_train[[m]] = complete(imp, m)
}

p.model <- function(dat) {
  lm(Y~., data = dat)
}

fit.imp <- lapply(imp_train, p.model)
yhat <- lapply(fit.imp, predict, newdata = test1)
yhat <- matrix(unlist(yhat), nrow(test1), M)
final.yhat <- apply(yhat, 1, mean)

MSE(final.yhat, test1[, "Y"])



# Q2
train2 <- read.csv("pm25_tr.csv")
test2 <- read.csv("pm25_te.csv")

## 1. EDA
# missing value 확인
train2 %>% is.na %>% apply(2,sum)
test2 %>% is.na %>% apply(2,sum)
# 결측치가 존재하지 않음

str(train2)
str(test2)
# 범주형 변환
train2$cbwd<- as.factor(train2$cbwd)
test2$cbwd<- as.factor(test2$cbwd)

# year 제거
train2 <- train2 %>% select(-year)
test2 <- test2 %>% select(-year)

# skewness확인
par(mfrow=c(1,4))
for (j in c('DEWP','TEMP','PRES', 'Iws')){
  hist(train2[,j], main=j,xlab = skewness(train2[,j]))}

# 상관관계 확인
library(corrplot)
par(mfrow = c(1,1))
cor1 <- cor(train2[,c('pm25', 'DEWP', 'TEMP', 'PRES', 'Iws')], method = 'pearson')
corrplot(cor1)

# 경향성 확인
par(mfrow = c(1,1))
plot(train2$pm25, type = 'line')

# 2. 전처리
## 2.1 Yeo-Johnson transformation
library(bestNormalize)
train2_1 <- train2
train2_1$Iws <- yeojohnson(train2_1$Iws)$x.t
skewness(train2_1$Iws)

test2_1 <- test2
test2_1$Iws <- yeojohnson(test2$Iws)$x.t
skewness(test2_1$Iws)

## 2.2 차원 축소 
pr <- prcomp(train2_1[, c("DEWP", "TEMP", "PRES")], center = TRUE, scale = TRUE)
summary(pr)
screeplot(pr, type = "l", main = "scree plot")

pca.tr <- predict(pr, newdata = train2_1[, c("DEWP", "TEMP", "PRES")])[,1]
pca.te <- predict(pr, newdata = test2_1[, c("DEWP", "TEMP", "PRES")])[,1]

train2_1$PC <- pca.tr
test2_1$PC <- pca.te

# 3. 모델링
library(gam)
library(lmtest)
train2_tr <- train2_1[,c("pm25", "cbwd", "Iws", "PC")]
fit1 = lm(pm25 ~ Iws + PC + cbwd, data = train2_tr )
summary(fit1) # 모델 설명력이 좋지 않음

fit2 = gam(pm25 ~ s(Iws, 5) + s(PC, 5) + cbwd, data = train2_tr)
summary(fit2)
par(mfrow = c(1,3))
plot(fit2)

fit2_1 = gam(pm25 ~ Iws + s(PC,5) + cbwd , data = train2_tr)
summary(fit2_1)

anova(fit2, fit2_1) # 차이가 유의하기 때문에 nonlinear가 되어야 함

# 최종모델
fit3 = lm(pm25 ~ poly(Iws,2) + poly(PC,2) + cbwd, data = train2_tr)
summary(fit3)
par(mfrow = c(2,2))
plot(fit3) # residual plot을 확인한 결과 -> 1. 이분산성문제 2. normal가정 위반 문제
fit3 = lm(log(pm25) ~ poly(Iws,2) + poly(PC,2) + cbwd, data = train2_tr)
summary(fit3)
par(mfrow = c(2,2))
plot(fit3)
library(lmtest)
dwtest(fit3) # 2. 독립성 위반

# 3.1. 이분산성 가정 위반 문제 해결 
# Variance function
X = model.matrix(log(pm25) ~ Iws + I(Iws^2) + PC+I(PC^2) + cbwd, data = train2_tr)
Y = as.vector(train2_tr$pm25)

library(matrixcalc)
beta.new = fit3$coefficients #initial parameter
W = diag(rep(1,length(Y)))
mdif = 100000
while(mdif > 0.000001){
  Yhat = X %*% beta.new
  r = Y - Yhat
  Z = cbind(1,Yhat)
  gam.hat = solve(t(Z) %*% W %*% Z) %*% t(Z) %*% W %*% abs(r)
  sigma = Z %*% gam.hat
  S = diag(as.vector(sigma^2))
  
  if (is.non.singular.matrix(S)){
    W = solve(S)} 
  else {
    W = solve(S + 0.000000001*diag(rep(1,nrow(S))))}
  
  beta.old = beta.new
  beta.new = solve(t(X) %*% W %*% X) %*% t(X) %*% W %*% Y
  mdif = max(abs(beta.new - beta.old))}

beta.new # 최종 beta

# Error in solve.default(t(X) %*% W %*% X) : system is computationally singular: reciprocal condition number = 3.8785e-20

Yhat = X %*% beta.new
sigma = Z %*% gam.hat
r = (Y - Yhat)/sigma

# residual plot
par(mfrow=c(1,1))
plot(Yhat,r,ylim=c(-4,4))
lines(c(-100,200),c(0,0),col='red')


# 3.2. 독립성 가정 위반 문제 해결
library(astsa)
acf2(residuals(fit3)) #  plot을 통해 AR(2) 모형이 적절하다고 판단
ar1 = sarima(residuals(fit3), 2,0,0, no.constant=T)
# AR(2)모형에 대한 문제를 해결하지 못함

## 4. prediction 
Y = test2_1$pm25
Yhat = exp(predict(fit3, test2_1))

# test MSE
testMSE = mean((Y-Yhat)^2)
testMSE
