getwd()
setwd("C:/Users/318si/OneDrive/바탕 화면/2023-1/통계적모델링과머신러닝실습/HW/HW1")

#### 4-(1)
data = read.csv('Q4.csv')

# Y와 X는 exponential 관계
# build a regression model
fit = lm(Y ~ exp(X), data=data)
summary(fit)
# estimate the model parameters
fit$coefficients
plot(data)
#### 4-(2)
# obtain residuals
fit$residuals
# investigate the iid normal assumption of errors
par(mfrow = c(2,2))
plot(fit)
# residual plot 결과 등분산성 가정 위반됨을 확인했다. 또한 이차의 패턴을 보인다.
# Normal Q-Q plot 결과 normal 가정 만족
# durbin-watson test
rt = fit$residuals
dw_fn = function(r){
  d=numeric(79)
  for (i in 1:79){
    d[i] = (r[i+1]-r[i])^2
  }
  dw = sum(d)/sum(r^2)
  return(dw)
}
dw_fn(rt)

# durbin-watson statistic 값이 1.691862나오면서 2에 근접한것으로 보아 error간의 correlation이 작다는 것을 확인할 수 있다.

# install.packages("lmtest")
library(lmtest)
dwtest(fit)
# 참고로 lmtest의 dwtest함수로 확인해본 결과로는 p-value = 0.08288로귀무가설을 기각할 수 없으므로 자기상관성이 없다라는 결론이이 나오게 된다.

#### 4-(3)
# 1. 이분산 문제 해결
library(matrixcalc)

X = data$X
Y = data$Y
n=length(Y)
w= diag(rep(1,n)) #initial covariance matrix

f = function(beta,X)
{
  X1 = X 
  beta[1] + beta[2]*exp(beta[3]*X1)
}

# objective function : RSS
RSS = function(beta, Y, X) sum((Y-f(beta,X))^2)
# gradient vector for the obj function
grv = function(beta,Y,X){
  X1 = X
  R = Y - f(beta,X)
  c(-2*sum(R), -2*sum(R*exp(beta[3]*X1)),
    -2*beta[2]*sum(R*X1*exp(beta[3]*X1)))  
}
# optimization
ml1=optim(rep(0.1,3), RSS, gr=grv, method='BFGS', X=X, Y=Y)
ml1


# objective function for mean function
obj.mean = function(beta,Y,X,S) t(Y-f(beta,X)) %*% solve(S) %*% (Y-f(beta,X))
# gradient vector for the obj function
gr.mean = function(beta,Y,X,S)
{
  sigma2 = diag(S)
  X1 = X
  R = Y - f(beta,X)
  c(-2*sum(R/sigma2), -2*sum(R*exp(beta[3]*X1)/sigma2),
    -2*beta[2]*sum(R*X1*exp(beta[3]*X1)/sigma2))  
}
lambda = 0.01

beta.new=ml1$par
# variance function
mdif = 100000
while(mdif > 0.000001){
  yhat = f(beta.new, X)
  r = Y- yhat
  z = cbind(1, yhat)
  gam.hat = solve(t(z) %*% w %*% z+ lambda * diag(rep(1, ncol(z)))) %*% t(z) %*% w %*% r^2 # nonlinear: r^2
  sigma = z %*% (gam.hat)
  s = diag(as.vector(sigma^2))
  
  if (is.non.singular.matrix(s)) w = solve(s)
  else w = solve(s + 0.000000001*diag(rep(1,nrow(s))))
  
  ml2 = optim(beta.new, obj.mean, gr=gr.mean, method = 'BFGS', Y=Y, X=X, S=s)
  beta.old = beta.new
  beta.new = ml2$par
  mdif = max(abs(beta.new - beta.old))
}

beta.new
gam.hat
yhat = f(beta.new,X)
sigma = z %*% gam.hat
r = (Y - yhat)/sigma

# residual plot
par(mfrow = c(1,1))
plot(yhat, r, ylim = c(-4,4))

lines(c(0,30), c(0,0), col='red')

# durbin-watson test
rt = r
dw_fn(rt)
## 결과값이 0.683334로 자기상관성이 높게 나왔다. 따라서 독립성 가정 위반 문제를 해결해보자.

library(astsa)
acf2(rt)

# ar(1)을 사용해보자

# MLE: Multivariate normal distribution
X = data$X
Y = data$Y
n = length(Y)
S = diag(rep(1,n))    # initial covariance matrix

mdif = 1000000
beta.old = ml2$par

f = function(beta,X)
{
  X1 = X 
  beta[1] + beta[2]*exp(beta[3]*X1)
}

obj.mean = function(beta,Y,X,S) t(Y-f(beta,X)) %*% solve(S) %*% (Y-f(beta,X))

while(mdif > 0.0000001){
  ml3 = optim(beta.old, obj.mean,method='BFGS', Y=Y, X=X, S=S)
  beta.new = ml3$par
  beta.old = beta.new
  
  yhat = f(beta.new, X)
  r = Y - yhat
  ar1 = sarima (r, 1,0,0, no.constant=T, details=F)
  alpha = ar1$fit$coef
  sigma2 = ar1$fit$sigma2
  
  mdif = max(abs(beta.new - beta.old))
  
  # Construct covariance matrix
  S = matrix(nrow=n,ncol=n)
  for (i in 1:n)
  {
    for (j in 1:n)
    {
      if (i == j) S[i,j] = 1
      if (i != j) S[i,j] = alpha^(abs(i-j))
    }
  }
  S = (sigma2 / (1-alpha^2)) * S
}
# estimate the parameters of the regression model
round(beta.new,4)

# residual plot
final_fit = lm(Y~f(beta.new,X))
summary(final_fit)
final_fit$coefficients

par(mfrow = c(2,2))
plot(final_fit)
lines(c(0,150),c(0,0),col='red')


# 4-(4)
round(S,10)


# 6-(1)
data = read.csv('Q6.csv')

colSums(is.na(data))
# NA값이 없음을 확인함

plot(data)

# linear regression
fit1 = lm(Y ~X1+X2+X3, data=data)
summary(fit1) 
plot(fit1)
# 분석 결과를 보면 X2변수가 y를 예측함에 있어서 유의미하지 않은 변수임을 알 수 있다.


# GAM
# install.packages('gam')
library(gam)
fit2 =gam(Y ~ s(X1, 5) + s(X2, 5) + s(X3, 5), data=data)
summary(fit2)
plot(fit2)
# 분석 결과를 보면 X2변수가 y를 예측함에 있어서 유의미하지 않은 변수임을 알 수 있다.

# 6-(2)
library(matrixcalc)

# X1과 X3에 대해 어떠한 관계가 있는지 정보가 없기 때문에 우선 gam model을 통해 각 변수와 y의 관계를 찾아보고자 한다.
fit2 <- gam(Y ~ s(X1,5) + s(X3,5), data = data)
summary(fit2)
par(mfrow = c(1,2))
plot(fit2)
# 위 그래프를 통해 X1에 대해서는 선형 함수, X3에 대해서는 이차 함수가 적절함을 확인할 수 있다.
fit2_1 <- gam(Y ~ X1 + s(X3,5), data = data)
anova(fit2, fit2_1) 
# anova 결과를 통해서도 X1에 대해서는 선형함수, X3에 대해서는 이차함수를 적용하는 것이 적절함을 확인했다.


X = cbind(data$X1,data$X3)
colnames(X) = c('X1', 'X3')
Y = data$Y

## 모델
fit3 <- lm(Y ~ X1 + X3 + I(X3^2), data = data)
summary(fit3)
plot(fit3)
# 그러나 residual plot을 확인한 결과 아직 이분산성 문제가 남아있기 때문에 이를 해결하기 위해 linear variance function을 사용하고자 한다.

# model : Y = beta1 + beta2*X1 + beta3*X3 + beta4*X3^2
f = function(beta,X){
  X1 = X[,1]; X2 = X[,2]
  beta[1] + beta[2]*X1 + beta[3]*X2 + beta[4]*X2^2
}

# objective function for mean fuction
obj.mean = function(beta,Y,X,S) t(Y-f(beta,X)) %*% solve(S) %*% (Y-f(beta,X))

# gradient vector of the objective function
gr.mean = function(beta,Y,X,S){
  sigma2 = diag(S)
  X1 = X[,1]; X2 = X[,2]
  R = Y - f(beta,X)
  c(-2*sum(R/sigma2), -2*sum(R*X1/sigma2), -2*sum(R*X2/sigma2), -2*sum(R*X2^2/sigma2)) 
}

#linear variance function
beta.new = coef(fit3)
W = diag(rep(1,length(Y)))
mdif = 100000

while(mdif > 0.000001){
  Yhat = f(beta.new,X)
  r = Y - Yhat
  Z = cbind(1,Yhat)
  gam.hat = solve(t(Z) %*% W %*% Z) %*% t(Z) %*% W %*% abs(r)
  sigma = Z %*% gam.hat
  S = diag(as.vector(sigma^2))
  if (is.non.singular.matrix(S)) W = solve(S)
  else W = solve(S + 0.000000001*diag(rep(1,nrow(S))))
  ml2 = optim(beta.new, obj.mean, gr=gr.mean,method='BFGS', Y=Y, X=X, S=S)
  beta.old = beta.new
  beta.new = ml2$par
  mdif = max(abs(beta.new - beta.old))
}

coef(fit3) # 기존 LSE값

beta.new # variance function을 고려한 새로운 회귀 계수

# 6-(3)
Yhat = f(beta.new,X)
sigma = Z %*% gam.hat
r = (Y - Yhat)/sigma
par(mfrow = c(1,1))
# residual plot
plot(Yhat,r,ylim=c(-4,4))
lines(c(0,150),c(0,0),col='red')

