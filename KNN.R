set.seed(1234)
#PROBLEM 1
library(MASS)
library(ggplot2)
library(boot)
boston_data<-data.frame(Boston)
lm_mod<-lm(medv~lstat, data=boston_data)
#PLOT linear model
ggplot(boston_data, aes(x = boston_data$lstat, y = boston_data$medv)) + 
  geom_point() +
  stat_smooth(method = "lm", col = "blue")
#PLOT FITTED VALUE VS RESIDUALS
glm.diag.plots(lm_mod)
p1<-ggplot(lm_mod, aes(.fitted, .resid))+geom_point()
p1<-p1+stat_smooth(method="loess")+geom_hline(yintercept=0, col="red", linetype="dashed")
p1<-p1+xlab("Fitted values")+ylab("Residuals")
p1<-p1+ggtitle("Residual vs Fitted Plot")+theme_bw()
p1

library(ggfortify)
autoplot(lm_mod)
#PREDICT and calculate CI PI for n values of lstat
n<-c(5,10,15)
j<-1
mod_pred=matrix(nrow = 3,ncol = 5)
colnames(mod_pred)<-c("fitted value","CI_low","CI_upp","PI_low","PI_Upp")
for (i in 1:3) {
  mod_pred[i,j]<-predict(lm_mod, data.frame(lstat =n[i]))
  CI<-predict(lm_mod, newdata = data.frame(lstat=n[i]),interval = 'confidence',level=0.95)
  mod_pred[i,j+1]<-CI[2]
  mod_pred[i,j+2]<-CI[3]
  PI<-predict(lm_mod, newdata = data.frame(lstat=n[i]),interval = 'prediction',level=0.95)
  mod_pred[i,j+3]<-PI[2]
  mod_pred[i,j+4]<-PI[3]
}
#The prediction inverval and confidence interval should not be the same as the se of the confidencei interval is not the same.

lm_mod2<-lm(boston_data$medv~poly(boston_data$lstat,2))
p3<-ggplot(lm_mod2, aes(.fitted, .resid))+geom_point()
p3 <- p3 + stat_smooth(method = "lm", formula = y ~ x + I(x^2))
r_2<-data.frame(summary(lm_mod)$r.squared,summary(lm_mod2)$r.squared)






#PROBLEM 2
set.seed(1234)
library(caret)
library(dplyr)
library(corrplot)
library(e1071)

#We create a data frame with abalone data
features<-c("Sex","Length","Diameter","Height","Whole Weight","Shucked Weight","Viscera Weight","Shell weight","Rings")
abalone <- read.csv("C:/Users/marta/Downloads/abalone.data", header=FALSE, col.names = features)
#We filter observations with Infant category
abalone_data<-data.frame(droplevels( abalone[-which(abalone$Sex == "I"), ] ))


#We create a partition of the data 80%20%
train_test<-createDataPartition(abalone_data$Sex,p=0.8,list = FALSE)
head(train_test)  
abalone_train<-abalone_data[train_test,]
abalone_test<-abalone_data[-train_test,]

glm_mod<-glm(abalone_train$Sex~.,family=binomial(),data = abalone_train)
summary(glm_mod)
confint(glm_mod)
pred<-predict(glm_mod,newdata = abalone_test)
pred.dt<-ifelse(pred>0.50, "M","F")
Pred <- as.factor(pred.dt)
Predicted <- ordered(Pred, levels = c("M", "F"))
Actual <- ordered(abalone_test$Sex,levels = c("M", "F"))
cm <-confusionMatrix(table(Predicted,Actual))
cm

#ROC METHOD RESPASAAAARRRR QUE ESTA MAL
library(ROCR)
predict_abalon <-predict(glm_mod,newdata = abalone_test)
prediction_abalone <- prediction(predict_abalon,abalone_test$Sex)
perf<- performance(prediction_abalone,"tpr","fpr")
plot(perf)

roc_empirical <- rocit(score =abalone_test$Length , class = abalone_test$Sex,
                       negref = "-") 
summary(roc_empirical)
plot(roc_empirical)
#We calculate the corr matrix
abalone_predictors<-abalone_data[c(-1)]
corMatrix <- cor(abalone_predictors)
corrplot(corMatrix,type = "upper", method = "circle", diag = TRUE, tl.col = "black",tl.srt = 45,)



#PROBLEM 3
names <- c("edibility","cap-shape","cap-surface","cap-color","bruises","odor","gill-attachment","gill-spacing","gill-size","gill-color","stalk-shape","stalk-root","stalk-surface-above-ring","stalk-surface-below-ring","stalk-color-above-ring","stalk-color-below-ring", "veil-type","veil-color","ring-number","ring-type","spore-print-color","population","habitat")
agaricus.lepiota <- read.csv("C:/Users/marta/Downloads/agaricus-lepiota.data",header = FALSE, col.names = names)
#We want to see how many missing values we have in the data
summary(agaricus.lepiota)
agaricus.lepiota[agaricus.lepiota == '?'] <- NA
num<-is.na(agaricus.lepiota$stalk.root)
sum(num)
#We can see that in column e.1 there are 2480 missing values 
#library(DMwR)
library(VIM)
knnOutput<-kNN(agaricus.lepiota,variable = colnames(agaricus.lepiota[12]), k=10)
knnOutput[knnOutput=='?']<-NA
anyNA(knnOutput)
agaricus_data<-data.frame(knnOutput[c(-24)])
#split data 80/20%
sample <- sample.int(n = nrow(agaricus_data), size = floor(.8*nrow(agaricus_data)), replace = F)
train_ex3 <- agaricus_data[sample, ]
test_ex3  <- agaricus_data[-sample, ]

#We create the naives-bayesian model
nvb_model<-naiveBayes(train_ex3$edibility~., train_ex3)
pred <- predict(nvb_model, test_ex3, type = "class")
#Test accuracy of the clasiffier
library(forecast)
pred_train<-predict(nvb_model,train_ex3,type="class")
cm_3<-table(pred_train, train_ex3$edibility,dnn=c("Prediction","Actual"))
n<-sum(cm_3)
diag<- diag(cm_3)
accuracy_train<-sum(diag)/n
a_train<-accuracy(nvb_model)
#Create a confusion matrix to see false positives and negatives
cm_3test<-table(pred, test_ex3$edibility,dnn=c("Prediction","Actual"))
n_test<-sum(cm_3test)
diag_test<- diag(cm_3test)
accuracy_test<-sum(diag_test)/n_test
