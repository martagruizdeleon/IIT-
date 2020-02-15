#PROBLEM 1
library(caret)
library(Metrics)
set.seed (123)

labels<-c("LongPosCenter","Prismatic coeff","Lenght-disp ratio","Beam-draught ratio","Length-beam ratio","Froude number","Resid res")
yacht_hydrodynamics <- read.table("C:/Users/marta/Downloads/yacht_hydrodynamics.data", quote="\"", comment.char="",header = FALSE,col.names = labels)

#Create a 80/20 test-train split
train_test<-createDataPartition(yacht_hydrodynamics$Resid.res,p=0.8,list = FALSE)
head(train_test)  
yacht_train<-yacht_hydrodynamics[train_test,]
yacht_test<-yacht_hydrodynamics[-train_test,]

#Create a lm 
mod_lm<-lm(Resid.res~.,data = yacht_train)

#Calculate MSE, RMSE and Rsquared
MSE_train<-mse(yacht_train$Resid.res,mod_lm$fitted.values)
RMSE_train<-rmse(yacht_train$Resid.res, mod_lm$fitted.values)
r_sq<-summary(mod_lm)$r.squared

#Bootstrap from N=1000 with traincontrol method
train.control<-trainControl(method = "boot",number=1000)
model<-train(Resid.res~.,data=yacht_hydrodynamics,method="lm",trControl=train.control)
print(model)
histogram(model,metric="RMSE")

#Compare test sets from lm and bootstrap model

pred_testlm<-predict(mod_lm,newdata =yacht_test)
MSE_test<-mse(yacht_test$Resid.res,pred_testlm)
boot_train<-(fitted.values(model))[train_test]
boot_test<-(fitted.values(model))[-train_test]
MSE_test_boot<-mse(yacht_test$Resid.res,boot_test)

#PROBLEM 2
library(caret)
library(Metrics)
library(corrplot)
set.seed (123)
german <- read.table("C:/Users/marta/Downloads/german.data-numeric", quote="\"", comment.char="")
names(german)[25]<-c("class")
summary(german)
num<-is.na(german)
sum(num)
levels(german$class)
#There are no missing values in the data
german$class<-factor(german$class)
levels(german$class)
unique(german$class)
#Create a data partition
#Create a 80/20 test-train split
train_test2<-createDataPartition(german$class,p=0.8,list = FALSE)
head(train_test2)  
german_train<-german[train_test2,]
german_test<-german[-train_test2,]

glm_mod<-glm(class~.,family=binomial(),data = german_train)
summary(glm_mod)

#We can see that not all the variables are significant
#Those significant are attributes 1,2,3,5,7,11,15,16,17,18,19
german_predictors<-german[c(-25)]
corMatrix <- cor(german_predictors)
corrplot(corMatrix,type = "upper", method = "circle", diag = TRUE, tl.col = "black",tl.srt = 45,)

german_train<-german_train[,-c(4,6,8,9,10,12,13,14,20,21,22,23,24)]
german_test<-german_test[,-c(4,6,8,9,10,12,13,14,20,21,22,23,24)]

glm_mod2<-glm(class~.,family=binomial(),data = german_train)
summary(glm_mod2)
# training Precision/Recall and F1 results 
pred<-fitted(glm_mod)
pred.dt<-ifelse(pred>=0.50, "2","1")
Pred <- as.factor(pred.dt)
Predicted <- ordered(Pred, levels = c("1", "2"))
Actual <- ordered(german_train$class,levels = c("1", "2"))
cm <-confusionMatrix(table(Predicted,Actual))
cm$byClass[5]
cm$byClass[6]
cm$byClass[7]


#Use traincontrol and train function to perform a k=10 fold cross-validation
train.control<-trainControl(method = "cv", number = 10)
# Train the model
model<- train(class~., data = german, method = "glm",trControl = train.control,family=binomial)
# Summarize the results
print(model)
#obtain cross-validated training Precision/Recall and F1 values
# training Precision/Recall and F1 results 
cv_train<-(fitted.values(model))[train_test2]
cv_test<-(fitted.values(model))[-train_test2]
pred_cv<-cv_train
pred.dt<-ifelse(pred_cv>=0.50, "2","1")
Pred <- as.factor(pred.dt)
Predicted <- ordered(Pred, levels = c("1", "2"))
Actual <- ordered(german_train$class,levels = c("1", "2"))
cm <-confusionMatrix(table(Predicted,Actual))
cm$byClass[5]
cm$byClass[6]
cm$byClass[7]

#Comparison of k-fold validation and original model

pred_glm<-predict(glm_mod,newdata = german_test,type = "response")
pred.dt<-ifelse(pred_glm>=0.50, "2","1")
Pred <- as.factor(pred.dt)
Predicted <- ordered(Pred, levels = c("1", "2"))
Actual <- ordered(german_test$class,levels = c("1", "2"))
cm <-confusionMatrix(table(Predicted,Actual))
cm$byClass[5]
cm$byClass[6]
cm$byClass[7]

pred_cv<-cv_test
pred.dt<-ifelse(pred_cv>=0.50, "2","1")
Pred <- as.factor(pred.dt)
Predicted <- ordered(Pred, levels = c("1", "2"))
Actual<- ordered(german_test$class,levels = c("1", "2"))
cm2 <-confusionMatrix(table(Predicted,Actual))
cm2$byClass[5]
cm2$byClass[6]
cm2$byClass[7]


#PROBLEM 3 
set.seed(123)
library(fastDummies)
library(dummies)
library(glmnet)
data_mtcars<-data.frame(mtcars)
data_mtcars$am<-factor(data_mtcars$am)
levels(data_mtcars$am)
#Create dummy variable for am 1=
mt_cars<-dummy_cols(data_mtcars,select_columns = c("am"),remove_first_dummy = TRUE)
mt_cars<-data_mtcars
mt_cars$am<-factor(mt_cars$am)


#Split the data 80/20
train_test3<-createDataPartition(mt_cars$mpg,p=0.8,list = FALSE)
mtcars_train<-mt_cars[train_test3,]
mtcars_test<-mt_cars[-train_test3,]

#create a linear model
lm_mod<-lm(mpg~.,data=mtcars_train)

summary(lm_mod)

#Create a ridge regression using glmnet
x<-model.matrix(mpg~.,mtcars_train)[,-1] 
y<-mtcars_train$mpg
#We create a sequence of 100 lambdas
lambdas<-10^seq(10,-2, length =100) 
#We fit the model
fit <- glmnet(x,y, alpha = 0, lambda = lambdas)
summary(fit)
#use cross validation to determine the minimun value for lambda
cv.ridge <-cv.glmnet (x, y ,alpha =0)
best.ridge<-cv.ridge$lambda.min
#Plot training MSE as a function of lambda
plot(cv.ridge)
#Out-of-sample test set performance
best_fit<-glmnet(x,y,alpha=0,lambda = best.ridge)
coef(best_fit)
x_test<-model.matrix(mpg~.,mtcars_test)[,-1]
predict_lm<-predict(lm_mod,newdata = mtcars_test)
predict_ridge<-predict(fit,s=best.ridge,newx = x_test)
mse_lm<-mse(mtcars_test$mpg,predict_lm)
mse_ridge<-mse(mtcars_test$mpg,predict_ridge)
mean((predict_lm-mtcars_test$mpg)^2)
mean((predict_ridge-mtcars_test$mpg)^2)

#PROBLEM 4
data_swiss<-data.frame(swiss)

#We create a data partition
train_test4<-createDataPartition(data_swiss$Fertility,p=0.8,list = FALSE)
swiss_train<-data_swiss[train_test4,]
swiss_test<-data_swiss[-train_test4,]

#Fit a lm with all features
lm_mod<-lm(Fertility~.,data=swiss_train)
summary(lm_mod)

#the following feautures are relevant: agriculture, education, catholic, infant.mortality

#Perform a lasso regression
x<-model.matrix(Fertility~.,swiss_train)[,-1] 
y<-swiss_train$Fertility
#We create a sequence of 100 lambdas
lambdas<-10^seq(10,-2, length =100) 
#We fit the model
fit <- glmnet(x,y, alpha = 1, lambda = lambdas)
summary(fit)

#Determine the minimun value for lamda
cv.lasso <-cv.glmnet (x, y ,alpha =1)
best.lasso<-cv.lasso$lambda.min

#Plot training MSE as a function of lambda
plot(cv.lasso)

#Out-of-sample test set performance
x_test<-model.matrix(Fertility~.,swiss_test)[,-1]
best_fit<-glmnet(x,y,alpha=0,lambda = best.lasso)
coef(best_fit)
predict_lm<-predict(lm_mod,newdata = swiss_test)
predict_lasso<-predict(fit,s=best.lasso, newx =x_test) 
mse_lm<-mse(swiss_test$Fertility,predict_lm)
mse_lasso<-mse(swiss_test$Fertility,predict_lasso)