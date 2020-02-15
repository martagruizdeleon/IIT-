#Homework1: Marta Garcia Ruiz de Leon A204

#Exercise 1
library(ggplot2)
library(dplyr)
library(datasets)

iris_data<-data.frame(iris)
#Boxplot of the 4 features
boxplot(iris_data[1:4])

#The feature with the biggest IQR is 
#Standard deviation for each of the features
sd_Sepal.Length=sd(iris_data$Sepal.Length)
sd_Sepal.Width=sd(iris_data$Sepal.Width)
sd_Petal.Lenght=sd(iris_data$Petal.Length)
sd_Petal.Width=sd(iris_data$Petal.Width)
#The resutls agree with the empirical value, Petal.Length is the feature with the largest IQR
#Boxplot with ggplot2

ggplot(iris_data, aes(x=Species, y=Sepal.Length, color=Species)) + 
  geom_boxplot(outlier.colour="red")
ggplot(iris_data, aes(x=Species, y=Sepal.Width, color=Species)) + 
  geom_boxplot(outlier.colour="red")
ggplot(iris_data, aes(x=Species, y=Petal.Length, color=Species)) + 
  geom_boxplot(outlier.colour="red")
ggplot(iris_data, aes(x=Species, y=Petal.Width, color=Species)) + 
  geom_boxplot(outlier.colour="red")
#The Virginica flower experiences a diferent petal length/width

#EXercise 2
library(moments)
trees_data<-data.frame(trees)
#5-NUmber summary of each feature
summary(trees_data$Girth)
summary(trees_data$Height)
summary(trees_data$Volume)

#Histograms for each of the features
hist(trees_data$Girth, border = "blue")
hist(trees_data$Height, border="red")
hist(trees_data$Volume, border="purple")
#The only variable that appears to be normaly distributed is the Height feauture
#The feature Volume appears to have positive skewness
skweness_trees<-data.frame(skewness(trees_data$Girth),skewness(trees_data$Height),skewness(trees_data$Volume))
#Numerically we can see that volume does have a positive skewness; Girth has also a positive skewness but smaller,
#making it difficult to see visually, whilst Height has a small negative skewness

#Exercise 3
#The data is missing the names of the features, so we create a features names vector
features<-c("mpg","cylinders","displacement","horsepower","weight","acceleration","model year","origin","car name")
data_auto<-read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"), header = F, col.names = features, sep = "", as.is=TRUE)
#We create a data frame with the imported file 
autompg_data<-data.frame(data_auto)
#We use the as.numeric function to obtain a numeric column
horsepower<-as.numeric(autompg_data$horsepower)
#We compute the median taking out the NA values
median_horsepwr=median(horsepower,na.rm = T)
mean_Horsepowr1<-mean(horsepower, na.rm = T)
#We substitute the NA values for the median
for (i in 1:length(horsepower)) {
  if(is.na(horsepower[i])){
    horsepower[i]<-median_horsepwr
  }
  
}
#We compute the new mean with the altered values
mean_Horsepowr2<-mean(horsepower)
#The means are very similar but slightly different as in the original we take out the observations that are missing

