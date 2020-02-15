#HOMEWORK 4
#QUESTION 1
par(xpd = NA)
plot(NA, NA, type = "n", xlim = c(0,100), ylim = c(0,100), xlab = "X1", ylab = "X2")
# t1: x = 30; (30, 0) (40, 100)
lines(x = c(30,30), y = c(0,100))
text(x = 30, y = 108, labels = c("t1"), col = "blue")
# t2: y = 40; (0, 40) (30, 40)
lines(x = c(0,30), y = c(40,40))
text(x = -8, y = 40, labels = c("t2"), col = "blue")
# t3: x = 50; (50,0) (50, 100)
lines(x = c(50,50), y = c(0,100))
text(x = 50, y = 108, labels = c("t3"), col = "blue")
# t4: x = 50; (50,0) (0, 30)
lines(x = c(30,50), y = c(30,30))
text(x = 25, y = 30, labels = c("t4"), col = "blue")
# t5: x = 70; (70,0) (70, 100)
lines(x = c(70,70), y = c(0,100))
text(x = 70, y = 108, labels = c("t5"), col = "blue")

text(x = (0+30)/2, y = 20, labels = c("R1"))
text(x = 15, y = (40+100)/2, labels = c("R2"))
text(x = 40, y = 15, labels = c("R3"))
text(x = 40, y = (30+100)/2, labels = c("R4"))
text(x = 60, y = 50, labels = c("R5"))
text(x = 85, y = 50, labels = c("R6"))

#QUESTION 3
p <- seq(0, 1, 0.01)
gini.index <- 2 * p * (1 - p)
class.error <- 1 - pmax(p, 1 - p)
cross.entropy <- - (p * log(p) + (1 - p) * log(1 - p))
matplot(p, cbind(gini.index, class.error, cross.entropy), col = c("purple", "green", "blue"))

#Question 4
par(xpd = NA)
plot(NA, NA, type = "n", xlim = c(-2, 2), ylim = c(-3, 3), xlab = "X1", ylab = "X2")
# X2 < 1
lines(x = c(-2, 2), y = c(1, 1))
# X1 < 1 with X2 < 1
lines(x = c(1, 1), y = c(-3, 1))
text(x = (-2 + 1)/2, y = -1, labels = c(-1.8))
text(x = 1.5, y = -1, labels = c(0.63))
# X2 < 2 with X2 >= 1
lines(x = c(-2, 2), y = c(2, 2))
text(x = 0, y = 2.5, labels = c(2.49))
# X1 < 0 with X2<2 and X2>=1
lines(x = c(0, 0), y = c(1, 2))
text(x = -1, y = 1.5, labels = c(-1.06))
text(x = 1, y = 1.5, labels = c(0.21))

#QUESTION 2
d = as.dist(matrix(c(0, 0.3, 0.4, 0.7, 
                     0.3, 0, 0.5, 0.8,
                     0.4, 0.5, 0.0, 0.45,
                     0.7, 0.8, 0.45, 0.0), nrow = 4))
plot(hclust(d, method = "complete"), labels = c(2,1,4,3))

#QUESTION 3
x <- cbind(c(1, 1, 0, 5, 6, 4), c(4, 3, 4, 1, 2, 0))
plot(x[,1], x[,2])

set.seed(123)
labels <- sample(2, nrow(x), replace = T)
labels
plot(x[, 1], x[, 2], col = (labels + 1), pch = 20, cex = 2)

centroid1 <- c(mean(x[labels == 1, 1]), mean(x[labels == 1, 2]))
centroid2 <- c(mean(x[labels == 2, 1]), mean(x[labels == 2, 2]))
plot(x[,1], x[,2], col=(labels + 1), pch = 20, cex = 2)
points(centroid1[1], centroid1[2], col = 2, pch = 4)
points(centroid2[1], centroid2[2], col = 3, pch = 4)

labels <- c(1, 1, 1, 2, 2, 2)
plot(x[, 1], x[, 2], col = (labels + 1), pch = 20, cex = 2)
points(centroid1[1], centroid1[2], col = 2, pch = 4)
points(centroid2[1], centroid2[2], col = 3, pch = 4)

centroid1 <- c(mean(x[labels == 1, 1]), mean(x[labels == 1, 2]))
centroid2 <- c(mean(x[labels == 2, 1]), mean(x[labels == 2, 2]))
plot(x[,1], x[,2], col=(labels + 1), pch = 20, cex = 2)
points(centroid1[1], centroid1[2], col = 2, pch = 4)
points(centroid2[1], centroid2[2], col = 3, pch = 4)

plot(x[, 1], x[, 2], col=(labels + 1), pch = 20, cex = 2)

#Question 6
set.seed(123)
Control <- matrix(rnorm(50 * 1000), ncol = 50)
Treatment <- matrix(rnorm(50 * 1000), ncol = 50)
X <- cbind(Control, Treatment)
 # linear trend in one dimension
pr.out <- prcomp(scale(X))
summary(pr.out)$importance[, 1]

X <- rbind(X, c(rep(10, 50), rep(0, 50)))
pr.out <- prcomp(scale(X))
summary(pr.out)$importance[, 1]
#PROBLEM 1
set.seed(1122)
X1 <- data.frame("X" = rnorm(100, mean = 5, sd = 2), "class" = as.factor(rep(0, 100)))
X2 <- data.frame("X" = rnorm(100, mean = -5, sd = 2), "class" = as.factor(rep(1, 100)))
X<- rbind(X1, X2)
str(X)
set.seed(1122)
tree1 <- rpart(class ~ ., data = X, method = "class")
rpart.plot(tree1, type=4, extra=104, fallen.leaves=TRUE, main="Full Tree")
summary(tree1)


set.seed(1122)
X1 <- data.frame("X" = rnorm(100, mean = 1, sd = 2), "class" = as.factor(rep(0, 100)))
X2 <- data.frame("X" = rnorm(100, mean = -1, sd = 2), "class" = as.factor(rep(1, 100)))
X<- rbind(X1, X2)
str(X)
set.seed(1122)
tree2 <- rpart(class ~ ., data = X, method = "class")
rpart.plot(tree2, type=4, extra=104, fallen.leaves=TRUE, main="Full Tree")
summary(tree2)
zp <- prune.rpart(tree2, cp = 0.1)
summary(zp)
plot(zp)

#PROBLEM 2
names<-c("Type","Alcohol","Malic Acid","Ash","Alcalinity of ash","Magnesium","Total phenols","Flavanoids","Nonflavanoid phenols","Proanthocyanins","Color intensity","Hue","OD280/OD315 of diluted wines","Proline")
wine <- read.csv("C:/Users/marta/Downloads/wine.data", header=FALSE, col.names = names)
apply(wine , 2, mean)
apply(wine , 2, var) 
#We must scale the data becuase they have different means and variances
pr.out<-prcomp(wine , scale=TRUE)
names(pr.out)
biplot(pr.out , scale=0)


# Variability of each principal component: pr.var
pr.var <- pr.out$sdev^2
# Variance explained by each principal component: pve
pve <- pr.var / sum(pr.var)
pve
plot(pve , xlab="Principal Component ", ylab="Proportion of Variance Explained ", ylim=c(0,1),type='b') 
plot(cumsum(pve), xlab="Principal Component ", ylab=" Cumulative Proportion of Variance Explained ", ylim=c(0,1), type='b')

#PROBLEM 3
set.seed(123)
library(tidyverse)
library(factoextra)
library(cluster)
df<-data.frame((USArrests))
#Lets see if we should scale the data
apply(df , 2, mean)
apply(df , 2, var)
#We can see there is a big difference between the means and variances so we should scale the data

df_scaled<-data.frame(scale(df))
#Calculate the k means 
k2 <- kmeans(df_scaled, centers = 2, nstart = 25)
k3<-kmeans(df_scaled, centers = 3, nstart = 25)
k4<-kmeans(df_scaled, centers = 4, nstart = 25)
k5<-kmeans(df_scaled, centers = 5, nstart = 25)
k6<-kmeans(df_scaled, centers = 6, nstart = 25)
k7<-kmeans(df_scaled, centers = 7, nstart = 25)
k8<-kmeans(df_scaled, centers = 8, nstart = 25)
k9<-kmeans(df_scaled, centers = 9, nstart = 25)
k10<-kmeans(df_scaled, centers = 10, nstart = 25)  

wss <- function(k) {
  kmeans(df_scaled, k, nstart = 10 )$tot.withinss
}

# Compute and plot wss for k = 2 to k = 10
k.values <- 2:10

# extract wss for 2-15 clusters
wss_values <- map_dbl(k.values, wss)

plot(k.values, wss_values,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")

fviz_nbclust(df_scaled, kmeans, method = "wss")

#plot the optimal cluster is 8
fviz_cluster(k8, data = df_scaled)


#PROBLEM 4
#import data frame 
winequality.white <- read.csv("C:/Users/marta/Downloads/winequality-white.csv", sep=";")

#Excluding the quality target variable, use hclust to perform a hierarchical clustering of the data 
#with single as well as complete linkage
#Lets see if we should center/scale the data
apply(winequality.white , 2, mean)
apply(winequality.white , 2, var)
df_wine<-winequality.white[-c(12)]
#Since the means are quite different, we should scale the data 
wine_scaled<-data.frame(scale(df_wine))
#We perform a single hierachical clustering
hc.single<-hclust(dist(wine_scaled), method ="single")
#We now perform a complete linkage
hc.complete<-hclust(dist(wine_scaled), method ="complete")

par(mfrow=c(1,3))
plot(hc.complete,main="Complete Linkage ", xlab="", sub="", cex=.9) 
plot(hc.single, main="Single Linkage ", xlab="", sub="", cex=.9)

cuts_complete<- cutree(hc.complete, k=2)
summary(cuts_complete)
cuts_single<-cutree(hc.single, k=2)
summary(cuts_single)
table(cuts, wine_scaled$quality)

