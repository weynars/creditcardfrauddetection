library(unbalanced)
library(data.table) #fread
library(caret) #data partition
library(e1071)
library(randomForest)
library(ROSE) #for unbalanced data
library(rpart) #decision tree algorithm

#set path
path <- "C:/Users/Credit Card"
setwd(path)# define the filename

filename <- "creditcard.csv"
# load the CSV file from the local directory
dataset <- fread(filename, stringsAsFactors = F, sep = ",", header =T)
#Column class to be predicted
col_class <- 31
#Set classification column from Integer to Factor
dataset[[col_class]] <- as.factor(dataset[[col_class]])

# dimensions of dataset
dim(dataset)
# dimensions of dataset
str(dataset)
# list types for each attribute
sapply(dataset, class)
# take a peek at the first 5 rows of the data
head(dataset)
# summarize attribute distributions
summary(dataset)

# list the levels for the class and proportions
levels(dataset[[col_class]])
table(dataset[[col_class]])
prop.table(table(dataset[[col_class]]))

# create a list of p% of the rows in the original dataset we can use for training
set.seed(7)
validation_index <- createDataPartition(dataset[[col_class]], p=0.80, list=FALSE)
# select (1-p)% of the data for validation
validation <- dataset[-validation_index,]
# use the remaining p% of data to training and testing the models
training <- dataset[validation_index,]

# split input and output
x <- training[,1:(col_class-1)]
y <- training[[col_class]]

#1 - Over Sampling
training_over <- ubOver(X=x, Y=y, k=2.5)
training_over <- cbind(training_over$X,training_over$Y)
colnames(training_over)[col_class] <- "Class"
prop.table(table(training_over$Class))

#2 - Under Sampling
training_under <- ubUnder(X=x, Y=y, perc=40,method="percPos")
training_under <- cbind(training_under$X,training_under$Y)
colnames(training_under)[col_class] <- "Class"
prop.table(table(training_under$Class))

#3 - Smote Sampling
training_smote <- ubSMOTE(X=x, Y=y, perc.over=200, k=5, verbose=TRUE)
training_smote <- cbind(training_smote$X,training_smote$Y)
colnames(training_smote)[col_class] <- "Class"
prop.table(table(training_smote$Class))

#4 - ROSE Sampling
training_rose <- ROSE(Class~., data=training, N=5000, p=0.5, seed=1)$data
prop.table(table(training_rose$Class))

#Training Regression Trees
mod.normal <- rpart(Class~., data=training)
mod.over <- rpart(Class~., data=training_over)
mod.under <- rpart(Class~., data=training_under)
mod.smote <- rpart(Class~., data=training_smote)
mod.rose <- rpart(Class~., data=training_rose)
#Predicting Validation Data
pred.normal <- predict(mod.normal, validation)
pred.over <- predict(mod.over, validation)
pred.under <- predict(mod.under, validation)
pred.smote <- predict(mod.smote, validation)
pred.rose <- predict(mod.rose, validation)
#Confusion Matrix
confusionMatrix(as.integer(pred.normal[,2]>0.5), validation[[col_class]])
confusionMatrix(as.integer(pred.over[,2]>0.5), validation[[col_class]])
confusionMatrix(as.integer(pred.under[,2]>0.5), validation[[col_class]])
confusionMatrix(as.integer(pred.smote[,2]>0.5), validation[[col_class]])
confusionMatrix(as.integer(pred.rose[,2]>0.5), validation[[col_class]])

#Training Regression Trees - Didn't use the normal and over-sampling because the computation time is too long
rf.under <- randomForest(Class~., data=training_under)
rf.smote <- randomForest(Class~., data=training_smote)
rf.rose <- randomForest(Class~., data=training_rose)
#Predicting Validation Data
pred.rf.under <- predict(rf.under, validation)
pred.rf.smote <- predict(rf.smote, validation)
pred.rf.rose <- predict(rf.rose, validation)
#Confusion Matrix
confusionMatrix(pred.rf.under, validation[[col_class]])
confusionMatrix(pred.rf.smote, validation[[col_class]])
confusionMatrix(pred.rf.rose, validation[[col_class]])
