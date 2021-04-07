#The caret package provides a consistent interface 
#into hundreds of machine learning algorithms 
#and provides useful convenience methods for 
#data visualization, data resampling, model tuning and model comparison etc.

library(randomForest)
library(caret)
library(ggplot2)
library(kernlab)
library(tm)
library(dplyr)
require(caTools) 
library(tidyr)
library(CatEncoders)
library(corrplot)

suppressMessages(library(caret))
#loading the train and test data
data=read.csv(file="fake_job_postings.csv")
#viewing the dataset
#View(data)
dim(data)

#dropinf Nan values
# make a copy
df_nan<- data
# replace empty factor levels with NA
df_nan[df_nan==''] <- NA
View(df_nan)
# keep only relevant factor vars 
names(df_nan)
df_nan<- df_nan[,c(10:16,17:18)]
str(df_nan)
colnames(df_nan)
# sum of null values in dataset
sum(is.na(df_nan))
#sum of null values columnwise in dataset
sapply(df_nan, function(x) sum(is.na(x)))

#handling missing values remove all NaN values
df_nan=df_nan[complete.cases(df_nan), ]
sum(is.na(df_nan))
dim(df_nan)
head(df_nan)

df_nan$employment_type<- factor(df_nan$employment_type)
df_nan$required_experience<- factor(df_nan$required_experience)
df_nan$required_education<- factor(df_nan$required_education)
df_nan$function.<- factor(df_nan$function.)
df_nan$industry<- factor(df_nan$industry)


# no of categorical coumns in dataframe
factors_nan <- names(which(sapply(df_nan, is.factor)))
factors_nan

# Label Encoder
for (i in factors_nan){
  encode <- LabelEncoder.fit(df_nan[, i])
  df_nan[, i] <- transform(encode, df_nan[, i])
}
sum(is.na(df_nan))
dim(df_nan)
head(df_nan)  


#Correlarion plot
x=cor(df_nan, use = "complete.obs")
corrplot(x, method="color")
findCorrelation(
  x,
  cutoff = 0.9,
  verbose = FALSE,
  names = TRUE
)
# There is no columns which are highly correlated


# coerce target var to factor
df_nan$fraudulent<- factor(df_nan$fraudulent)
levels(df_nan$fraudulent)<- c("not_fake","fake")


table(df_nan$fraudulent) # 349 fake job postings
df_nan %>%
  ggplot(aes(x=fraudulent))+
  geom_bar()+
  ggtitle("Imbalanced class distribution")+
  labs(x="fraudulent", y="failure count")+
  theme_bw()+
  theme(panel.border = element_rect(colour = "black", fill=NA, size=1))



#PREDICTIVE MODELLING ON IMBALANCED TRAINING DATA
#Splitting data into train and test date
require(caTools)  
set.seed(123)
sample_nan = sample.split(df_nan,SplitRatio = 0.8)
train_data_nan =subset(df_nan,sample_nan ==TRUE)
test_data_nan=subset(df_nan, sample_nan==FALSE)


#get details
str(train_data_nan)
str(test_data_nan)

#dimention
dim(train_data_nan)
dim(test_data_nan)

validation_nan=test_data_nan$fraudulent
test_data_nan=test_data_nan[,c(1:8)]

#list the levels for the class label
levels(train_data_nan$fraudulent)
#we can see there are two levels, making it a binary classification problem

#statistical summary
summary(train_data_nan)
str(train_data_nan)
str(test_data_nan)


#Prediction algorithms before any pre processing
#1.Set-up the test harness to use 10-fold  cross validation.
#2.Build 5 different models to predict.
#3.Select the best model.
#We will use 10-fold cross validation to estimate accuracy.
#This will split our dataset into 10 parts, train in 9 and test on 1 and release for all combinations of train-test splits. We will also repeat the process 3 times for each algorithm with different splits of the data into 10 groups, in an effort to get a more accurate estimate.

#Run algorithms using 10-fold cross validation
control<-trainControl(method="cv",number=10)
#We are using the metric of “Accuracy” to evaluate models. This is a ratio of the number of correctly predicted instances in divided by the total number of instances in the dataset multiplied by 100 to give a percentage (e.g. 95% accurate). We will be using the metric variable when we run build and evaluate each model next.
metric<-"Accuracy"

#Build models
#We reset the random number seed before reach run to ensure that the evaluation of each algorithm is performed using exactly the same data splits. It ensures the results are directly comparable.
#5 different algorithms


# a)Linear Algorithms 
#1.Linear Discrimant Analysis(LDA)    #0.9474
set.seed(200)
fit.lda_nan<-train(fraudulent~.,data=train_data_nan,method="lda",metric=metric, trControl=control)
## Predicting fraud job postings
LDAPredict_nan <- predict(fit.lda_nan, newdata = test_data_nan )
## Confusion matrix
CM_LDA_nan <- confusionMatrix(LDAPredict_nan, validation_nan )
CM_LDA_nan

#b)Nonlinear Algorithms  # 0.9684 
#1.Classification and Regression Trees(Cart)
set.seed(200)
fit.cart_nan<-train(fraudulent~.,data=train_data_nan,method="rpart",metric=metric, trControl=control)
## Predicting fraud job postings
cartPredict_nan <- predict(fit.cart_nan, newdata = test_data_nan )
## Confusion matrix
CM_cart_nan <- confusionMatrix(cartPredict_nan, validation_nan)
CM_cart_nan

#2. k-Nearest Neighbors(KNN)
set.seed(200)
fit.knn_nan<-train(fraudulent~.,data=train_data_nan,method="knn",metric=metric, trControl=control)
## Predicting fraud job postings
KNNPredict_nan <- predict(fit.knn_nan, newdata = test_data_nan )
## Confusion matrix
CM_KNN_nan<- confusionMatrix(KNNPredict_nan, validation_nan)
CM_KNN_nan

#c) Advanced algorithms

#1. Random Forests
set.seed(200)
control<-trainControl(method="cv",number=10)
fit.rf_nan<-train(fraudulent~.,data=train_data_nan, method="rf",metric=metric, trControl=control)
## Predicting fraud job postings
Predict_rf_nan <- predict(fit.rf_nan, newdata = test_data_nan )
## Confusion matrix
CM_RF_nan <- confusionMatrix(Predict_rf_nan, validation_nan)
CM_RF_nan


#2.Gradient Boosting method

set.seed(200)
fit.gbm_nan<-train(fraudulent~.,data=train_data_nan,method="gbm",metric=metric, trControl=control)
## Predicting fraud job postings
Predict_gbm_nan <- predict(fit.gbm_nan, newdata = test_data_nan )
## Confusion matrix
CM_GBM_nan <- confusionMatrix(Predict_gbm_nan, validation_nan )
CM_GBM_nan

#3. SVM Support Vector Machines(SVM)

#linear kernel

set.seed(200)
fit.svmL_nan<-train(fraudulent~.,data=train_data_nan,method="svmLinear",metric=metric, trControl=control)
## Predicting fraud job postings
Predict_svmL_nan <- predict(fit.svmL_nan, newdata = test_data_nan )
## Confusion matrix
CM_SVML_nan <- confusionMatrix(Predict_svmL_nan, validation_nan)
CM_SVML_nan

#Non-Linear Kernel
set.seed(200)
fit.svmR_nan<-train(fraudulent~.,data=train_data_nan,method="svmRadial",metric=metric, trControl=control)
## Predicting fraud job postings
Predict_svmR_nan <- predict(fit.svmR_nan, newdata = test_data_nan )
## Confusion matrix
CM_SVMR_nan <- confusionMatrix(Predict_svmR_nan, validation_nan )
CM_SVMR_nan

set.seed(200)
fit.svmP_nan<-train(fraudulent~.,data=train_data_nan,method="svmPoly",metric=metric, trControl=control)
## Predicting fraud job postings
Predict_svmP_nan <- predict(fit.svmP_nan, newdata = test_data_nan )
## Confusion matrix
CM_SVMP_nan <- confusionMatrix(Predict_svmP_nan, validation_nan)
CM_SVMP_nan

#summarize accuracy of models

results_nan<-resamples(list(lda=fit.lda_nan, cart=fit.cart_nan, knn=fit.knn_nan, rf=fit.rf_nan, gbm=fit.gbm_nan, svmL=fit.svmL_nan, svmR=fit.svmR_nan, svmP=fit.svmP_nan))
summary(results_nan)

#We can also create a plot of the model evaluation results and compare the spread and the mean accuracy of each model. There is a population of accuracy measures for each algorithm because each algorithm was evaluated 10 times (10 fold cross validation).
dotplot(results_nan)

# From the above predictions and plots we conclude that RF has highest accuracy and highest Kappa-->0.62 under range(0.61–0.80) result can be interpreted as substantial.

#Pre-Processing
#as there are many columns it is time consuming to plot a bell curve or histogram to check the skewness. Hence we use box-cox
#if the accuracy increases after applying box cox we will know that there was skewness
#we use box cox and not Yeo-Johnson transformation as all the values are positive and non-zero.
#If a logarithmic transformation is applied to this distribution, the differences between smaller values will be expanded (because the slope of the logarithmic function is steeper when values are small) whereas the differences between larger values will be reduced (because of the very moderate slope of the log distribution for larger values).

#applying boxcox without removing outliers
train_data_nan1=train_data_nan
preprocessParams_nan1<- preProcess(train_data_nan1[,], method=c("BoxCox"))
print(preprocessParams_nan1)
transformed_nan1<- predict(preprocessParams_nan1, train_data_nan1[,])
summary(transformed_nan1)



df_nan1=df_nan
preprocessParams_nan1<- preProcess(df_nan1[,], method=c("BoxCox"))
print(preprocessParams_nan1)
transformed_nan1<- predict(preprocessParams_nan1, df_nan1[,])
summary(transformed_nan1)

require(caTools)  
set.seed(123)
sample_nan1 = sample.split(transformed_nan1,SplitRatio = 0.8)
train_data_nan1 =subset(transformed_nan1,sample_nan1 ==TRUE)
test_data_nan1=subset(transformed_nan1, sample_nan1==FALSE)
validation_nan1=test_data_nan1$fraudulent
test_data_nan1=test_data_nan1[,c(1:8)]


#Random Forest
set.seed(200)
control<-trainControl(method="cv",number=10)
fit.rf_nan1<-train(fraudulent~.,data=train_data_nan1, method="rf",metric=metric, trControl=control)
## Predicting fraud job postings
Predict_rf_nan1 <- predict(fit.rf_nan1, newdata = test_data_nan1)
## Confusion matrix
CM_RF_nan1 <- confusionMatrix(Predict_rf_nan1, validation_nan1)
CM_RF_nan1

#comparing fit.rf & fit.rf1 
results_nan1<-resamples(list(rf=fit.rf_nan, rf1=fit.rf_nan1))
summary(results_nan1)
dotplot(results_nan1)
#after applying box-cot the accuracy increased slightly so there is very less skewnesss



#The value of center determines how column centering is performed. 
#The value of scale determines how column scaling is performed (after centering).
#applying center,scale on the train data without box cox applied and without removing outliers
train_data_nan2=train_data_nan
preprocessParams_nan2 <- preProcess(train_data_nan2[,], method=c("center","scale"))
print(preprocessParams_nan2)
transformed_nan2 <- predict(preprocessParams_nan2, train_data_nan2[,])
summary(transformed_nan2)

#Random Forest
set.seed(200)
control<-trainControl(method="cv",number=10)
fit.rf_nan2<-train(fraudulent~.,data=transformed_nan2, method="rf",metric=metric, trControl=control)
## Predicting fraud job postings
Predict_rf_nan2 <- predict(fit.rf_nan2, newdata = test_data_nan)
## Confusion matrix
CM_RF_nan2 <- confusionMatrix(Predict_rf_nan2, validation_nan)
CM_RF_nan2

#comparing fit.rf & fit.rf1 
results_nan2<-resamples(list(rf=fit.rf_nan, rf2=fit.rf_nan2))
summary(results_nan2)
dotplot(results_nan2)


#applying centre,scale on the train data with box cox applied and without removing outliers
preprocessParams3 <- preProcess(transformed1[,], method=c("center","scale"))
print(preprocessParams3)
transformed3 <- predict(preprocessParams3, transformed1[,])
summary(transformed3)

#Random Forest
set.seed(200)
control<-trainControl(method="cv",number=10)
fit.rf3<-train(fraudulent~.,data=transformed3, method="rf",metric=metric, trControl=control)
## Predicting fraud job postings
Predict_rf3 <- predict(fit.rf3, newdata = validation)
## Confusion matrix
CM_RF3 <- confusionMatrix(Predict_rf3, validation$fraudulent )
CM_RF3

#comparing fit.rf & fit.rf1 
results3<-resamples(list(rf=fit.rf, rf3=fit.rf3))
summary(results3)
dotplot(results3)



#PREDICTIVE MODELLING On BALANCED DATA

# Method 1: Under Sampling
set.seed(2020)
ctrl <- trainControl(method = "repeatedcv",
                     number = 3,
                     # repeated 3 times
                     repeats = 3, 
                     verboseIter = FALSE, 
                     classProbs=TRUE, 
                     summaryFunction=twoClassSummary,
                     sampling = "down"
)
fit_under<-caret::train(fraudulent~.,data = train_data,
                        method = "rpart",
                        preProcess = c("scale", "center"),
                        trControl = ctrl 
                        ,metric= "ROC"
)
# Method 2: Over Sampling
set.seed(2020)
ctrl <- trainControl(method = "repeatedcv",
                     number = 3,
                     # repeated 3 times
                     repeats = 3, 
                     verboseIter = FALSE, 
                     classProbs=TRUE, 
                     summaryFunction=twoClassSummary,
                     sampling = "up"
)
fit_over<-caret::train(fraudulent ~ .,data = train_data,
                       method = "rpart",
                       preProcess = c("scale", "center"),
                       trControl = ctrl 
                       ,metric= "ROC"
)

# summarize accuracy of models
models <- resamples(list(rpart_under=fit_under, rpart_over=fit_over))
summary(models) # highest sensistivity is for over sampling
bwplot(models)

# Make Predictions using the best model
predictions <- predict(fit_over, test_data)
# Using under-balancing as a method for balancing the data
confusionMatrix(predictions, test_data$fraudulent)



