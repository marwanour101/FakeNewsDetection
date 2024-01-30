#setwd("F:/3rd Year (IS) materials/Statistical Inference/Project")

# Importing the dataset
df = read.csv('news.csv', stringsAsFactors = FALSE) # As Character

# Explore The Data
head(df, n = 5)     # Show first 5 rows
tail(df, n = 5)     # Show last 5 rows
summary(df)         # get summary of the data (numeric, char, char, char)
dim(df)             # dimension of the data -> 6335 obs , 4 variables
str(df)             # structure of Data
# show The Unique Values in Each variable (column) 
length(unique(df$title))    # 6256 -> approximate all values are unique
length(unique(df$text))     # 6060 -> approximate all values are unique
length(unique(df$label))    # 2    -> categorical variable
barplot(sort(table(df$label)), decreasing = TRUE) # show the frequency of each value of the class label
length(unique(df$X))        # 6335

# Cleaning the DataSet
df = df[-1]                     # remove row index column (excluded from features) 
df <- df[!duplicated(df), ]     # Remove duplicate rows -> 29
dim(df)                         # 6306 obs , 3 variables
#remove rows with any NUll values
df <- df[!(is.na(df$text)), ]   # none
df <- df[!(is.na(df$title)), ]  # none
df <- df[!(is.na(df$label)), ]  # none
# Encoding the text (char) variables that will be Features -> text , label
library(tm)             # text mining library
library(SnowballC)      # helper fun in NLP
# cleaning text column
v_text = VCorpus(VectorSource(df$text))  # Deal with it as vector 
v_text = tm_map(v_text, content_transformer(tolower))  # all texts as lower case
v_text = tm_map(v_text, removeNumbers)      # remove numbers: like 1,99,...
v_text = tm_map(v_text, removePunctuation)  # remove punctuation: like .,',?,...
v_text = tm_map(v_text, removeWords, stopwords())  # remove stop words: like the,this,is,... 
v_text = tm_map(v_text, stemDocument)  # get the root of the word: ex. love,loved,loving -> love (reducing the variations)
v_text = tm_map(v_text, stripWhitespace) # remove unnecessary white spaces  
# cleaning title column
v_title = VCorpus(VectorSource(df$title))
v_title = tm_map(v_title, content_transformer(tolower))
v_title = tm_map(v_title, removeNumbers)
v_title = tm_map(v_title, removePunctuation)
v_title = tm_map(v_title, removeWords, stopwords())
v_title = tm_map(v_title, stemDocument)
v_title = tm_map(v_title, stripWhitespace)

# Creating the Bag of Words model
# Sparcing (Encoding) The text column
sparse_text = DocumentTermMatrix(v_text) # Extracting the words with its No.appearance in each observation
dim(sparse_text)    # 6306 obs , 83111 variables (words)
sparse_text = removeSparseTerms(sparse_text, 0.994) # remove less frequent words: more sparse than 0.994
dim(sparse_text)    # 6306 obs , 5087 variables (words)
df_text = as.data.frame(as.matrix(sparse_text))

# Sparcing (Encoding) The title column
sparse_title = DocumentTermMatrix(v_title)
dim(sparse_title)    # 6306 obs , 8192 variables (words)
sparse_title = removeSparseTerms(sparse_title, 0.994)
dim(sparse_title)    # 6306 obs , 183 variables (words)
df_title = as.data.frame(as.matrix(sparse_title))

# create the new Dataset with encoding columns of text and label, and the target (class label)
new_dataset = cbind(df_title, df_text)  
new_dataset$classlabel = df$label       
# Encoding the target feature as factor
new_dataset$classlabel = factor(new_dataset$classlabel, levels = c("FAKE", "REAL"))
summary(new_dataset$classlabel)    # FAKE: 3152 , REAL: 3154 
dim(new_dataset)   # 6306 obs , 5271 variables 
new_dataset <- new_dataset[, !duplicated(colnames(new_dataset))] # remove duplicated column names (words) -> 183
dim(new_dataset)   # 6306 obs , 5088 variables (final Data set after cleaning)

# Splitting into train data (80% of data) and test data (20% of data) 
index <- sample(2, nrow(new_dataset), prob = c(0.8,0.2), replace = TRUE)
train_data <- new_dataset[index == 1,]
test_data <- new_dataset[index == 2,]

# decision tree Model
library(party)
# training phase and constructing the model
dt_model <- ctree(classlabel ~ ., data = train_data) # target: class label, features: remaining variables , Data: train data
print(dt_model)    # show root, internal nodes and responses (decision / classified label)
summary(dt_model)  # Length: 1, class: Binary Tree, mode: S4
# plotting the tree 
dt_plot <- plot(dt_model, type = "simple")
# Evaluate the training process
dt_conf_train <- table(train_data$classlabel, predict(dt_model, train_data[-5088])) # create confusion matrix for label of train data (predicted and actual)
print(dt_conf_train) # TN: 2066, TP: 2068, FP: 475, FN: 433  
dt_train_evaluaion <- sum(diag(dt_conf_train)) / sum(dt_conf_train) # calculate percentage of training (TP + TN) / Total
print(dt_train_evaluaion)   # train evaluation: 0.8199127 
# testing phase
dt_test_prediction <- predict(dt_model, newdata = test_data[-5088]) # target to predict: class label, given features: All, Data: test data
# Evaluate the testing process (accuracy)
dt_conf_test <- table(test_data$classlabel, dt_test_prediction) # create confusion matrix for label of test data (predicted and actual)
print(dt_conf_test) # TN: 506, TP: 543, FP: 105, FN: 110
dt_accuracy <- sum(diag(dt_conf_test)) / sum(dt_conf_test)  # calculate percentage of testing -> accuracy -> (TP + TN) / Total
print(dt_accuracy)  # accuracy: 0.8299051 (Highest accuracy -> Decision Tree is the Best Model for the given data)
# Visualizing Model Performance 
dt_bp = barplot(c(dt_train_evaluaion * 100, dt_accuracy * 100), main="Performance of Decision Tree", names.arg = c("Train Evaluation", "Accuracy"), ylim = c(0,100), ylab = "Percentage", col = c("lightblue", "mistyrose"))
text(dt_bp, 0, round(c(dt_train_evaluaion * 100, dt_accuracy * 100), 2),cex=3,pos=3) 

# Naive Bayes Classifier
library(e1071)
# training phase and constructing the model
NB_model <- naiveBayes(y = train_data$classlabel, x = train_data[-5088], laplace = 0.01) # laplace: for replacing any zero probability with (0.01) this small value 
print(NB_model)    # show the probabilities of each independent feature to be in specific class
summary(NB_model)  # get model specifications: tables length: 5087 mode: list,....
# Evaluate the training process
NB_conf_train <- table(train_data$classlabel, predict(NB_model,train_data[-5088]))  # create confusion matrix for label of train data (predicted and actual)
print(NB_conf_train)  # TN: 2330, TP: 1906, FP: 211, FN: 595
NB_train_evaluaion <- sum(diag(NB_conf_train)) / sum(NB_conf_train)  # calculate percentage of training (TP + TN) / Total
print(NB_train_evaluaion)  # train evaluation: 0.8401428
# testing phase
NB_test_prediction <- predict(NB_model, newdata = test_data[-5088]) 
# Evaluate the testing process (accuracy)
NB_conf_test <- table(test_data$classlabel, NB_test_prediction) # create confusion matrix for label of test data (predicted and actual)
print(NB_conf_test)  # TN: 538, TP: 493, FP: 73, FN: 160
NB_accuracy <- sum(diag(NB_conf_test)) / sum(NB_conf_test) # calculate percentage of testing -> accuracy -> (TP + TN) / Total
print(NB_accuracy) # accuracy: 0.8156646
# Visualizing Model Performance 
NB_bp = barplot(c(NB_train_evaluaion * 100, NB_accuracy * 100), main="Performance of Naive Bayes", names.arg = c("Train Evaluation", "Accuracy"), ylim = c(0,100), ylab = "Percentage", col = c("lightblue", "mistyrose"))
text(NB_bp, 0, round(c(NB_train_evaluaion * 100, NB_accuracy * 100), 2),cex=3,pos=3) 

# logistic regression Model
# Feature Selection: Remove Highly Correlated Variables (Features) as The logistic regression Model is sensitive to them
corr_mat <- cor(new_dataset[-5088]) # get correlation matrix of all variables (Features) with each other                    
print(corr_mat)
mod_cor_mat <- corr_mat   # take a copy of correlation matrix to modify it
mod_cor_mat[upper.tri(mod_cor_mat)] <- 0  # set the upper triangle of correlation matrix to zeroes (preventing duplication) 
diag(mod_cor_mat) <- 0  # set the diagonal of correlation matrix to zeroes (correlation between variable and itself)
print(mod_cor_mat)

# create the new DataSet with variables (Features) less correlated with each other
# select Features has correlation less than or equal to (0.19)
# this is a critical value to get a better model performance
# the higher values of correlation (0.20,...) leads to Overfitting (train gets higher and test gets lower) 
# the lower values of correlation (0.18,...) leads to Underfitting (train gets lower and test gets lower) 
new_dataset_logReg <- new_dataset[ , !apply(mod_cor_mat, 2,function(x) any(x > 0.19))] 
class(new_dataset_logReg$classlabel)   # the target still of type factor
summary(new_dataset_logReg$classlabel) # the factors: FAKE (3152), REAL (3154)

#Splitting the DataSet into train data (80% of DataSet) and test data (20% of DataSet)
index_logReg <- sample(2, nrow(new_dataset_logReg), prob = c(0.8,0.2), replace = TRUE)
train_data_logReg <- new_dataset_logReg[index == 1,]
test_data_logReg <- new_dataset_logReg[index == 2,]

# applying the model on train data 
library(stats)
logitReg_model <- glm(formula = classlabel ~ ., data = train_data_logReg, family = "binomial")
print(logitReg_model)   # show the model parameters like coefficients and degree of dependency (correlation) of class label on each feature (ex. spark : 2.820e+14 , spectacular: -1.302e+15),... 
summary(logitReg_model) # Number of Fisher Scoring iterations: 25,...
# Evaluate the training process
logit_conf_train <- table(train_data_logReg$classlabel, ifelse(predict(logitReg_model, type = 'response', train_data_logReg[-5088]) > 0.5, "REAL", "FAKE")) # create confusion matrix for label of train data (predicted and actual) with prob("REAL") > 0.5 and prob("FAKE") <= 0.5 
print(logit_conf_train) # TN: 2423, TP: 2176, FP: 118, FN: 325
logit_train_evaluaion <- sum(diag(logit_conf_train)) / sum(logit_conf_train) # calculate percentage of training (TP + TN) / Total
print(logit_train_evaluaion)  # train evaluation: 0.912138
# Evaluate the testing process (accuracy)
logit_test_prediction <- ifelse(predict(logitReg_model, type = 'response', test_data_logReg[-5088]) > 0.5, "REAL", "FAKE")  # predict the result (class label) with prob("REAL") > 0.5 and prob("FAKE") <= 0.5 
logit_conf_test <- table(test_data_logReg$classlabel, logit_test_prediction)  # create confusion matrix for label of test data (predicted and actual)
print(logit_conf_test)  # TN: 494, TP: 462, FP: 117, FN: 191
logit_accuracy <- sum(diag(logit_conf_test)) / sum(logit_conf_test)  # calculate percentage of testing -> accuracy -> (TP + TN) / Total
print(logit_accuracy)  # accuracy: 0.7563291  
# Visualizing Model Performance 
logit_bp = barplot(c(logit_train_evaluaion * 100, logit_accuracy * 100), main="Performance of logistic regression", names.arg = c("Train Evaluation", "Accuracy"), ylim = c(0,100), ylab = "Percentage", col = c("lightblue", "mistyrose"))
text(logit_bp, 0, round(c(logit_train_evaluaion * 100, logit_accuracy * 100), 2),cex=3,pos=3) 

# Sum Up Accuracy (All Algorithms)
sumUp_bp = barplot(c(dt_accuracy * 100, NB_accuracy * 100, logit_accuracy * 100), main="Performance of All Algorithms", names.arg = c("Decision Tree Accuracy", "Naive Bayes Accuracy", "logistic regression Accuracy"), ylim = c(0,100), ylab = "Percentage", col = c("lightblue", "mistyrose", "lavender"))
text(sumUp_bp, 0, round(c(dt_accuracy * 100, NB_accuracy * 100, logit_accuracy * 100), 2),cex=3,pos=3) 

