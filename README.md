
install.packages(c("tm", "caret"))
library(tm)
library(caret)
data_dir <- "https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset"
emails <- read.csv(file.path(data_dir, "spam.csv"), stringsAsFactors = FALSE)
emails_corpus <- Corpus(VectorSource(emails$text))
emails_corpus <- tm_map(emails_corpus, content_transformer(tolower))
emails_corpus <- tm_map(emails_corpus, removePunctuation)
emails_corpus <- tm_map(emails_corpus, removeNumbers)
emails_corpus <- tm_map(emails_corpus, removeWords, stopwords("en"))
emails_corpus <- tm_map(emails_corpus, stripWhitespace)
dtm <- DocumentTermMatrix(emails_corpus)
set.seed(123)
split_index <- createDataPartition(emails$spam, p = 0.8, list = FALSE)
train_emails <- emails[split_index, ]
test_emails <- emails[-split_index, ]
model <- train(as.factor(spam) ~ ., data = train_emails, method = "naive_bayes")
predictions <- predict(model, newdata = test_emails)
confusionMatrix(predictions, test_emails$spam)
