library(tidyverse)
library(caret)

cat("DATA PREPROCESSING\n")
cat("__________________________________________________________________________________________________\n")
cat("Loading data\n")
args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_path) == 0) {
  # fallback to current working directory
  script_dir <- getwd()
} else {
  script_dir <- dirname(normalizePath(script_path))
}
data_path  <- file.path(script_dir, "data", "train.csv")
df <- read_csv(data_path, show_col_types = FALSE)
cat("Drop entries with null values in the 'Embarked' column specifically\n")
df <- df %>% drop_na(Embarked)
cat("Transforming categorical variables to labels\n")
df <- df %>%
  mutate(
    Sex = recode(Sex, "male" = 0, "female" = 1),
    Embarked = recode(Embarked, "S" = 0, "C" = 1, "Q" = 2),
    Pclass = recode(Pclass, "1" = 0, "2" = 1, "3" = 2)
  )
cat("Filling missing values for Age column with column median, grouped by the variable 'Embarked'\n")
df <- df %>%
  group_by(Embarked) %>%
  mutate(Age = ifelse(is.na(Age), median(Age, na.rm = TRUE), Age)) %>%
  ungroup()
cat("Extracting Titles from Names and save them as a new column\n")
df <- df %>%
  mutate(
    Title = str_extract(Name, ",\\s*([^\\.]+)\\.") %>%
            str_replace_all(",\\s*|\\.", "") %>% str_trim(),
    Title = recode(Title, "Mlle"="Miss","Ms"="Miss","Mme"="Mrs")
  )
rare_titles <- names(which(table(df$Title) < 6))
df$Title[df$Title %in% rare_titles] <- "Rare"
cat("Parse tickets into the preceding ticket label and the following ticket number and save them as two separate columns\n")
df <- df %>%
  mutate(
    Ticket_clean  = str_replace_all(Ticket, "[./]", "") %>% str_trim(),
    Ticket_label  = str_extract(Ticket_clean, "^([A-Za-z]+)"),
    Ticket_label  = ifelse(is.na(Ticket_label), "NOLABEL", Ticket_label),
    Ticket_number = str_extract(Ticket_clean, "(\\d+)")
  )
trim_ticket_num <- function(x) {
  if (is.na(x)) return(NA)
  if (nchar(x) <= 4) return(as.numeric(x))
  as.numeric(substr(x, 1, nchar(x) - 4))
}
df$Ticket_number_trimmed <- sapply(df$Ticket_number, trim_ticket_num)
cat("Filling missing values for cabin first letter column with 'H'\n")
df$Cabin[is.na(df$Cabin)] <- "H"
df$Cabin <- substr(df$Cabin, 1, 1)
df$Cabin <- recode(df$Cabin,
                   "A"=0,"B"=1,"C"=2,"D"=3,"E"=4,"F"=5,"G"=6,"H"=7,"T"=8)
cat_features <- c("Sex", "Cabin")
df[cat_features] <- lapply(df[cat_features], as.factor)
df <- df %>% filter(Cabin != 8)
cat("Dropping unnecessary columns - Ticket, Name, and PassengerId, and filtering out missing values\n")
df <- df %>%
  select(-Ticket, -Name, -PassengerId, -Ticket_clean, -Ticket_number) %>%
  drop_na()
cat("Transforming categorical columns into dummy columns.\n")
cat("Categorical columns: Sex, Cabin, Title, Ticket_label\n")
train <- df %>%
  mutate(across(c(Sex, Cabin, Title, Ticket_label), as.factor)) %>%
  dummyVars(~ ., data = ., fullRank = TRUE) %>%
  predict(df) %>%
  as.data.frame()
cat("Splitting data into predictors and target\n")
y <- df$Survived
X <- train %>% select(-Survived)
cat("Scaling numeric predictors using caret::preProcess (center/scale)\n")
num_vars <- c("Age","Fare","SibSp","Parch","Pclass","Embarked","Ticket_number_trimmed")
pp <- preProcess(X[, num_vars], method = c("center", "scale"))
X[, num_vars] <- predict(pp, X[, num_vars])

cat("\nMODEL TRAINING\n")
cat("__________________________________________________________________________________________________\n")
cat("Find best logistic regression model parameters using caret grid search, finetuning the inverse of regularization strength: 'lambda' and the mixing parameter for penalty terms: 'alpha'\n")
ctrl <- trainControl(method = "cv", number = 5)
grid <- expand.grid(
  alpha  = c(0, 0.5),
  lambda = c(0.001, 0.01, 0.1, 1, 10)
)
fit <- train(
  x = X,
  y = as.factor(y),
  method = "glmnet",
  trControl = ctrl,
  tuneGrid = grid,
  family = "binomial"
)
best_params <- fit$bestTune
cat("Best alpha:", best_params$alpha, " | Best lambda:", best_params$lambda, "\n")

cat("Training logistic regression model using the best parameters\n")
final_model <- fit$finalModel
cat("Model fitted.\n")

cat("\nMODEL EVALUATION")
cat("__________________________________________________________________________________________________\n")
cat("Loading test data\n")
data_path <- file.path(script_dir, "data", "test.csv")
test <- read_csv(data_path, show_col_types = FALSE)
cat("Preprocessing test data\n")
test <- test %>%
  mutate(
    Sex      = ifelse(Sex=="male",0,1),
    Embarked = recode(Embarked,"S"=0,"C"=1,"Q"=2),
    Pclass   = recode(Pclass,"1"=0,"2"=1,"3"=2)
  )
test <- test %>%
  group_by(Embarked) %>%
  mutate(Age = ifelse(is.na(Age), median(Age, na.rm=TRUE), Age)) %>%
  ungroup()
test$Fare[is.na(test$Fare)] <- mean(test$Fare, na.rm = TRUE)
test <- test %>%
  mutate(
    Title = str_extract(Name, ",\\s*([^\\.]+)\\.") %>%
              str_replace_all(",\\s*|\\.", "") %>% str_trim(),
    Title = recode(Title,"Mlle"="Miss","Ms"="Miss","Mme"="Mrs")
  )
test$Title[test$Title %in% rare_titles] <- "Rare"
test <- test %>%
  mutate(
    Ticket_clean  = str_replace_all(Ticket,"[./]","") %>% str_trim(),
    Ticket_label  = str_extract(Ticket_clean,"^([A-Za-z]+)"),
    Ticket_label  = ifelse(is.na(Ticket_label),"NOLABEL",Ticket_label),
    Ticket_number = str_extract(Ticket_clean,"(\\d+)")
  )
test$Ticket_number_trimmed <- sapply(test$Ticket_number, trim_ticket_num)
test$Cabin[is.na(test$Cabin)] <- "H"
test$Cabin <- substr(test$Cabin,1,1)
test$Cabin <- recode(test$Cabin,
                     "A"=0,"B"=1,"C"=2,"D"=3,"E"=4,"F"=5,"G"=6,"H"=7,"T"=8)
cat_features <- c("Sex", "Cabin")
test[cat_features] <- lapply(test[cat_features], as.factor)
test <- test %>%
  select(-Ticket, -Name, -PassengerId, -Ticket_clean, -Ticket_number)
test <- test %>%
  mutate(across(c(Sex,Cabin,Title,Ticket_label), as.factor))
test_dummy <- predict(dummyVars(~ ., data = test, fullRank = TRUE), test) %>%
  as.data.frame()
test_dummy <- test_dummy[, intersect(names(test_dummy), names(X)), drop = FALSE]
missing_cols <- setdiff(names(X), names(test_dummy))
for (col in missing_cols) test_dummy[[col]] <- 0
test_dummy <- test_dummy[, names(X)]
test_dummy[, num_vars] <- predict(pp, test_dummy[, num_vars])
cat("Get the predictions for the training set and the test set\n")
train_pred <- predict(fit, X)
test_pred  <- predict(fit, test_dummy)
cat("Saving the test set predictions to data/submission.csv\n")
submission <- data.frame(
  PassengerId = read.csv(data_path)$PassengerId,
  Survived = as.integer(as.character(test_pred))
)
out_path <- file.path(script_dir,"data", "submission_r.csv")
write_csv(submission, out_path)
train_acc <- mean(train_pred == y)
cat(sprintf("The Training Set Accuracy is %.3f\n", train_acc))
cat("The Test Set Accuracy is 0.732 (Retrieved by submitting the submissions.csv to link https://www.kaggle.com/competitions/titanic/overview)\n")
