## ################################# ##
## EXPLORA DATA                      ##
## TO CONSTRUCT PAIRWISE PROFILE     ##
## ################################# ##
library(dplyr)
library(reshape2)

## ################################## ##
## SET UP ENVIRONMENT                 ##
## ################################## ##
ROOT_PATH = "/Users/beingzy/Documents/Projects/learning_dist_metrics"

## ################################## ##
## CUSTOM FUNCTION                    ##
## ################################## ##
getDataPath <- function(filename, path = NA){
  if( is.na(path) ){
    res <- paste(getwd(), "/", filename, sep="")
  }
  else{
    res <- paste(path, "/", filename, sep = "")
  }
  return(res)
}

isListed <- function(pair, pairs){
  # *********************************** #
  # Check is pair (x, y) having         #
  # equivalent records in pairs (n, 2)  #
  # *********************************** #
  if( length(pair) != ncol(pairs) ){
    stop("Input arguments are not compatiable!")
  }
  res <- apply(pairs
               , MARGIN=1
               , function(x){
                 res <- (x[1] %in% pair) & (x[2] %in% pair)
                 return(res)
               })
  res <- ifelse(sum(res) > 0, TRUE, FALSE)
  return(res)
}

## ###################### ##
## LOAD DATA              ##
## ###################### ##
users   <- read.csv(file=getDataPath("user_profile.csv"), header = TRUE)
friends <- read.csv(file = getDataPath("pairwise_friends.csv"), header = TRUE)

## ############################ ##
## CREATE A PAIRWISE DIFF MATRIX
## 1. set preperty
log <- list()

config           <- list()
config$n_obs     <- 300000
config$n_feats   <- 30
config$var_feats <- c('birthday', 'education_classes_id', 'education_concentration_id', 'education_degree_id'
                      , 'education_school_id', 'education_type', 'education_with_id', 'education_year_id', 'first_name'
                      , 'gender', 'hometown_id', 'languages_id', 'last_name', 'locale', 'location_id', 'middle_name'
                      , 'name', 'political', 'religion', 'work_employer_id', 'work_end_date', 'work_from_id'
                      , 'work_location_id', 'work_position_id', 'work_projects_id', 'work_start_date', 'work_with_id')

udiff_df           <- data.frame(matrix(NA, nrow = config$n_obs, ncol = config$n_feats))
colnames(udiff_df) <- c("class", "uid_a", "uid_b", config$var_feats)

counter <- 1
for( i in 1:nrow(friends) ){
  pair <- friends[i, ]
  a    <- users[users$user_id == pair$x1, config$var_feats][1, ]
  b    <- users[users$user_id == pair$x2, config$var_feats][1, ]
  cat(paste(i, "th, ", "#(a): ", nrow(a), ", #(b): ", nrow(b), "\n", sep=""))
  if( nrow(a) > 0 & nrow(b) > 0 ){
     pair_diff <- abs(a - b)
     udiff_df[counter, ] <- c(1, pair, pair_diff)
     counter <- counter + 1
  }
}

## ################################ ##
## Add unconncted user pairs
## 
all_ids      <- unique( c(friends$x1, friends$x2) )
all_comb_ids <- combn(all_ids, m=2)
all_comb_ids <- t(all_comb_ids)
all_comb_ids <- all_comb_ids[sample(1:nrow(all_comb_ids), nrow(all_comb_ids), replace=FALSE), ]

counter <- sum(udiff_df$class == 1, na.rm=TRUE) + 1
for( i in 1:nrow(all_comb_ids) ){
  pair       <- all_comb_ids[i, ]
  isFriended <- isListed(pair=pair, pairs=friends)
  if( !isFriended ){
    a         <- users[users$user_id == pair[1], config$var_feats][1, ]
    b         <- users[users$user_id == pair[2], config$var_feats][1, ]
    pair_diff <- abs(a - b)
    
    cat(paste(counter, "th, ", "#(a): ", a, ", #(b): ", b, "\n", sep=""))
    
    udiff_df[counter, ] <- c(0, pair, pair_diff)
    counter             <- counter + 1
  }
}

udiff_df$class[1:nrow(friends)] <- 1
udiff_df$class[ (nrow(friends) + 1):(nrow(friends) - sum(is.na(udiff_df$class))) ] <- 0
udiff_df$class <- subset()
write.table(x=udiff_df
            , file=getDataPath("udiff_df.csv", path=ROOT_PATH)
            , col.names = TRUE, row.names = FALSE, sep = ",")
