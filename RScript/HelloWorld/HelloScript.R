X <- c(1, 7.3, 2.2, 10 ,3)

curSum <- function(x){
   vector_1 <- rep(c(1), length(x))
   (vector_1 %*% x)[1]
}

curSum(X)
