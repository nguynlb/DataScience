df <- c(8.24, 8.25, 8.2, 8.24, 8.21, 8.26, 8.26, 8.2, 8.25, 8.23, 8.19, 8.28, 8.24)
muy <- mean(df)
sd <- sd(df)
SE <- sd/sqrt(14)

t <- (muy - 8.24) / SE
pt(q=t, df=13, lower.tail = FALSE)
