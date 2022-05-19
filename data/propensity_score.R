install.packages("BiasedUrn")
library(BiasedUrn)

#Calculate probabilities of being drawn in lottery using noncentral hypergeometric distribution
#Three categories corresponding to number of people in waitlist
N1 <- 57528
N2 <- 17236
N3 <- 158
N <- N1 + N2 + N3
#Total nr of balls sampled
n <- 29834

#P1
m1 <- c(1, N1-1, N2, N3)
odds1 <- c(1, 1, 2, 3)
x1 <- meanMWNCHypergeo(m = m1, n = n, odds = odds1)

#P2
m2 <- c(1, N1, N2-1, N3)
odds2 <- c(2, 1, 2, 3)
x2 <- meanMWNCHypergeo(m = m2, n = n, odds = odds2)

#P3
m3 <- c(1, N1, N2, N3-1)
odds3 <- c(3, 1, 2, 3)
x3 <- meanMWNCHypergeo(m = m3, n = n, odds = odds3)

#Calculated probabilities