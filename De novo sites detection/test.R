library(MASS)
library(scatterplot3d) 
library(rgl)

size <- 4
m1 <- matrix(255, nrow=size, ncol=size)
diag(m1) <- 0
m1
mds1 <- isoMDS(dist(m1), k = 3)
mds1$points
x <- mds1$points[,1]
y <- mds1$points[,2]
z <- mds1$points[,3]
scatterplot3d(x, y, z)
dist(mds1$points)
plot3d(x = x, y = y, z = z)
m <- matrix(0, nrow = size, ncol = size)
m[upper.tri(m)] <- dist(mds1$points)
m <- m + t(m) - diag(diag(m))
head(m)

size <- 16
m2 <- matrix(255, nrow = size, ncol = size)
diag(m2) <- 0
mds2 <- isoMDS(dist(m2), k = 3)
mds2
x <- mds2$points[,1]
y <- mds2$points[,2]
z <- mds2$points[,3]
scatterplot3d(x, y, z)
plot3d(x = x, y = y, z = z)
m <- matrix(0, nrow = size, ncol = size)
m[upper.tri(m)] <- dist(mds2$points)
m <- m + t(m) - diag(diag(m))
head(m)

size <- 64
m3 <- matrix(255, nrow = size, ncol = size)
diag(m3) <- 0
m3
mds3 <- isoMDS(dist(m3), k = 3)
mds3$points
x <- mds3$points[,1]
y <- mds3$points[,2]
z <- mds3$points[,3]
# scatterplot3d(x, y, z)
plot3d(mds$points)
m <- matrix(0, nrow = size, ncol = size)
m[upper.tri(m)] <- dist(mds3$points)
m <- m + t(m) - diag(diag(m))
head(m)
