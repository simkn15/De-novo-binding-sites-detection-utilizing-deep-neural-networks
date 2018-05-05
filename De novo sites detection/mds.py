import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt
from scipy.spatial import distance
from mpl_toolkits.mplot3d import Axes3D

# Dissimilarity matrix
dimensions = 14
size = 16
dissimilarities = np.zeros((size, size))
dissimilarities.fill(255)
np.fill_diagonal(dissimilarities, 0)
print(dissimilarities)

# Transformation to 3 dimensional space in respect to the dissimilarity matrix
mds = manifold.MDS(dimensions, dissimilarity='precomputed')

pos = mds.fit(dissimilarities).embedding_
# Transformed points in the 3d space
transformedPos = mds.fit_transform(dissimilarities, init=pos)
print(transformedPos)
x = transformedPos[:, 0]
y = transformedPos[:, 1]
z = transformedPos[:, 2]

# # Plot transformed points
# fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# ax = Axes3D(fig)
# ax.scatter(x, y, z, zdir='z', c='red')
# plt.show()
# Between objects distances in one vector (see it as a lower matrix form, by row)
distances = distance.pdist(transformedPos)
print(distances)

# Produce plot of distances
roundDistances = [round(e) for e in distances]
dict = {}
for e in roundDistances:
    if dict.get(e) == None:
        dict[e] = 1
    else:
        dict[e] += 1

# for k in dict.items():
#     print(k)

plt.hist(roundDistances, bins='auto', ec='black')
plt.ylabel('Points')
plt.xlabel('Distances')
plt.title('Distances between ' + str(size) + ' points in ' + str(dimensions) + ' dimensions')
plt.show()

# rng = np.random.RandomState(10)
# a = np.hstack((rng.normal(size=1000),
#                rng.normal(loc=5, scale=2, size=1000)))
# print(a)