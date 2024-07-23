import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# 生成示例数据
np.random.seed(0)
cov = np.array([[0.5, 0.2], [0.2, 0.5]])
X1 = np.random.multivariate_normal(mean=[0, 0], cov=cov, size=300)
X2 = np.random.multivariate_normal(mean=[2, 2], cov=cov, size=300)
X3 = np.random.multivariate_normal(mean=[-2, 2], cov=cov, size=300)
X = np.vstack([X1, X2, X3])

# 定义GMM模型
gmm = GaussianMixture(n_components=3, random_state=42)

# 拟合数据
gmm.fit(X)

# 可视化
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = -gmm.score_samples(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap='Blues_r', alpha=0.5)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=gmm.predict(X), s=30, cmap='viridis', edgecolor='k')

plt.title('Gaussian Mixture Model')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()
