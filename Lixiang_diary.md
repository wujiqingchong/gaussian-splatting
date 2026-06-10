# 协方差（高斯球形状）
在 3DGS 中，高斯球的形状通过一个 3x3 的*对称正定矩阵*来表示。
代码采用了数学上非常经典的Cholesky 分解形式来构建这个矩阵。

协方差矩阵 $\Sigma$ 必须是半正定对称矩阵，否则它无法代表一个合法的空间概率分布。<br>
为了保证 $\Sigma$ 始终是合法的，3DGS 不直接训练 $\Sigma$ 的 9 个参数，而是将其分解为：<br>
$$\Sigma = R \cdot S \cdot S^T \cdot R^T = (RS) \cdot (RS)^T = L \cdot L^T$$
$S$ (Scaling)：一个对角矩阵，代表椭球在 X, Y, Z 三个轴上的拉伸程度。<br>
$R$ (Rotation)：一个旋转矩阵，代表椭球在空间中的朝向。<br>
$L$ (Cholesky Factor)：代码中的 $L$ 即为 $R \cdot S$ 的乘积。<br>

# 四元数与旋转矩阵
如果有一个单位四元数 $q = w + xi + yj + zk$，它对应的旋转矩阵 $R$ 为：$$R = \begin{bmatrix} 
1 - 2(y^2 + z^2) & 2(xy - wz) & 2(xz + wy) \\
2(xy + wz) & 1 - 2(x^2 + z^2) & 2(yz - wx) \\
2(xz - wy) & 2(yz + wx) & 1 - 2(x^2 + y^2)
\end{bmatrix}$$

