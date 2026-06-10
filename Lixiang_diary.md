# 球谐函数
数学来源：球谐函数定义球谐函数 $Y_l^m(\theta, \phi)$ 本质上是由勒让德多项式 (Associated Legendre Polynomials) 乘以一个角度相关的正弦/余弦项构成的：$$Y_l^m(\theta, \phi) = N_l^m \cdot P_l^m(\cos \theta) \cdot e^{im\phi}$$
$N_l^m$：这就是你看到的那些常数（归一化因子），目的是为了确保函数满足 $\int |Y|^2 d\Omega = 1$。$P_l^m$：勒让德多项式，负责定义函数在球面上的波动频率。2. 为什么会有这些具体的浮点数（如 0.28209...）？这些常数是为了在编程实现时消除复杂的开方和阶乘运算。它们是由以下公式推导出来的：$$N_l^m = \sqrt{\frac{2l+1}{4\pi} \frac{(l-m)!}{(l+m)!}}$$当你计算 $l=0, m=0$ 时：$$N_0^0 = \sqrt{\frac{1}{4\pi}} \approx 0.28209479...$$这就是代码里的 C0。同理，计算 $l=1$ 的三个项时：$$N_1^0 = \sqrt{\frac{3}{4\pi}} \approx 0.4886025...$$这就是代码里的 C1。

数学定义的球谐函数（归一化后）在笛卡尔坐标下的表达如下：<br>
![alt text](QQ_1781077907210.png)

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


# dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001
distCUDA2(...)：寻找最近邻距离<br>
这是 3DGS 源码中一个专门优化的 CUDA 函数。它的逻辑大致如下：输入：点云的所有坐标（pcd.points）。计算：对于每一个点，找到距离它最近的 3 个点（$k=3$）。输出：返回一个数组，包含每个点到其最近邻点的距离的平方（$dist^2$）。

为什么要算这个距离？<br>
它是 3D 高斯球“初始大小”的来源：<br>
在 3DGS 初始化时，我们希望高斯球的大小能够适配场景的局部密度：<br>
如果点云分布稀疏，点与点之间距离很远，那么对应的高斯球就应该设置得大一点。<br>
如果点云分布密集，高斯球就应该小一点。<br>