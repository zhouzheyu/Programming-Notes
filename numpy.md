# numpy

初始化一个numpy数组：
```
arr = np.array([...], dtype = np.float32)
```

np.random.rand：生成一个随机数组
```
arr = np.random.rand(a, b)
```

数组的变形：
```
arr2 = arr1.reshape()
```

np.matmul：矩阵乘法
```
arr = np.matmul(arr1, arr2)
```

np.transpose：求矩阵转置
```
arr = np.transpose(arr1)
```

断言numpy数组形状，符合则往下执行，否则停住报错：
```
assert(arr.shape == (num1, num2))
```

等间隔生成：
```python
cube_list = [get_cube(center = [0, i, 2], rotation_angles=(0, 30, 50), with_normals=True) for i in np.linspace(-2, 2, 30)]
```

np.sum(arr, axis)求和：
```
arr = np.array([[1, 2, 3], [4, 5, 6]])
# arr.shape = (2, 3)

# 计算沿着 axis=0 的和，即每列的和
sum_along_axis0 = np.sum(arr, axis=0)
# sum_along_axis0.shape = (3, )

# 计算沿着 axis=1 的和，即每行的和
sum_along_axis1 = np.sum(arr, axis=1)
# sum_along_axis1.shape = (2, )
```

* np.max(arr, axis)：求对应轴的最大值
* np.argmax(arr, axis)：求对应轴的最大值的下标
* 操作维度与上述同理

数组拼接np.concatenate：不会创建新轴。axis在哪一维上，就改变哪一维。
```python
arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[7, 8, 9], [10, 11, 12]])
arr = np.concatenate([arr1, arr2], axis = -1)

# arr1.shape = (2, 3)
# arr.shape = (2, 6)

# 当没有指定axis的时候，默认在第0维上拼接
arr = np.concatenate([arr1, arr2])
# arr.shape = (4, 3)
```

数组堆叠np.stack：得到的是更高一维的numpy数组：

```python
# 假设arr1和arr2的形状都是(3, 3)。
# 没指定维度的时候，默认在第0维拼接

arr = np.stack([arr1, arr2]) 
# arr.shape = (2, 3, 3)

A = np.stack([
    [x1 * x2, y1 * x2, x2,
     x1 * y2, y1 * y2, y2,
     x1, y1, 1.0]
    for (x1, y1, z1), (x2, y2, z2) in zip(keypoints1, keypoints2)])
```

np.stack和np.concatenate的混合使用：
``\underbrace{
\begin{bmatrix}
\mathbf{0}^\top & -\tilde{w}_i^{\prime}\tilde{\mathbf{x}}_i^\top & \tilde{y}_i^{\prime}\tilde{\mathbf{x}}_i^\top \\
\tilde{w}_i^{\prime}\tilde{\mathbf{x}}_i^\top & \mathbf{0}^\top & -\tilde{x}_i^{\prime}\tilde{\mathbf{x}}_i^\top \\
-\tilde{y}_i^{\prime}\tilde{\mathbf{x}}_i^\top & \tilde{x}_i^{\prime}\tilde{\mathbf{x}}_i^\top & \mathbf{0}^\top
\end{bmatrix}}_{\mathbf{A}_i}\underbrace{
\begin{bmatrix}
\tilde{\mathbf{h}}_1 \\
\tilde{\mathbf{h}}_2 \\
\tilde{\mathbf{h}}_3
\end{bmatrix}}_{\tilde{\mathbf{h}}}=\mathbf{0}``

代码实现：
```python
def get_Ai(xi_vector, xi_prime_vector):

    assert(xi_vector.shape == (3,) and xi_prime_vector.shape == (3,))

    # Insert your code here
    zero_vector = np.zeros(3, dtype = np.float32)
    xi = xi_prime_vector[0], yi = xi_prime_vector[1], wi = xi_prime_vector[2]
    A_i = np.stack([
        np.concatenate([zero_vector, - wi * xi_vector, yi * xi_vector], axis = -1),
        np.concatenate([wi * xi_vector, zero_vector, - xi * xi_vector], axis = -1)
    ])

    assert(Ai.shape == (2, 9))
    return Ai
```

np.vstack：沿着二维数组的行方向（axis = 0）堆叠元素
```
arr = np.vstack([arr1, arr2])
# arr.shape = (6, 3)
```
np.hstack：沿着二维数组的列方向（axis = 1）堆叠元素
```
arr = np.hstack([arr1, arr2])
# arr.shape = (3, 6)
```
np.dstack：沿着二维数组的列方向（axis = 2）堆叠元素
```
arr = np.dstack([arr1, arr2])
# arr.shape = (3, 3, 2)
```

np.split(arr, indices_or_sections, axis)：定义如何将数组拆分

参数indices_or_sections
* 整数：表示将数组**平均拆分成n个子数组**。
* 1-D 数组：表示**在指定的索引位置拆分数组**。数组中的每个值是一个索引位置，数组会在这些索引处拆分。

np.linalg.svd：矩阵SVD分解
```python
u, s, vh = np.linalg.svd(A)
H = vh[-1].reshape(3, 3)
```

文件的加载和内容读取（.npz）
```
# 获取数据字典
data_dict = np.load(file_path)
# 根据字典索引得到相应属性
points = data_dict['points']
occupancies = data_dict['occupancies']
```

np.meshgrid(x_, y_, z_, indexing="ij")：生成多维网格
```python
def create_voxel_coords_grid(size_x, grid_size, size_y, size_z):
    x_ = np.linspace(-0.5 * size_x, 0.5 * size_x, grid_size)
    y_ = np.linspace(-0.5 * size_y, 0.5 * size_y, grid_size)
    z_ = np.linspace(-0.5 * size_z, 0.5 * size_z, grid_size)
    # x_.shape == y_.shape == z_.shape == (N,)

    x, y, z = np.meshgrid(x_, y_, z_, indexing="ij")
    # x.shape == y.shape == z.shape == (N, N, N)
    # x[i, j, k] = x_[i], y[i, j, k] = y_[j], z[i, j, k] = z_[k]
    assert np.all(x[:, 0, 0] == x_)
    assert np.all(y[0, :, 0] == y_)
    assert np.all(z[0, 0, :] == z_)

    voxel_coordinates = np.stack([x, y, z], axis=-1)
    # voxel_coordinates.shape == (N, N, N, 3)
    # 每个 (i, j, k) 位置上的值是一个三维坐标 [x, y, z]
    return voxel_coordinates
```

np.all(): 用于判断数组中的所有元素是否满足某个条件
```python
输入：一个布尔数组（True / False）；
输出：True 表示所有元素都为 True，否则返回 False。
assert np.all(x[:, 0, 0] == x_)
```

np.linalg.norm(voxel_coordinates, axis=-1, keepdims=True): 求每个点到原点的距离
* axis: 操作的维度
* keepdims: 保留该维度，也就是那个1
```python
def create_artificial_sphere_sdf(voxel_coordinates, radius):
    # (N, N, N, 3) -> (N, N, N, 1)
    voxel_dist_to_center = np.linalg.norm(voxel_coordinates, axis=-1, keepdims=True) 

    # lets have a sdf, where at center of sphere sdf = 1, at border = 0, linear
    
    # BEGIN REGION SOLUTION
    sdf_vals = 1.0 - voxel_dist_to_center / radius

    # END REGION SOLUTION

    assert sdf_vals.shape[:-1] == voxel_coordinates.shape[:-1]
    assert sdf_vals.shape[-1] == 1
    return sdf_vals
```

np.pi, np.sin(), np.cos():
```python
def random_points_on_sphere(radius, num_points, center=np.array([0.0, 0.0, 0.0])):

    theta = np.random.uniform(0, 2 * np.pi, num_points)
    phi = np.random.uniform(0, np.pi, num_points)
    x = radius * np.cos(theta) * np.sin(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(phi)
    points = np.stack([x, y, z], axis = -1) + center

    assert points.shape == (num_points, 3)
    return points

```