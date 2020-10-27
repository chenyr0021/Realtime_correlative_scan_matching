# Realtime_correlative_scan_matching

概述：给定两帧雷达数据，source 和 target，遍历所有可能的位姿变换，寻找能够使 source 变为 target 的概率最大的变换。思想类似于极大似然估计。

## 算法简述
#### 1. 将 target 进行栅格化。

- 对连续数值进行离散，用概率值表示每个栅格内出现障碍物的可能性。得到文中的 lookup table，代码中使用的名称是 map。
- 在 multi-level 的方法中，有多种不同分辨率的 map。低分辨率的 lower_map 是由高分辨率 higher_map 进行最大池化得到。
- 根据设置的resolution的数量，对最高分辨率的map进行多级池化，得到图像金字塔。

#### 2. 迭代计算求出最优变换

- 从最低分辨率开始，每一次迭代，取两帧相邻分辨率的map，lower_map & higher_map
- 维护一个在 higher_map 上的最大概率值 H_best，并在 lower_map 上遍历每一个栅格：
    - 若该栅格对应的概率 L_i < H_best, 跳过
    - 否则该栅格中有可能存在概率大于 H_best 的小栅格，则进入High_map继续计算。如果发现更优解就更新 H_best
    - 重复上述过程直整个搜索窗口遍历完成。这时就得到了 higher_map 中的最优值
- higher_map 的栅格的边长就是下一级搜索的窗口，重复这个过程直到遍历至最高分辨率的 map
- 将以上所有计算得到的变换求和，即得到 source -> target 的变换

## 实现细节

- 代码里使用的 target 是用 source 数据通过指定变换（x, y, theta）生成的
- 在对搜索窗口进行变换时，循环从外到内依次为 （theta, x, y）
- map中每个栅格的概率计算公式为：

![](https://github.com/chenyr0021/Realtime_correlative_scan_matching/blob/main/svg.latex.svg)


等式右边为落在该栅格内的点的个数乘以传感器的对数几率，其中p为传感器置信度

等式左边为该栅格的总概率的对数几率，P即为该栅格的概率值

