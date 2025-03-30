利用“括号算法”解决杂交方式中的图谱问题

**Solving the mapping problem in the hybridization method by using the “bracket algorithm”**

# 背景引入

最近写作业的时候遇到这样的问题


![](https://files.mdnice.com/user/64884/d1441eec-4201-4e32-968a-1ce23ee6fb80.png)



而以前或许也只学过AxB这样的


![](https://files.mdnice.com/user/64884/950c9e10-3803-43d1-909f-3332891ac8c4.png)


那么我们该如何从最基础的不断进阶到复杂，甚至通用的"算法"呢。

今天我们就提出了一种用于解决这种问题的“杂交图谱括号算法”。



# 基础知识

首先，了解我们的科学问题。

杂交问题的提出和目的


![](https://files.mdnice.com/user/64884/e6c6faf6-bd26-4dec-a11e-2500c11d56cb.jpg)


杂交育种的概念


![](https://files.mdnice.com/user/64884/c516d3f8-5534-4862-aceb-36bf03fdff2f.png)


杂交方式


![](https://files.mdnice.com/user/64884/549b32b9-ec02-4a36-a4ab-63ddc2f8992c.png)




![](https://files.mdnice.com/user/64884/26074e3f-462f-4fff-a915-62d9601cc680.png)



![](https://files.mdnice.com/user/64884/6f32a5f5-4018-4858-b0be-9b294c5a19fb.png)


![](https://files.mdnice.com/user/64884/57b66bf0-842d-4d98-8d77-7245c4a8553e.png)





![](https://files.mdnice.com/user/64884/ac9f8654-dfd4-48e0-8144-24f33fca1425.png)


![](https://files.mdnice.com/user/64884/6f26ab2f-29f4-40ae-8bdd-d44937fbef29.png)




# 算法介绍




![](https://files.mdnice.com/user/64884/e3f3029c-cd52-4418-939b-dfabba81bdc6.jpg)



括号算法的核心思想是，定位、分割、分组、逆推

1. 定位分割：找到大于等于2的数字做定位，其中2用圆圈标注，大于2的用倒三角标注
2. 集团分组：从头到尾，以倒三角（≥3）为分割点，用中括号分为不同集团，在后面的写法中，就是不同的一小团
3. 括号分组：不同集团内，从头到尾，以圆圈（=2）为分割点，用括号分为不同小组，每个小组就是一次单交组，实现降维处理
4. 数字杂交：以数字1为一次单交，对应写法中的”x"，然后与相邻的圆圈（2）周围的小组二交，再和相邻的集团（≥3）进行复交，并且不断递推，实现升维处理。

The core idea of the bracketed algorithm is to locate, segment, group, and backward push：

1. **positioning segmentation**: find the number greater than or equal to 2 to do positioning, where 2 with a circle labeled, greater than 2 with an inverted triangle labeling
2. **group grouping**: from beginning to end, inverted triangles (≥ 3) as a division point, with parentheses into different groups, in the back of the write-up, is a different group of small groups
3. **bracket grouping**: different groups, from head to tail, with a circle (= 2) as the division point, with brackets into different groups, each group is a single cross group, to achieve the dimensionality reduction process
4. **Digital hybridization**: take the number 1 as a single intersection, corresponding to “x” in the writing method, and then intersect with the neighboring groups around the circle (2), and then intersect with the neighboring groups (≥3), and continuously recursive, to realize the upward dimension processing.



# 问题解决


![](https://files.mdnice.com/user/64884/c44eea93-ea37-4dfa-892f-1f16e9ea45d9.png)




# 写在最后

复杂的问题都有方法。

本方法引用格式：

Wang, Y. (2025). Solving the mapping problem in the hybridization method by using the “bracket algorithm” [Source code]. GitHub. https://github.com/yuhong2024/Learn_AI/tree/main/idea

并联系作者 

(wyhstar@email.swu.edu.cn)



# 参考

部分课件引用

张建奎. (2025). 作物育种学 [课件]. 西南大学超星课程. https://www.xueyinonline.com/detail/200426399



