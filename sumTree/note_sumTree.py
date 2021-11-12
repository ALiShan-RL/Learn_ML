import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity                        # 这颗树存储的容量
                                                        # tree 中只存放优先级    data中存放状态
        self.tree = np.zeros(2 * capacity - 1)          # 满二叉树的叶子节点是总结点的关系 n2 = n0 - 1  ,因为满二叉树只有度为2和度为0的节点
        # [--------------父亲节点-------------]           [-------叶子节点-------]
        #             个数: capacity - 1                       个数: capacity
        self.data = np.zeros(capacity, dtype=object)    # 存储的大小为capacity , 存储的类型为 object
        # [--------------data frame-------------]
        #             size: capacity

    # 树中替换的策略没有更换，还是先进先出策略
    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1    # 算出新状态加入的位置
        self.data[self.data_pointer] = data                 # 将此位置的状态更新为加入的状态
        self.update(tree_idx, p)                            # 由于树更新了，需要对树整体进行更新

        self.data_pointer += 1                              # 存储状态的指针+1
        if self.data_pointer >= self.capacity:              # 如果指针大于存储的容量则将指针置为0
            self.data_pointer = 0

    # 根据优先级更新树
    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]                    # 新的优先级与老的优先级的差
        self.tree[tree_idx] = p                             # 将此叶节点的优先级换成新的优先级
                                                            # 向其父节点传递，更新父节点的优先级
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2                  # //代表整除,因为数组从0开始所以要先减1
            self.tree[tree_idx] += change                   # 更新

    def get_leaf(self, v):
        """

        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # 父节点的左节点和右节点
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # 如果左节点到达树的极限时结束
                leaf_idx = parent_idx
                break                           # v代表一段切片中的优先级
            else:                               # 向下搜寻，如果v <= 此时的这个节点的优先级，则更新父节点为此节点
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:                           # 如果v大于cl_idx中的优先级   v = v - self.tree[cl_idx],去另一个树节点中寻找，更新另一个节点为父节点
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1  # 计算出状态所在的位置
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # 返回所有优先级总额和
class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # 裁剪绝对值误差上界

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])  # 求堆中最大的优先级
        if max_p == 0:
            max_p = self.abs_err_upper                        # 如果最大的优先级为0，则赋予新状态优先级为 最高的上界1
        self.tree.add(max_p, transition)                      # 将新状态和他的优先级存到堆中

    def sample(self, n):                                      # np.empty()根据给定的维度和数值类型返回一个新的数组，其元素不进行初始化
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.total_p / n                       # 优先级按照n进行切片，分成n段
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1，不能超过1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # 最小的优先级/总优先级
        for i in range(n):                                    # n个切片
            a, b = pri_seg * i, pri_seg * (i + 1)             # a = 切片优先级 * i ||  b = 切片优先级 * (i+1)
            v = np.random.uniform(a, b)                       # 在[a,b]有限级中选取一个v
            idx, p, data = self.tree.get_leaf(v)              # 得到符合要求的状态
            prob = p / self.tree.total_p                      # 得到状态优先级的prob
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)  # IS权重
            b_idx[i], b_memory[i, :] = idx, data              # 存储节点与状态值
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):                   # 经验池更新
        abs_errors += self.epsilon                                  # 避免有限级为0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper) # 裁剪优先级，有限级也是TD_error,这个优先级是一个数列，有多个数值，所以要用到minimum，将所有大于1的值变成1
        ps = np.power(clipped_errors, self.alpha)                   # clipped_errors的self.alpha次方 ,对应于文章中的P(i)
        for ti, p in zip(tree_idx, ps):                             # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
            self.tree.update(ti, p)
