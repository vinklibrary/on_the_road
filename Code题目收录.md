# Code题目收录

# LeetCode11 盛水最多的容器


```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        area = 0
        left = 0
        right = len(height)-1
        while left < right:
            tmp_area = (right-left)*(height[left] if height[left]<height[right] else height[right])
            area = tmp_area if tmp_area>area else area
            if height[right] > height[left]:
                left+=1
            else:
                right-=1
        return area
```


# LeetCode12 整数转罗马数字
```python
class Solution:
    def intToRoman(self, num: int) -> str:
        hash_map = {
            1000:'M',
            900:'CM',
            500:'D',
            400:'CD',
            100:'C',
            90:'XC',
            50:'L',
            40:'XL',
            10:'X',
            9:'IX',
            5:'V',
            4:'IV',
            1:'I'
        }
        re = ''
        for i in hash_map:
            if num//i>0:
                counts=num//i
                re+=hash_map[i]*counts
            num=num%i
        return re
```




# LeetCode20 有效的括号
```python
class Solution:
    def cmp(self ,a,b):
        if a=="(" and b==")":
            return 1
        elif a=="[" and b=="]":
            return 1
        elif a=="{" and b=="}":
            return 1
        else:
            return 0

    def isValid(self, s: str) -> bool:
        stack = []
        for i in s:
            if len(stack)==0:
                stack.append(i)
            elif self.cmp(stack[-1], i)==0:
                stack.append(i)
            else:
                stack.pop()
        if len(stack)==0:
            return True
        else:
            return False
```


# LeetCode 21 合并两个有序的链表
```python
21. 合并两个有序链表
将两个升序链表合并为一个新的升序链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 
示例：
输入：1->2->4, 1->3->4
输出：1->1->2->3->4->4
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        head = ListNode()
        tmp = head
        while l1 and l2:
            if l1.val < l2.val:
                tmp.next = l1 
                l1 = l1.next
            else:
                tmp.next = l2
                l2 = l2.next
            tmp = tmp.next
        if l1:
            tmp.next = l1
        else:
            tmp.next = l2
        return head.next
```
# LeetCode 23 合并K个有序的链表
```python
合并 k 个排序链表，返回合并后的排序链表。请分析和描述算法的复杂度。
示例:
输入:
[
  1->4->5,
  1->3->4,
  2->6
]
输出: 1->1->2->3->4->4->5->6

# 采用了偷懒的方案，不断合并两个链表。 效率需要进行优化
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        head = ListNode()
        tmp = head
        while l1 and l2:
            if l1.val < l2.val:
                tmp.next = l1 
                l1 = l1.next
            else:
                tmp.next = l2
                l2 = l2.next
            tmp = tmp.next
        if l1:
            tmp.next = l1
        else:
            tmp.next = l2
        return head.next

    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        if len(lists)==1:
            return lists[0]
        if len(lists)==0:
            return None
        result = ListNode()
        result.next = lists[0]
        for i in range(0,len(lists)-1):
            result.next = self.mergeTwoLists(result.next, lists[i+1])
        return result.next
```
# LeetCode46 全排列


```python
# 给定一个 没有重复 数字的序列，返回其所有可能的全排列。
# 采用回溯法
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        state = []
        def back(nums, state):
            if len(nums)==0 :
                res.append(state)
                return
            for i in range(len(nums)):
                back(nums[:i]+nums[i+1:], state+[nums[i]])
        back(nums, state)
        return res
```
# 
# LeetCode73 矩阵置零
```python
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        m = len(matrix)
        n = len(matrix[0])
        rows = []
        cols = []
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 0:
                    if i not in rows:
                        rows.append(i)
                    if j not in cols:
                        cols.append(j)
        for i in rows:
            matrix[i] = [0] * n
        for j in cols:
            for k in range(m):
                matrix[k][j] = 0
        return matrix
```


# LeetCode 78 子集


```python
# 给定一组不含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。
# 说明：解集不能包含重复的子集。

# 采用回溯法，其中state为中间保存的状态。
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = []
        state = []

        def back(nums, state):
            if len(nums) == 0 and state not in res:
                res.append(state)
                
            for i in range(len(nums)):
                back(nums[i+1:], state+[nums[i]])
                back(nums[i+1:], state)
        
        back(nums, state)
        return res
```
# 
# LeetCode 91 解码方法


```python
class Solution:
    def numDecodings(self, s: str) -> int:
        lists = ['1','2','3','4','5','6','7','8','9','10',
                '11','12','13','14','15','16','17','18','19','20',
                '21','22','23','24','25','26']
        if s[0]=='0':return 0
        
        result = [1, 1]

        for i in range(1, len(s)):
            if s[i-1:i+1] in lists and s[i] in lists:
                result.append(result[i]+result[i-1])
            elif s[i-1:i+1] in lists and s[i] not in lists:
                result.append(result[i-1])
            elif s[i-1:i+1] not in lists and s[i] in lists:
                result.append(result[i])
            else:
                return 0
        return result[-1]
```


# LeetCode 98 验证二叉搜索树
```python
自己写的方法，只要验证中序遍历是递增的就好。 当前的中序遍历采用递归实现，最好的话可以改成不用递归实现的版本。
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def __init__(self):
        self.order_array = []

    def isValidBST(self, root: TreeNode) -> bool:
        self.pre_trav(root)
        if len(self.order_array) == 0:
            return True

        for i in range(len(self.order_array)-1):
            if self.order_array[i+1]<=self.order_array[i]:
                return False
        return True

    def pre_trav(self, root: TreeNode):
        if not root:
            return
        if root.left:
            self.pre_trav(root.left)
        self.order_array.append(root.val)
        if root.right:
            self.pre_trav(root.right)
```


# LeetCode 113 路径总和 II
```python
给定一个二叉树和一个目标和，找到所有从根节点到叶子节点路径总和等于给定目标和的路径。

说明: 叶子节点是指没有子节点的节点。

示例:
给定如下二叉树，以及目标和 sum = 22，

              5
             / \
            4   8
           /   / \
          11  13  4
         /  \    / \
        7    2  5   1
返回:

[
   [5,4,11,2],
   [5,8,4,5]
]

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def pathSum(self, root: TreeNode, target: int) -> List[List[int]]:
        res = []
        state = []
        def back(node, state):
            if node is None:
                return
            if sum(state) + node.val == target and node.left is None and node.right is None:
                state.append(node.val)
                res.append(state)
                return 
            back(node.left, state+[node.val])
            back(node.right, state+[node.val])
        back(root, state)
        return res
```


# LeetCode 136 只出现一次的数字
```python

# 解法一、数学
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        num_arr = []
        for num in nums:
            if num not in num_arr:
                num_arr.append(num)
        return sum(num_arr)*2-sum(nums)

# 解法二、位运算 两个相同数字异或为0
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        result = nums[0]
        for i in range(1,len(nums)):
            result = result ^ nums[i]
        return result
```
# LeetCode 137 只出现一次的数字II
```python
给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现了三次。找出那个只出现了一次的元素。
说明：
你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？
示例 1:
输入: [2,2,3,2]
输出: 3
示例 2:
输入: [0,1,0,1,0,1,99]
输出: 99

class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        return int((sum(set(nums))*6 - sum(nums)*2 )/ 4)
    
```
# LeetCode 169 多数元素 简单
```python
给定一个大小为 n 的数组，找到其中的多数元素。多数元素是指在数组中出现次数大于 ⌊ n/2 ⌋ 的元素。
你可以假设数组是非空的，并且给定的数组总是存在多数元素。
示例 1:
输入: [3,2,3]
输出: 3
示例 2:
输入: [2,2,1,1,1,2,2]
输出: 2
    
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        num_dict = {}
        length = len(nums)
        
        for num in nums:
            num_dict.setdefault(num, 0)
            num_dict[num] += 1
            if num_dict[num] > int(length/2):
                return num
        return 0
```


# LeetCode 199 二叉树的右视图 中等
```python
给定一棵二叉树，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。
示例:
输入: [1,2,3,null,5,null,4]
输出: [1, 3, 4]
解释:
   1            <---
 /   \
2     3         <---
 \     \
  5     4       <---

广度优先遍历，选择每一深度最右边的即可。

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def rightSideView(self, root: TreeNode) -> List[int]:
        re = []
        if root is None:
            return re
        queue = deque([(root, 0)])
        result = {}
        max_depth = -1
        while queue:
            node,depth = queue.popleft()
            max_depth = max(depth, max_depth)
            result.setdefault(max_depth, node.val)
            result[max_depth] = node.val
            
            if node.left is not None:
                queue.append((node.left, max_depth+1))
            if node.right is not None:
                queue.append((node.right, max_depth+1))
        for i in range(max_depth+1):
            re.append(result[i])
        return re
```
# Leetcode 200 岛屿数量
```python
给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。

岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。

此外，你可以假设该网格的四条边均被水包围。

示例 1:

输入:
11110
11010
11000
00000
输出: 1
示例 2:

输入:
11000
11000
00100
00011
输出: 3
解释: 每座岛屿只能由水平和/或竖直方向上相邻的陆地连接而成。

class Solution:
    def dfs(self,grid, i, j):
        nr,nc = len(grid), len(grid[0])
        grid[i][j] = '0'
        for x,y in [(i-1,j), (i,j-1),(i+1, j),(i, j+1)]:
            if 0 <= x < nr and 0 <= y < nc and grid[x][y] == "1":
                self.dfs(grid, x, y)

    def numIslands(self, grid: List[List[str]]) -> int:
        nr = len(grid)
        if nr == 0: return 0
        nc = len(grid[0])

        times = 0
        for i in range(nr):
            for j in range(nc):
                if grid[i][j] == '1':
                    times+=1
                    self.dfs(grid, i,j)
        return times 
```
# LeetCode 283 移动0 简单
```python
给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
示例:
输入: [0,1,0,3,12]
输出: [1,3,12,0,0]
说明:
必须在原数组上操作，不能拷贝额外的数组。
尽量减少操作次数。

class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        lastNotZeroIdx = 0
        for i in range(len(nums)):
            if nums[i]!=0:
                nums[lastNotZeroIdx] = nums[i]
                lastNotZeroIdx+=1
        for j in range(lastNotZeroIdx, len(nums)):
            nums[j] = 0
        return nums
```
# LeetCode 322 零钱兑换 中等
```python
给定不同面额的硬币 coins 和一个总金额 amount。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回 -1。
示例 1:
输入: coins = [1, 2, 5], amount = 11
输出: 3 
解释: 11 = 5 + 5 + 1
示例 2:
输入: coins = [2], amount = 3
输出: -1

# 自下而上的动态规划
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        result = [0]*(amount+1)
        for i in range(1, amount+1):
            tmp = []
            for coin in coins:
                if i-coin>=0 and result[i-coin]!=-1:
                    tmp.append(result[i-coin]+1)
            if len(tmp)>0:
                result[i] = min(tmp)
            else:
                result[i] = -1
        return result[amount]
```


# LeetCode 344 反转字符串
```python
class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        return s.reverse()

# 双指针算法，在实际使用中效果逊于直接reverse()
class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        left, right = 0, len(s)-1
        while left < right:
            s[left], s[right] = s[right],s[left]
            left+=1
            right-=1
        return s
```


# LeetCode 461 汉明距离 简单
```python
两个整数之间的汉明距离指的是这两个数字对应二进制位不同的位置的数目。
给出两个整数 x 和 y，计算它们之间的汉明距离。
注意：
0 ≤ x, y < 231.
示例:
输入: x = 1, y = 4
输出: 2
解释:
1   (0 0 0 1)
4   (0 1 0 0)
       ↑   ↑
上面的箭头指出了对应二进制位不同的位置。

class Solution:
    def hammingWeight(self, n):
        count = 0
        while n>0:
            count+=1
            n = n&(n-1)
        return count

    def hammingDistance(self, x: int, y: int) -> int:
        n = x^y
        return self.hammingWeight(n)
```


# LeetCode 463 岛屿的周长 简单
```python
给定一个包含 0 和 1 的二维网格地图，其中 1 表示陆地 0 表示水域。
网格中的格子水平和垂直方向相连（对角线方向不相连）。整个网格被水完全包围，但其中恰好有一个岛屿（或者说，一个或多个表示陆地的格子相连组成的岛屿）。
岛屿中没有“湖”（“湖” 指水域在岛屿内部且不和岛屿周围的水相连）。格子是边长为 1 的正方形。网格为长方形，且宽度和高度均不超过 100 。计算这个岛屿的周长。

示例 :
输入:
[[0,1,0,0],
 [1,1,1,0],
 [0,1,0,0],
 [1,1,0,0]]
输出: 16
    
class Solution:
    def cal_girth(self, grid, i, j):
        tmp = 0
        nr, nc = len(grid), len(grid[0])
        if i==0:
            tmp+=1
        if i==nr-1:
            tmp+=1
        if j==0:
            tmp+=1
        if j==nc-1:
            tmp+=1
        for x,y in [(i-1, j), (i, j-1), (i+1, j), (i, j+1)]:
            if 0<=x<nr and 0<=y<nc and grid[x][y]==0:
                tmp+=1
        return tmp

    def islandPerimeter(self, grid: List[List[int]]) -> int:
        nr = len(grid)
        if nr == 0:return
        nc = len(grid[0])

        girth = 0
        for i in range(nr):
            for j in range(nc):
                if grid[i][j] == 1:
                    girth+=self.cal_girth(grid, i, j)
        return girth
```
# LeetCode 541 反转字符串 II
```python
class Solution:
    def reverseStr(self, s: str, k: int) -> str:
        s = list(s)
        for i in range(0, len(s), k*2):
            s[i:i+k] = reversed(s[i:i+k])
        return ''.join(s)
```


# LeetCode 557 反转字符串中的单词 III
```python
class Solution:
    def reverseWords(self, s: str) -> str:
        s = s.split(" ")
        for i in range(len(s)):
            s[i] = s[i][::-1]
        return " ".join(s)
```


# LeetCode 645 错误的集合
```python
集合 S 包含从1到 n 的整数。不幸的是，因为数据错误，导致集合里面某一个元素复制了成了集合里面的另外一个元素的值，导致集合丢失了一个整数并且有一个元素重复。
给定一个数组 nums 代表了集合 S 发生错误后的结果。你的任务是首先寻找到重复出现的整数，再找到丢失的整数，将它们以数组的形式返回。
示例 1:
输入: nums = [1,2,2,4]
输出: [2,3]
注意:
给定数组的长度范围是 [2, 10000]。
给定的数组是无序的。

class Solution:
    def findErrorNums(self, nums: List[int]) -> List[int]:
        num_dict = {}
        for num in nums:
            if num in num_dict:
                double = num
                break
            else:
                num_dict.setdefault(num, 0)
        b = num - sum(nums) + sum([i for i in range(1,len(nums)+1)])
        return [num, b]
    


```


# LeetCode 709 转化成小写字母
```python
class Solution:
    def toLowerCase(self, str: str) -> str:
        return str.lower()

# 更加高效的办法
class Solution:
    def toLowerCase(self, str: str) -> str:
        re = list(str)
        for i in range(len(str)):
            if str[i]>='A' and str[i]<='Z':
                re[i] = chr(ord(str[i])+32)
        return ''.join(re)
```


# LeetCode 784 字母大小写全排列
```python
# 题目描述
给定一个字符串S，通过将字符串S中的每个字母转变大小写，我们可以获得一个新的字符串。返回所有可能得到的字符串集合。
示例:
输入: S = "a1b2"
输出: ["a1b2", "a1B2", "A1b2", "A1B2"]

输入: S = "3z4"
输出: ["3z4", "3Z4"]

输入: S = "12345"
输出: ["12345"]

class Solution:
    def letterCasePermutation(self, S: str) -> List[str]:
        res = []
        state = ''
        length = len(S)
        def back(S, state):
            if len(state) == length:
                res.append(state)
                return
            for i in range(len(S)):
                if S[i].isdigit():
                    back(S[i+1:], state+S[i])
                else:
                    back(S[i+1:], state + S[i].lower())
                    back(S[i+1:], state + S[i].upper())
        back(S, state)
        return res
   
# 改写成 非递归形式  效率提升了非常多
class Solution:
    def letterCasePermutation(self, S: str) -> List[str]:
        res = ['']
        for i in range(len(S)):
            if S[i].isdigit():
                for j in range(len(res)):
                    res[j] = res[j]+S[i]
            else:
                for j in range(len(res)):
                    res.append(res[j]+S[i].upper())
                    res[j] = res[j]+S[i].lower()
                # for j in range(len(res)):
        return res
```


# LeetCode 1006 笨阶乘
```python
class Solution:
    def clumsy(self, N: int) -> int:
        if N==0: return 0
        elif N==1: return 1
        elif N==2: return 2
        elif N==3: return 6
        else:
            tmp_value = 2*int(N*(N-1)/(N-2))
            while N>=4:
                tmp_value+=-int(N*(N-1)/(N-2))+(N-3)
                N-=4
                print(tmp_value)
            return int(tmp_value-self.clumsy(N))
```


# LeetCode 983 最低票价
```python
在一个火车旅行很受欢迎的国度，你提前一年计划了一些火车旅行。在接下来的一年里，你要旅行的日子将以一个名为 days 的数组给出。每一项是一个从 1 到 365 的整数。
火车票有三种不同的销售方式：
一张为期一天的通行证售价为 costs[0] 美元；
一张为期七天的通行证售价为 costs[1] 美元；
一张为期三十天的通行证售价为 costs[2] 美元。
通行证允许数天无限制的旅行。 例如，如果我们在第 2 天获得一张为期 7 天的通行证，那么我们可以连着旅行 7 天：第 2 天、第 3 天、第 4 天、第 5 天、第 6 天、第 7 天和第 8 天。
返回你想要完成在给定的列表 days 中列出的每一天的旅行所需要的最低消费。

class Solution:
    def mincostTickets(self, days: List[int], costs: List[int]) -> int:
        mincost = [0 for i in range(366)]
        if 366 in days:
            mincost[366-1] = min(costs)
        for i in range(1, 366):
            if 366 - i in days:
                tmp1 = costs[0] + mincost[366 - i]
                if 366 - i + 6 >= 366:
                    tmp2 = costs[1]
                else:
                    tmp2 = costs[1]+ mincost[366 - i + 6]
                
                if 366 - i + 29 >= 366:
                    tmp3 = costs[2]
                else:
                    tmp3 = costs[2] + mincost[366 - i + 29]
                mincost[366 - i -1] = min([tmp1, tmp2, tmp3])
            else:
                mincost[366 - i -1] = mincost[366 - i]
        return mincost[0]
```
# Leetcode 1248 统计优美子数组
```python
给你一个整数数组 nums 和一个整数 k。
如果某个 连续 子数组中恰好有 k 个奇数数字，我们就认为这个子数组是「优美子数组」。
请返回这个数组中「优美子数组」的数目。

 示例 1：
输入：nums = [1,1,2,1,1], k = 3
输出：2
解释：包含 3 个奇数的子数组是 [1,1,2,1] 和 [1,2,1,1] 。
示例 2：
输入：nums = [2,4,6], k = 1
输出：0
解释：数列中不包含任何奇数，所以不存在优美子数组。
示例 3：
输入：nums = [2,2,2,1,2,2,1,2,2,2], k = 2
输出：16

# 结果超时 采用动态规划的思想。其实也浪费了大量的时间。
class Solution:
    def numberOfSubarrays(self, nums: List[int], k: int) -> int:
        matrix = [[0 for i in range(len(nums))] for j in range(len(nums))]
        for i in range(len(nums)):
            for j in range(i,len(nums)):
                if j==i:
                    matrix[i][j] = nums[i]%2
                else:
                    matrix[i][j] = matrix[i][j-1]+nums[j]%2
        times = 0
        for i in range(len(nums)):
            for j in range(i,len(nums)):
                if matrix[i][j] == k:
                    times+=1
        return times

# 本题可以在o(n)的时间复杂度内完成.看解析后第二次通过。
class Solution:
    def numberOfSubarrays(self, nums: List[int], k: int) -> int:
        odd_index = []
        for i,num in enumerate(nums):
            if num%2==1:
                odd_index.append(i)
        times = 0
        print(odd_index)
        for i in range(len(odd_index)):
            if i + k - 1 <len(odd_index):
                times += (odd_index[i]+1 if i==0 else odd_index[i]-odd_index[i-1]) * (len(nums)-odd_index[i+k-1] if i+k==len(odd_index) else odd_index[i+k]-odd_index[i+k-1])
            else:
                return times
        return times
```
# LCP1 猜数字


```python
class Solution:
    def game(self, guess: List[int], answer: List[int]) -> int:
        nums = 0
        for i in range(len(guess)):
            if guess[i]==answer[i]:
                nums+=1
        return nums

class Solution:
    def game(self, guess: List[int], answer: List[int]) -> int:
        nums = 0
        for i in zip(*[guess,answer]):
            if len(set(i))==1:
                nums+=1
        return nums
```


# 剑指Offer 面试题3


```python
class Solution:
    def findRepeatNumber(self, nums: List[int]) -> int:
        dicts = {}
        for num in nums:
            if num in dicts:
                return num
            else:
                dicts[num]=1
```


# 剑指Offer 面试题4


```python
class Solution:
    def findNumberIn2DArray(self, matrix: List[List[int]], target: int) -> bool:
        if len(matrix)==0:
            return False
        rows = 0
        cols = len(matrix[0]) - 1

        while cols >= 0 and rows<len(matrix):
            if matrix[rows][cols] == target:
                return True
            elif matrix[rows][cols] > target:
                cols-=1
            else:
                rows+=1

        return False
```


# 面试题 08.11.硬币
```python
硬币。给定数量不限的硬币，币值为25分、10分、5分和1分，编写代码计算n分有几种表示法。(结果可能会很大，你需要将结果模上1000000007)
示例1:
输入: n = 5
输出：2
解释: 有两种方式可以凑成总金额:
5=5
5=1+1+1+1+1
示例2:
输入: n = 10
输出：4
解释: 有四种方式可以凑成总金额:
10=10
10=5+5
10=5+1+1+1+1+1
10=1+1+1+1+1+1+1+1+1+1
说明：
注意:
你可以假设：
0 <= n (总金额) <= 1000000

class Solution:
    def waysToChange(self, n: int) -> int:
        if n == 0: return 0
        if n == 1: return 1
        result= [1]*(n+1)
        for coin in [5, 10, 25]:
            if coin<=n:
                for i in range(coin, n+1):
                    result[i] = result[i-coin] + result[i]
        return result[n] % 1000000007
```
# 面试题 15 二进制中1的个数
```python

class Solution:
    def hammingWeight(self, n: int) -> int:
        count = 0
        while n > 0:
            n = n&(n-1)
            count+=1
        return count
```
# 面试题56 - I. 数组中数字出现的次数
```python
一个整型数组 nums 里除两个数字之外，其他数字都出现了两次。请写程序找出这两个只出现一次的数字。要求时间复杂度是O(n)，空间复杂度是O(1)。
示例 1：
输入：nums = [4,1,4,6]
输出：[1,6] 或 [6,1]
示例 2：
输入：nums = [1,2,10,4,1,4,3,3]
输出：[2,10] 或 [10,2]
限制：2 <= nums <= 10000

class Solution:
    def singleNumbers(self, nums: List[int]) -> List[int]:
        result = nums[0]
        for i in range(1, len(nums)):
            result^=nums[i]
        h = 1
        while result & h == 0:
            h = h<<1
        a = 0
        b = 0
        for i in range(0, len(nums)):
            if nums[i] & h == 0:
                a ^= nums[i]
            else:
                b ^= nums[i]
        return [a, b]
```
