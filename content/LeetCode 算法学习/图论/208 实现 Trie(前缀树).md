---
created: '2026-04-25 14:10:11.361961+00:00'
order: 3
pinned: false
summary: https://leetcode.cn/problems/implement-trie-prefix-tree/
title: 208 实现 Trie(前缀树)
updated: '2026-04-25 14:10:11.362022+00:00'
---

# LeetCode 208. 实现 Trie (前缀树)

## 1. 题目描述
**Trie**（发音类似 "try"）或者说 **前缀树** 是一种树形数据结构，用于高效地存储和检索字符串数据集中的键。这一数据结构有相当多的应用情景，例如自动补完和拼写检查。

请你实现 `Trie` 类：
* `Trie()`：初始化前缀树对象。
* `void insert(String word)`：向前缀树中插入字符串 `word` 。
* `boolean search(String word)`：如果字符串 `word` 在前缀树中，返回 `true`（即，在检索之前已经插入）；否则，返回 `false` 。
* `boolean startsWith(String prefix)`：如果之前插入的字符串 `word` 中有一个前缀为 `prefix` ，返回 `true` ；否则，返回 `false` 。

---

### 示例：
* **输入**：
    `["Trie", "insert", "search", "search", "startsWith", "insert", "search"]`
    `[[], ["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]]`
* **输出**：
    `[null, null, true, false, true, null, true]`
* **解释**：
    Trie trie = new Trie();
    trie.insert("apple");
    trie.search("apple");   // 返回 True
    trie.search("app");     // 返回 False
    trie.startsWith("app"); // 返回 True
    trie.insert("app");
    trie.search("app");     // 返回 True

---

## 2. 核心分析
Trie 的核心思想是**空间换时间**，利用字符串的公共前缀来减少无谓的字符串比较。



### 节点结构设计：
每个节点（TrieNode）通常包含两个部分：
1.  **子节点指针**：一个映射或数组（通常大小为 26，对应英文字母），用于指向下一个字符的节点。
2.  **结束标记 (isEnd)**：一个布尔值，用于表示是否有单词以当前字符结尾。

### 操作逻辑：
* **insert**：从根开始，根据单词字符依次向下寻找。如果子节点不存在则创建，最后在末尾节点标记 `isEnd = True`。
* **search**：从根开始查找。如果中途某个字符对应的子节点不存在，返回 `False`；如果查找完所有字符，返回末尾节点的 `isEnd` 状态。
* **startsWith**：逻辑与 `search` 几乎一致，区别在于只要找完了前缀中的所有字符且都存在，就返回 `True`，无需检查 `isEnd`。

---

## 3. Python 代码实现

```python
class TrieNode:
    """前缀树节点结构"""
    def __init__(self):
        # 使用字典存储子节点，key 为字符，value 为下一个 TrieNode
        # 也可以使用长度为 26 的数组: [None] * 26
        self.children = {}
        # 标记是否是一个完整单词的结尾
        self.is_end = False

class Trie:
    def __init__(self):
        """初始化根节点"""
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        """插入单词"""
        node = self.root
        for char in word:
            # 如果当前字符不在子节点中，则创建一个新节点
            if char not in node.children:
                node.children[char] = TrieNode()
            # 移动到子节点
            node = node.children[char]
        # 单词插入完毕，将最后一个节点标记为单词结尾
        node.is_end = True

    def search(self, word: str) -> bool:
        """查找完整单词"""
        node = self.root
        for char in word:
            # 如果中途某个字符不存在，说明单词没存过
            if char not in node.children:
                return False
            node = node.children[char]
        # 搜索完所有字符，必须该位置是单词结尾才算找到
        return node.is_end

    def startsWith(self, prefix: str) -> bool:
        """查找前缀"""
        node = self.root
        for char in prefix:
            # 如果中途某个字符不存在，前缀必不存在
            if char not in node.children:
                return False
            node = node.children[char]
        # 只要能顺利走完前缀的所有字符，就返回 True
        return True
```

---

## 4. 复杂度分析

设 $L$ 为操作字符串的长度，$N$ 为所有插入单词的字符总数。

| 维度 | 复杂度 | 说明 |
| :--- | :--- | :--- |
| **时间复杂度 (Insert)** | $O(L)$ | 遍历一次单词长度。 |
| **时间复杂度 (Search)** | $O(L)$ | 遍历一次单词长度。 |
| **时间复杂度 (StartWith)** | $O(L)$ | 遍历一次前缀长度。 |
| **空间复杂度** | $O(N \times \Sigma)$ | $\Sigma$ 为字符集大小（此处为 26）。最坏情况下（无公共前缀），每个字符都需要一个节点。 |

---

## 5. 重点总结
* **与哈希表的区别**：哈希表只能完整匹配。Trie 不仅能匹配完整单词，还能高效处理**前缀匹配**、**自动补全**等场景。
* **isEnd 的作用**：区分“前缀”和“完整单词”。例如存了 "apple"，如果没有 `is_end`，查询 "app" 也会返回真，这在很多业务场景是不合理的。