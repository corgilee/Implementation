#把字典树想象成tree，要先定义一个类似TreeNode 的class

class Node:
    def __init__(self):
        self.children={}
        self.endofword=False

class Trie:
    def __init__(self):
        self.root=Node()

    def insert(self, word: str) -> None:
        cur=self.root
        for c in word:
            if c not in cur.children:
                cur.children[c]=Node() #儿子都要初始化成node
            cur=cur.children[c]
        cur.endofword=True
        
    def search(self, word: str) -> bool:
        cur=self.root
        for c in word:
            if c not in cur.children:
                return False
            cur=cur.children[c]
        return cur.endofword
        

    def startsWith(self, prefix: str) -> bool:
        cur=self.root
        for c in prefix:
            if c not in cur.children:
                return False
            cur=cur.children[c]
        return True
        


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)