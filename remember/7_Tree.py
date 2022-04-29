from collections import deque


queue = __import__('5_Queue')

################
# Binary Tree
################


class BinaryNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

    def __repr__(self):
        return str(self.value)


class BinaryTree:
    def __init__(self):
        self.root = None

    def inOrderTraversalRecursive(self, current_node):
        if (current_node is not None):
            self.inOrderTraversal(current_node.left)
            print(current_node)
            self.inOrderTraversal(current_node.right)

    def inOrderTraversal(self, current_node):
        stack = []

        while stack or current_node:
            while current_node:
                stack.append(current_node)
                current_node = current_node.left

            current_node = stack.pop()
            # Process node

            current_node = current_node.right

    def preOrderTraversal(self, current_node):
        if (current_node is not None):
            print(current_node)
            self.preOrderTraversal(current_node.left)
            self.preOrderTraversal(current_node.right)

    def postOrderTraversal(self, current_node):
        if (current_node is not None):
            self.postOrderTraversal(current_node.left)
            self.postOrderTraversal(current_node.right)
            print(current_node)

    def levelOrderTraversal(self, root_node):
        node_queue = deque()
        node_queue.push(root_node)

        while node_queue:
            current_node = node_queue.popleft()
            print(current_node)

            if current_node.left:
                node_queue.push(current_node.left)
            if current_node.right:
                node_queue.push(current_node.right)


#######################
# Binary Search Tree
#######################

class BinarySearchTree(BinaryTree):
    def find(self, value):
        current_node = self.root

        while (current_node is not None):
            if value == current_node.value:
                return True

            if value <= current_node.value:
                current_node = current_node.left
            else:
                current_node = current_node.right

        return False

    def insert(self, value):
        if self.root is None:
            self.root = BinaryNode(value)
            return

        current_node = self.root

        while (True):
            if value <= current_node.value:
                # If left is not None, go left. Else create node
                if current_node.left is not None:
                    current_node = current_node.left
                else:
                    current_node.left = BinaryNode(value)
                    return
            else:
                if current_node.right is not None:
                    current_node = current_node.right
                else:
                    current_node.right = BinaryNode(value)
                    return

    def deleteNode(self, target_node, parent_node, replace_node):
        # Case A - target node is not root node
        if target_node.value != self.root.value:
            # If target node is to the left
            if target_node.value <= parent_node.value:
                parent_node.left = replace_node
            else:
                parent_node.right = replace_node

        # Case B - target node is root node
        else:
            self.root = replace_node

    def delete(self, value):
        # Find value with two pointers, current & parent
        parent_node = None
        current_node = self.root

        while (current_node is not None):
            if value == current_node.value:
                break
            elif value < current_node.value:
                parent_node = current_node
                current_node = current_node.left
            else:
                parent_node = current_node
                current_node = current_node.right

        assert current_node is not None, "Value not in tree"

        # Case 1 & 2 - if node is leaf or has one child
        if current_node.left is None or current_node.right is None:
            child_node = current_node.left if current_node.left is not None else current_node.right
            self.deleteNode(current_node, parent_node, child_node)

        # Case 3 - node has two children
        else:
            # Find largest value in left sub-tree
            parent_node = current_node
            replace_node = current_node.left

            while (replace_node.right is not None):
                parent_node = replace_node
                replace_node = replace_node.right

            # Swap values with target node
            current_node.value, replace_node.value = replace_node.value, current_node.value

            # Delete replacement node
            child_node = replace_node.left if replace_node.left is not None else replace_node.right
            self.deleteNode(replace_node, parent_node, child_node)


################
# Trie
################

class TrieNode:
    def __init__(self, char):
        self.char = char
        self.children = {}
        self.is_end = False


class Trie:
    def __init__(self):
        self.root = TrieNode('')

    def insert(self, word: str) -> None:
        node = self.root

        for char in word:
            if char in node.children:
                node = node.children[char]
            else:
                new_node = TrieNode(char)
                node.children[char] = new_node
                node = new_node

        node.is_end = True

    def search(self, word: str) -> bool:
        node = self.root

        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]

        return node.is_end

    def startsWith(self, prefix: str) -> bool:
        node = self.root

        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]

        return True


################
# Main
################

def test_bst():
    bst = BinarySearchTree()
    assert not bst.find(10)
    bst.insert(10)
    assert bst.find(10)
    assert not bst.find(6)
    bst.insert(6)
    assert bst.find(10)
    assert bst.find(6)
    bst.insert(8)
    assert bst.find(10)
    assert bst.find(6)
    assert bst.find(8)
    bst.insert(4)
    bst.insert(20)
    bst.insert(2)

    # bst.delete(10)
    bst.levelOrderTraversal(bst.root)
    # bst.delete(10)


if __name__ == '__main__':
    test_bst()
