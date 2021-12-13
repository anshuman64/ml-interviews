##################
## ArrayList
##################

class ArrayList:
    def __init__(self):
        self.max_size = 10
        self.list = [None] * self.max_size
        self.count = 0

    def __len__(self):
        """
        Time: O(1)
        """
        return self.count

    def __getitem__(self, index):
        """
        Time: O(1)
        """
        assert index < self.count, "Not enough elements"

        return self.list[index]

    def __repr__(self):
        return str(self.list[:self.count])

    def increase_size(self):
        """
        Time: O(n)
        """
        self.max_size *= 2
        new_list = [None] * self.max_size

        for i in range(self.count):
            new_list[i] = self.list[i]

        self.list = new_list
        
    def insert(self, value):
        """
        Time: 
            Average: O(1)
            Worst: O(n) - doubling
        """
        if self.count >= self.max_size:
            self.increase_size()

        self.list[self.count] = value
        self.count += 1

    def delete(self, index):
        """
        Time: 
            Average: O(n)
            Worst: O(n) - doubling
        """
        assert index < self.count, "Not enough elements"

        for i in range(index, self.count):
            self.list[i] = self.list[i+1]

        self.count -= 1

    def test_arrayList(self):
        a = ArrayList()
        print(a)
        a.insert(1)
        print(a)
        a.insert(2)
        print(a)
        a.insert(3)
        print(a)
        a.delete(0)
        print(a)
        a.insert(4)
        print(a)


##################
## Singly-Linked List
##################

class SingleNode: 
    def __init__(self, value):
        self.value = value
        self.next = None

class SingleLinkedList:
    def __init__(self):
        self.head = None
        self.length = 0

    def __len__(self):
        """
        Time: O(1)
        """
        return self.length

    def __getitem__(self, index):
        """
        Time: O(K) - 1 < K < N
        """
        assert index < self.length, "Element doesn't exist"
        current_node = self.head

        for i in range(index):
            current_node = current_node.next 
        
        return current_node.value

    def get_idx(self, value):
        assert self.length > 0, "List is empty"

        current_node = self.head 
        idx = 0

        while(current_node is not None):
            if current_node.value == value:
                return idx
            else:
                current_node = current_node.next 
                idx += 1

        return -1

    def __repr__(self):
        """
        Time: O(N)
        """
        if self.head is None:
            return '[ ]'
        
        return_string = '[ ' + str(self.head.value)
        current_node = self.head 

        while(current_node.next is not None):
            return_string += ' '
            current_node = current_node.next
            return_string += str(current_node.value)

        return return_string + ' ]'

    def insertAt(self, value, index):
        """
        Time: O(K) - 1 < K < N
        """
        assert index <= self.length, "Element doesn't exist"
        new_node = SingleNode(value)
        
        if (self.head is None):
            self.head = new_node
        elif (index == 0):
            new_node.next = self.head 
            self.head = new_node
        else:
            current_node = self.head 

            # Go to node before new node
            for i in range(index-1):
                current_node = current_node.next

            new_node.next = current_node.next
            current_node.next = new_node

        self.length += 1
            
    def insert(self, value):
        """
        Time: O(N)
        """
        self.insertAt(value, self.length)

    def deleteAt(self, index):
        """
        Time: O(K) - 1 < K < N
        """
        assert index < self.length, "Element doesn't exist"

        if (index == 0):
            self.head = self.head.next 
        else:
            current_node = self.head 

            # Go to node before node-to-delete
            for i in range(index-1):
                current_node = current_node.next

            current_node.next = current_node.next.next
        
        self.length -= 1

    def delete(self, value):
        idx = self.get_idx(value)
        assert idx > -1, "Value not found"
        self.deleteAt(idx)

    def test_linkedlist(self):
        l = SingleLinkedList()
        assert len(l) == 0
        print(l)

        l.insert(3)
        assert len(l) == 1
        assert l.head.value == 3
        print(l)

        l.insert(5)
        assert len(l) == 2
        assert l.head.value == 3
        assert l.head.next.value == 5
        assert l[0] == 3
        assert l[1] == 5
        print(l)

        l.insertAt(1, 1)
        l.insertAt(0,0)
        l.insertAt(7, 4)
        print(l)

        l.deleteAt(0)
        print(l)

        l.deleteAt(3)
        print(l)

        l.delete(5)
        print(l)

        l.delete(7)
        print(l)

        l.delete(3)
        print(l)


##################
## Doubly-Linked List
##################

class DoubleNode: 
    def __init__(self, value):
        self.value = value
        self.next = None
        self.previous = None

class DoubleLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.length = 0

    def __len__(self):
        """
        Time: O(1)
        """
        return self.length

    def __getitem__(self, index):
        """
        Time: 
            Worst: O(N/2)
            Best:  O(1)
        """
        assert index < self.length, "Element doesn't exist"
        current_node = None

        if (index < self.length / 2.0):
            current_node = self.head

            for i in range(index):
                current_node = current_node.next 
        else:
            current_node = self.tail

            for i in range(self.length - index - 1):
                current_node = current_node.previous 
            
        return current_node.value

    def get_idx(self, value):
        assert self.length > 0, "List is empty"

        current_node = self.head 
        idx = 0

        while(current_node is not None):
            if current_node.value == value or (type(current_node.value) is tuple and current_node.value[0] == value):
                return idx
            else:
                current_node = current_node.next 
                idx += 1

        return -1

    def __repr__(self):
        if self.head is None:
            return '[ ]'
        
        return_string = '[ ' + str(self.head.value)
        current_node = self.head 

        while(current_node.next is not None):
            return_string += ' '
            current_node = current_node.next
            return_string += str(current_node.value)

        return return_string + ' ]'

    def insertAt(self, value, index):
        """
        Time: 
            Worst: O(N/2)
            Best:  O(1)
        """
        assert index <= self.length, "Element doesn't exist"
        new_node = DoubleNode(value)

        if (self.head is None):
            self.head = new_node 
            self.tail = self.head
        elif (index == self.length):
            new_node.previous = self.tail 
            self.tail.next = new_node 
            self.tail = new_node
        elif (index == 0):
            new_node.next = self.head 
            self.head.previous = new_node
            self.head = new_node
        else:
            current_node = None
            if (index < self.length / 2.0):
                current_node = self.head 

                for i in range(index-1):
                    current_node = current_node.next
            else:
                current_node = self.tail

                for i in range(self.length - index):
                    current_node = current_node.previous 

            new_node.next = current_node.next
            new_node.previous = current_node
            
            if (current_node.next is not None):
                current_node.next.previous = new_node
            current_node.next = new_node

        self.length += 1
            
    def insert(self, value):
        """
        Time: O(1)
        """
        self.insertAt(value, self.length)

    def deleteAt(self, index):
        """
        Time: 
            Worst: O(N/2)
            Best:  O(1)
        """
        assert index < self.length, "Element doesn't exist"

        if (self.length == 1):
            self.head = None 
            self.tail = None 
        elif (index == 0):
            self.head = self.head.next 
            self.head.previous = None
        elif (index == self.length - 1):
            self.tail = self.tail.previous
            self.tail.next = None
        else:
            current_node = None
            if (index < self.length / 2.0):
                current_node = self.head 

                for i in range(index):
                    current_node = current_node.next
            else:
                current_node = self.tail

                for i in range(self.length - index - 1):
                    current_node = current_node.previous 
            
            if (current_node.next is not None):
                current_node.next.previous = current_node.previous
            if (current_node.previous is not None):
                current_node.previous.next = current_node.next

        self.length -= 1

    def delete(self, value):
        idx = self.get_idx(value)
        assert idx > -1, "Value not found"
        self.deleteAt(idx)


##################
## Stack
##################

class ListStack:
    def __init__(self):
        self.items = []

    def __len__(self):
        """
        Time: O(1)
        """
        return len(self.items)

    def push(self, item):
        """
        Time: O(1)
        """
        self.items.append(item)

    def pop(self):
        """
        Time: O(1)
        """
        assert len(self.items) > 0, "No items in stack"
        return self.items.pop()

    def peek(self):
        """
        Time: O(1)
        """
        assert len(self.items) > 0, "No items in stack"
        return self.items[len(self.items) - 1]

    def isEmpty(self):
        return len(self.items) == 0

class Stack:
    def __init__(self):
        self.top = None 
        self.length = 0

    def __len__(self):
        """
        Time: O(1)
        """
        return self.length
    
    def push(self, item):
        """
        Time: O(1)
        """
        new_node = SingleNode(item)
        
        if (self.top is None):
            self.top = new_node
        else:
            new_node.next = self.top 
            self.top = new_node
        
        self.length += 1

    def pop(self):
        """
        Time: O(1)
        """
        assert self.length > 0, "No items in stack"
        
        return_node = self.top 
        self.top = self.top.next 
        self.length -= 1

        return return_node.value 

    def peek(self):
        """
        Time: O(1)
        """
        assert self.length > 0, "No items in stack"
        return self.top.value

    def isEmpty(self):
        return self.length == 0

def test_stack(self):
    l = Stack()
    assert len(l) == 0

    l.push(3)
    assert l.peek() == 3
    assert len(l) == 1

    l.push(5)
    assert l.peek() == 5
    assert len(l) == 2

    assert l.pop() == 5
    assert len(l) == 1

    assert l.pop() == 3
    assert len(l) == 0

    l.peek()
    l.pop()        


##################
## Queue
##################

class ListQueue:
    def __init__(self):
        self.items = []

    def __len__(self):
        """
        Time: O(1)
        """
        return len(self.items)

    def push(self, item):
        """
        Time: O(1)
        """
        self.items.append(item)

    def pop(self):
        """
        Time: O(1)
        """
        assert len(self.items) > 0, "No items in queue"
        return self.items.pop(0)

    def peek(self):
        """
        Time: O(1)
        """
        assert len(self.items) > 0, "No items in queue"
        return self.items[0]

    def isEmpty(self):
        return len(self.items) == 0

class Queue:
    def __init__(self):
        self.first = None 
        self.last = None
        self.length = 0

    def __len__(self):
        """
        Time: O(1)
        """
        return self.length
    
    def push(self, item):
        """
        Time: O(1)
        """
        new_node = SingleNode(item)
        if (self.first is None):
            self.first = new_node
            self.last  = self.first
        else:
            self.last.next = new_node
            self.last = new_node
        
        self.length += 1

    def pop(self):
        """
        Time: O(1)
        """
        assert self.length > 0, "No items in queue"
        
        return_node = self.first 
        self.first = self.first.next
        self.length -= 1

        if (self.length == 0):
            self.last = None

        return return_node.value 

    def peek(self):
        """
        Time: O(1)
        """
        assert self.length > 0, "No items in queue"
        return self.first.value

    def isEmpty(self):
        return self.length == 0

def test_queue(self):
    l = Queue()
    assert len(l) == 0

    l.push(3)
    assert l.peek() == 3
    assert len(l) == 1

    l.push(5)
    assert l.peek() == 3
    assert len(l) == 2

    assert l.pop() == 3
    assert len(l) == 1

    assert l.pop() == 5
    assert len(l) == 0

    l.peek()
    l.pop()        


##################
## HashTable
##################    

class HashTable:
    def __init__(self, size):
        self.table = [DoubleLinkedList() for i in range(size)]
        self.length = 0

    def __len__(self):
        return self.length 

    def hash(self, key):
        return key % len(self.table)

    def insert(self, key, value):
        """
        Time: 
            Worst:   O(N)
            Best:    O(1)
            Average: O(1)
        """
        hash_idx = hash(key)

        self.table[hash_idx].insert((key, value))
        self.length += 1

    def delete(self, key):
        """
        Time: 
            Worst:   O(N)
            Best:    O(1)
            Average: O(1)
        """
        hash_idx = hash(key)

        self.table[hash_idx].delete(key)
        self.length -= 1

    def get(self, key):
        """
        Time: 
            Worst:   O(N)
            Best:    O(1)
            Average: O(1)
        """
        hash_idx = hash(key)

        key_idx = self.table[hash_idx].get_idx(key)

        return self.table[hash_idx][key_idx][1]

def test_hashtable(self):
    h = HashTable(10)

    h.insert(1, 'hi')
    h.insert(2, 'bye')

    assert h.get(1) == 'hi'
    assert h.get(2) == 'bye'

    h.delete(1)
    h.delete(1)

##################
## Tree
##################  

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
    
    def inOrderTraversal(self, current_node):
        """
        Time: O(N)
        """
        if (current_node is not None):
            self.inOrderTraversal(current_node.left)
            print(current_node)
            self.inOrderTraversal(current_node.right)

    def preOrderTraversal(self, current_node):
        """
        Time: 
            Worst:   O(N)
            Best:    O(logN) - balanced
            Average: O(N)
        """
        if (current_node is not None):
            print(current_node)
            self.preOrderTraversal(current_node.left)
            self.preOrderTraversal(current_node.right)
    
    def postOrderTraversal(self, current_node):
        """
        Time: 
            Worst:   O(N)
            Best:    O(logN) - balanced
            Average: O(N)
        """
        if (current_node is not None):
            self.postOrderTraversal(current_node.left)
            self.postOrderTraversal(current_node.right)
            print(current_node)

    def levelOrderTraversal(self, root_node):
        """
        Time: 
            Worst:   O(N)
            Best:    O(logN) - balanced
            Average: O(N)
        """
        node_queue = Queue()
        node_queue.push(root_node)

        while(not node_queue.isEmpty()):
            current_node = node_queue.pop()
            print(current_node)

            if current_node.left:
                node_queue.push(current_node.left)
            if current_node.right:
                node_queue.push(current_node.right)
                

class BinarySearchTree(BinaryTree):
    def find(self, value):
        """
        Time: 
            Worst:   O(N)
            Best:    O(logN)
            Average: O(logN)
        """
        current_node = self.root

        while(current_node is not None):
            if value == current_node.value:
                return True
            
            if value <= current_node.value:
                current_node = current_node.left
            else:
                current_node = current_node.right
        
        return False

    def insert(self, value):
        """
        Time: O(logN)
        """
        if self.root is None:
            self.root = BinaryNode(value)
            return 
        
        current_node = self.root 

        while(True):
            if value <= current_node.value:
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

    def deleteNode(self, current_node, parent_node, replace_node):
        # Case A - target node is not root node
        if current_node.value != self.root.value:
            if current_node.value <= parent_node.value:
                parent_node.left = replace_node
            else:
                parent_node.right = replace_node
        # Case B - target node is root node
        else:
            self.root = replace_node

    
    def delete(self, value):
        """
        Time: O(logN)
        """
        # Find value with two pointers
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

            while(replace_node.right is not None):
                parent_node = replace_node
                replace_node = replace_node.right

            # Swap values with node to delete
            temp = current_node.value
            current_node.value = replace_node.value
            replace_node.value = temp
            
            # Delete that node
            child_node = replace_node.left if replace_node.left is not None else replace_node.right
            self.deleteNode(replace_node, parent_node, child_node)

    def test_bst(self):
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

##################
## Heap
##################    

class MaxHeap:
    def __init__(self, max_size):
        """
        root = heap[0]
        bottom-most, right-most node = heap[count-1]
        bottom-most, left-most empty spot = heap[count]
        node.left = heap[2i+1]
        node.right = heap[2i+2]
        child.parent = heap[(i-1)/2]
        """
        self.heap = [None] * max_size
        self.count = 0

    def insert(self, value):
        """
        Time: 
            Worst - O(logN)
        """
        # Insert in bottom-most, left-most empty spot
        self.heap[self.count] = value
        self.count += 1

        # Reheapification - swap values with parent until in right place
        i = self.count - 1
        while (True):
            # Change this line for MinHeap
            if i == 0 or self.heap[i] <= self.heap[(i-1)//2]:
                break
            else:
                temp = self.heap[i]
                self.heap[i] = self.heap[(i-1)//2]
                self.heap[(i-1)//2] = temp

                i = (i-1)//2

    def retrieve(self):
        """
        Time: 
            Worst - O(logN)
        """
        assert self.count > 0, "No items in heap"
        return_value = self.heap[0]

        if self.count == 1:
            self.count -= 1
            return return_value
        else:
            # Replace with bottom-most, right-most node
            self.heap[0] = self.heap[self.count-1]
            self.count -= 1

            # Sift down - swap values with children until right place
            i = 0

            while(True):
                left_idx = 2*i+1
                right_idx = 2*i+2

                # Find larger of two children
                if left_idx > self.count:
                    break
                elif right_idx > self.count:
                    child_idx = left_idx
                else:
                    child_idx = left_idx if self.heap[left_idx] > self.heap[right_idx] else right_idx

                # Swap with parent
                temp = self.heap[i]
                self.heap[i] = self.heap[child_idx]
                self.heap[child_idx] = temp

                i = child_idx

            return return_value

    def test_heap(self):
        m = MaxHeap(10)
        m.insert(5)
        assert m.retrieve() == 5
        m.insert(5)
        m.insert(10)
        assert m.retrieve() == 10
        m.insert(3)
        m.insert(10)
        assert m.retrieve() == 10
        m.insert(20)
        m.insert(10)
        assert m.retrieve() == 20


##################
## Graph
##################  

class GraphNode:
    def __init__(self, name):
        self.name = name
        self.children = []

    def __repr__(self):
        return str(self.name)

class Graph:
    def __init__(self):
        self.nodes = {}

    def add_vertex(self, name):
        self.nodes[name] = GraphNode(name)

    def add_edge(self, start_vertex, end_vertex):
        self.nodes[start_vertex].children.append(self.nodes[end_vertex])

    def dfsRecursive(self, root_node):
        visited = set()

        def dfsHelper(node):
            if node is None:
                return 
            
            print(node)
            visited.add(node)

            for child_node in node.children:
                if child_node not in visited:
                    dfsHelper(child_node)

        return dfsHelper(root_node)

    def depthFirstTraversal(self, root_node):
        node_stack = Stack()
        visited = set()
        node_stack.push(root_node)

        while(not node_stack.isEmpty()):
            current_node = node_stack.pop()
            if current_node not in visited:
                print(current_node)
                visited.add(current_node)

                for child_node in current_node.children:
                    if child_node not in visited:
                        node_stack.push(child_node)

    def breadthFirstTraversal(self, root_node):
        node_queue = Queue()
        visited = set()
        node_queue.push(root_node)

        while(not node_queue.isEmpty()):
            current_node = node_queue.pop()
            if current_node not in visited:
                print(current_node)
                visited.add(current_node)

                for child_node in current_node.children:
                    if child_node not in visited:
                        node_queue.push(child_node)

    def test_graph(self):
        g = Graph(6)
        g.add_edge(0,1)
        g.add_edge(0,4)
        g.add_edge(0,5)
        g.add_edge(1,3)
        g.add_edge(1,4)
        g.add_edge(2,1)
        g.add_edge(3,2)
        g.add_edge(3,4)

        g.dfsRecursive(g.nodes[0])

g = Graph(6)
g.test_graph()


class GraphAdjacencyList:
    def __init__(self):
        self.adjacencies = {}

    def add_vertex(self, name):
        self.adjacencies[name] = list()

    def add_edge(self, start_vertex, end_vertex):
        self.adjacencies[start_vertex].append(end_vertex)

class GraphAdjacencyMatrix:
    def __init__(self, num_vertices):
        self.adjacencies = [[0 for _ in range(num_vertices)] for _ in range(num_vertices)]

    def add_edge(self, start_vertex, end_vertex):
        self.adjacencies[start_vertex][end_vertex] = 1


