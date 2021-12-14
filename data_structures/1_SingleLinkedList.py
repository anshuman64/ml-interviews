class SingleNode: 
    def __init__(self, value):
        self.value = value
        self.next = None

class SingleLinkedList:
    def __init__(self):
        self.head = None
        self.length = 0

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert index < self.length, "Index out of bounds"

        current_node = self.head
        for i in range(index):
            current_node = current_node.next 
        
        return current_node.value

    def __repr__(self):
        if self.head is None:
            return '[]'
        
        return_string = '[' + str(self.head.value)
        current_node = self.head 

        while(current_node.next is not None):
            return_string += ', '
            current_node = current_node.next
            return_string += str(current_node.value)

        return return_string + ']'

    def get_idx(self, value):
        assert self.length > 0, "No elements in list"

        current_node = self.head 
        idx = 0

        while (current_node is not None):
            if current_node.value == value:
                return idx
            else:
                current_node = current_node.next 
                idx += 1

        return -1

    def insertAt(self, value, index):
        assert index <= self.length, "Index out of bounds"
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
        self.insertAt(value, 0)

    def deleteAt(self, index):
        assert index < self.length, "Index out of bounds"

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
        assert self.length > 0, "No elements in list"

        if self.head.value == value:
            self.head = self.head.next
            self.length -= 1
            return True
        else:
            current_node = self.head

            while (current_node.next is not None):
                # Go to node before node-to-delete
                if current_node.next.value == value:
                    current_node.next = current_node.next.next
                    self.length -= 1
                    return True
                else:
                    current_node = current_node.next

        return False


def test_linkedlist():
    l = SingleLinkedList()
    assert len(l) == 0
    assert str(l) == '[]'

    l.insert(3)
    assert len(l) == 1
    assert l.head.value == 3
    assert str(l) == '[3]'

    l.insert(5)
    assert len(l) == 2
    assert l.head.value == 5
    assert l.head.next.value == 3
    assert l[0] == 5
    assert l[1] == 3
    assert str(l) == '[5, 3]'

    l.insertAt(1, 1)
    assert str(l) == '[5, 1, 3]'
    l.insertAt(4, 0)
    assert str(l) == '[4, 5, 1, 3]'
    l.insertAt(6, 4)
    assert str(l) == '[4, 5, 1, 3, 6]'
    l.deleteAt(0)
    assert str(l) == '[5, 1, 3, 6]'
    l.deleteAt(1)
    assert str(l) == '[5, 3, 6]'
    l.delete(6)
    assert str(l) == '[5, 3]'
    l.delete(5)
    assert str(l) == '[3]'
    l.delete(3)
    assert str(l) == '[]'

if __name__ == '__main__':
    test_linkedlist()