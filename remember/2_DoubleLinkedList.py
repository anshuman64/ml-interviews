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
        return self.length

    def __getitem__(self, index):
        assert index < self.length, "Index out of bounds"
        current_node = None

        if (index < self.length / 2):
            current_node = self.head

            for i in range(index):
                current_node = current_node.next
        else:
            current_node = self.tail

            for i in range(self.length - index - 1):
                current_node = current_node.previous

        return current_node.value

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

    def insertAt(self, value, index):
        assert index <= self.length, "Index out of bounds"
        new_node = DoubleNode(value)

        if (self.head is None):
            self.head = new_node
            self.tail = self.head
        elif (index == 0):
            new_node.next = self.head
            self.head.previous = new_node
            self.head = new_node
        elif (index == self.length):
            new_node.previous = self.tail
            self.tail.next = new_node
            self.tail = new_node
        else:
            if (index < self.length / 2):
                current_node = self.head

                # Go to node before new node
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
        self.insertAt(value, self.length)

    def deleteAt(self, index):
        assert index < self.length, "Index out of bounds"

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
            if (index < self.length / 2):
                current_node = self.head

                # Go to node to delete
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
        assert self.length > 0, "No elements in list"

        if self.head.value == value:
            self.head = self.head.next
            if self.head is not None:
                self.head.previous = None
            else:
                self.tail = None

            self.length -= 1
            return True
        elif self.tail.value == value:
            self.tail = self.tail.previous
            if self.tail is not None:
                self.tail.next = None
            else:
                self.head = None

            self.length -= 1
            return True
        else:
            current_node = self.head.next

            while (current_node is not None):
                # Go to node to delete
                if current_node.value == value:
                    if (current_node.next is not None):
                        current_node.next.previous = current_node.previous
                    if (current_node.previous is not None):
                        current_node.previous.next = current_node.next

                    self.length -= 1
                    return True
                else:
                    current_node = current_node.next

        return False


def test_linkedlist():
    l = DoubleLinkedList()
    assert len(l) == 0
    assert str(l) == '[]'

    l.insert(3)
    assert len(l) == 1
    assert l.head.value == 3
    assert str(l) == '[3]'

    l.insert(5)
    assert len(l) == 2
    assert l.head.value == 3
    assert l.head.next.value == 5
    assert l[0] == 3
    assert l[1] == 5
    assert str(l) == '[3, 5]'

    l.insertAt(1, 1)
    assert str(l) == '[3, 1, 5]'
    l.insertAt(4, 0)
    assert str(l) == '[4, 3, 1, 5]'
    l.insertAt(6, 4)
    assert str(l) == '[4, 3, 1, 5, 6]'
    l.deleteAt(0)
    assert str(l) == '[3, 1, 5, 6]'
    l.deleteAt(1)
    assert str(l) == '[3, 5, 6]'
    l.delete(6)
    assert str(l) == '[3, 5]'
    l.delete(5)
    assert str(l) == '[3]'
    l.delete(3)
    assert str(l) == '[]'

    l.insert(3)
    assert len(l) == 1
    assert l.head.value == 3
    assert str(l) == '[3]'


if __name__ == '__main__':
    test_linkedlist()
