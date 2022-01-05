class QueueNode: 
    def __init__(self, value):
        self.value = value
        self.next = None

class SimpleQueue:
    def __init__(self):
        self.items = []

    def __len__(self):
        return len(self.items)

    def push(self, item):
        self.items.append(item)

    def pop(self):
        assert len(self.items) > 0, "No items in queue"
        return self.items.pop(0)

    def peek(self):
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
        return self.length
    
    def push(self, item):
        new_node = QueueNode(item)
        if (self.first is None):
            self.first = new_node
            self.last  = self.first
        else:
            self.last.next = new_node
            self.last = new_node
        
        self.length += 1

    def pop(self):
        assert self.length > 0, "No items in queue"
        
        return_node = self.first 
        self.first = self.first.next
        self.length -= 1

        if (self.length == 0):
            self.last = None

        return return_node.value 

    def peek(self):
        assert self.length > 0, "No items in queue"
        return self.first.value

    def isEmpty(self):
        return self.length == 0


def test_queue():
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

if __name__ == '__main__':
    test_queue()