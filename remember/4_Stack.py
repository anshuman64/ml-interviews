class StackNode:
    def __init__(self, value):
        self.value = value
        self.next = None


class SimpleStack:
    def __init__(self):
        self.items = []

    def __len__(self):
        return len(self.items)

    def push(self, item):
        self.items.append(item)

    def pop(self):
        assert len(self.items) > 0, "No items in stack"
        return self.items.pop()

    def peek(self):
        assert len(self.items) > 0, "No items in stack"
        return self.items[len(self.items) - 1]

    def isEmpty(self):
        return len(self.items) == 0


class Stack:
    def __init__(self):
        self.top = None
        self.length = 0

    def __len__(self):
        return self.length

    def push(self, item):
        new_node = StackNode(item)

        if (self.top is None):
            self.top = new_node
        else:
            new_node.next = self.top
            self.top = new_node

        self.length += 1

    def pop(self):
        assert self.length > 0, "No items in stack"

        return_node = self.top
        self.top = self.top.next
        self.length -= 1

        return return_node.value

    def peek(self):
        assert self.length > 0, "No items in stack"
        return self.top.value

    def isEmpty(self):
        return self.length == 0


def test_stack():
    l = SimpleStack()
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

    # l.peek()
    # l.pop()


if __name__ == '__main__':
    test_stack()
