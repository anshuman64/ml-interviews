class HashNode:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.next = None

class HashLinkedList:
    def __init__(self):
        self.head = None
        self.length = 0

    def get_value(self, key):
        current_node = self.head 

        while (current_node is not None):
            if current_node.key == key:
                return current_node.value
            else:
                current_node = current_node.next
    
    def insert(self, key, value):
        new_node = HashNode(key, value)

        if self.head is None:
            self.head = new_node
        else:
            new_node.next = self.head
            self.head = new_node

        self.length += 1

    def delete(self, key):
        assert self.length > 0, "No elements in list"

        if self.head.key == key:
            self.head = self.head.next
            self.length -= 1
            return True
        else:
            current_node = self.head

            while (current_node.next is not None):
                if current_node.next.key == key:
                    current_node.next = current_node.next.next
                    self.length -= 1
                    return True
                else:
                    current_node = current_node.next
        
        return False

class HashTable:
    def __init__(self, size):
        self.table = [HashLinkedList() for i in range(size)]
        self.count = 0

    def __len__(self):
        return self.count 

    def hash(self, key):
        return key % len(self.table)

    def insert(self, key, value):
        hash_idx = hash(key)

        self.table[hash_idx].insert(key, value)
        self.count += 1

    def delete(self, key):
        hash_idx = hash(key)

        if self.table[hash_idx].delete(key):
            self.count -= 1

    def get(self, key):
        hash_idx = hash(key)
        return self.table[hash_idx].get_value(key)


def test_hashtable():
    h = HashTable(10)

    h.insert(1, 'hi')
    h.insert(2, 'bye')

    assert h.get(1) == 'hi'
    assert h.get(2) == 'bye'

    h.delete(1)
    h.delete(2)

if __name__ == '__main__':
    test_hashtable()