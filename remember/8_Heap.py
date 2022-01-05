class MaxHeap:
    def __init__(self, max_size):
        """
        heap[0]       = root
        heap[count-1] = bottom-most, right-most node
        heap[count]   = bottom-most, left-most empty spot
        heap[2i+1]    = node.left
        heap[2i+2]    = node.right
        heap[(i-1)/2] = node.parent
        """
        self.heap = [None] * max_size
        self.count = 0

    def push(self, value):
        # Insert in bottom-most, left-most empty spot
        self.heap[self.count] = value
        self.count += 1

        # Reheapification - swap values with parent until in right place
        current_idx = self.count - 1
        parent_idx = (current_idx-1) // 2
        while (True):
            # Change this line for MinHeap
            if current_idx == 0 or self.heap[current_idx] <= self.heap[parent_idx]:
                break
            else:
                self.heap[current_idx], self.heap[parent_idx] = self.heap[parent_idx], self.heap[current_idx]

                current_idx = parent_idx
                parent_idx = (current_idx-1) // 2

    def pop(self):
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
            parent_idx = 0

            while(True):
                left_idx = 2*parent_idx + 1
                right_idx = 2*parent_idx + 2

                # Find larger of two children
                if left_idx > self.count:
                    break
                elif right_idx > self.count:
                    child_idx = left_idx
                else:
                    child_idx = left_idx if self.heap[left_idx] > self.heap[right_idx] else right_idx

                # Swap with parent
                if self.heap[parent_idx] < self.heap[child_idx]:
                    self.heap[parent_idx], self.heap[child_idx] = self.heap[child_idx], self.heap[parent_idx]
                    parent_idx = child_idx
                else:
                    break

            return return_value

def test_heap():
    m = MaxHeap(10)
    m.push(5)
    assert m.pop() == 5
    m.push(5)
    m.push(10)
    assert m.pop() == 10
    m.push(3)
    m.push(10)
    assert m.pop() == 10
    m.push(20)
    m.push(10)
    assert m.pop() == 20

    m = MaxHeap(10)
    m.push(5)
    m.push(3)
    m.push(20)
    m.push(8)
    m.push(10)
    assert m.pop() == 20
    assert m.pop() == 10
    assert m.pop() == 8
    assert m.pop() == 5
    assert m.pop() == 3

if __name__ == '__main__':
    test_heap()