##################
## ArrayList
##################

class ArrayList:
    def __init__(self):
        self.max_size = 10
        self.list = [None] * self.max_size
        self.count = 0

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        assert index < self.count, "Not enough elements"

        return self.list[index]

    def __repr__(self):
        return str(self.list)

    def increase_size(self):
        self.max_size *= 2
        new_list = [None] * self.max_size

        for i in range(self.count):
            new_list[i] = self.list[i]

        self.list = new_list
        
    def insert(self, value):
        if self.count >= self.max_size:
            self.increase_size

        self.list[self.count] = value
        self.count += 1

    def delete(self, index):
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