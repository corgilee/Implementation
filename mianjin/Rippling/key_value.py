#https://leetcode.com/discuss/interview-question/279913/key-value-store-with-transaction

'''
Key value store with transactions


Set: sets the value of the specified key for either global transaction or whatever transaction we are on
Get: gets most recent value of key if it exists. Can look at all transactions to find key
Delete: removes key-value if it exiists

Begin: starts new transaction
Commit: will merge current transaction with previous
Rollback: will remove last transaction

'''



from typing import *

class KVStore:
    def __init__(self):
        self.stack = [{}] #数据结构是list，list里面包含{}

    def set(self, key: Any, value: Any):
        """O(1)"""
        self.stack[-1][key] = value

    def get(self, key: Any) -> Optional[Any]:
        """O(transaction)"""
        for i in range(len(self.stack) - 1, -1, -1):
            if key in self.stack[i]:
                return self.stack[i][key]

    def begin(self):
        """O(1)"""
        self.stack.append({})

    def commit(self):
        """O(n_keys)"""
        last_dic = self.stack.pop() #把最后一个pop出来

        for k, v in last_dic.items():
            self.stack[-1][k] = v #合并到当前的stack 顶

    def rollback(self):
        """O(1)"""
        self.stack.pop() #直接弹出


def test_KVStore():
    kv = KVStore()
    kv.set(1, 3)

    assert kv.get(1) == 3
    assert kv.get(2) is None
    print("test_KVStore pass")


def test_KVStore_single_transaction():
    kv = KVStore()
    kv.set(1, 3)

    kv.begin()
    kv.set(2, 4)
    assert kv.get(1) == 3
    assert kv.get(2) == 4
    kv.commit()

    assert kv.get(1) == 3
    assert kv.get(2) == 4
    print("test_KVStore_single_transaction")


def test_KVStore_rollback():
    kv = KVStore()
    kv.set(1, 3)

    kv.begin()
    kv.set(2, 4)
    assert kv.get(1) == 3
    assert kv.get(2) == 4
    kv.rollback()

    assert kv.get(1) == 3
    assert kv.get(2) is None
    print("test_KVStore_rollback")


def test_KVStore_multiple_begin():
    kv = KVStore()
    kv.set(1, 3)

    kv.begin()
    kv.set(2, 4)

    kv.begin()
    kv.set(3, 5)

    assert kv.get(1) == 3
    assert kv.get(2) == 4
    assert kv.get(3) == 5

    kv.commit()

    assert kv.get(1) == 3
    assert kv.get(2) == 4
    assert kv.get(3) == 5

    kv.rollback()

    assert kv.get(1) == 3
    assert kv.get(2) == None
    assert kv.get(3) == None
    print("test_KVStore_multiple_begin")

test_KVStore()
test_KVStore_single_transaction()
test_KVStore_rollback()
test_KVStore_multiple_begin()


