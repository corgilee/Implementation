
'''
Key value store with transactions


Set: sets the value of the specified key for either global transaction or whatever transaction we are on
Get: gets most recent value of key if it exists. Can look at all transactions to find key
Delete: removes key-value if it exiists

Begin: starts new transaction
Commit: will merge current transaction with previous
Rollback: will remove last transaction

'''

class KVstore:
    def __init__(self):
        self.stack=[{}] #set up a stack, initialize with a blank dict first

    def set(self,key,value):
        # get the lastest transaction
        self.stack[-1][key]=value

    def get(self,key):
        #check if the key exists in any of the dictionary:
        #iterate the stack
        n=len(self.stack)
        for i in range(n-1,-1,-1):
            if key in self.stack[i]:
                return self.stack[i][key]
        return None

    def delete(self,key):
        
        
        

