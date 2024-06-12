'''
https://www.1point3acres.com/bbs/thread-1042581-1-1.html
'''
# part 1

class BillingStatus:
    def __init__(self):
        self.ad_delivery_pennies = 0
        self.payment_pennies = 0
    
    def apply_transaction(self, transaction, monetary_columns):
        for col in monetary_columns:
            if col in transaction:
                setattr(self, col, getattr(self, col) + transaction[col])
    
    def __repr__(self):
        return f"BillingStatus(ad_delivery_pennies={self.ad_delivery_pennies}, payment_pennies={self.payment_pennies})"

def process_transactions(transactions, monetary_columns):
    user_billing_status = {}
    
    for transaction in transactions.values():
        user_id = transaction['user_id']
        if user_id not in user_billing_status:
            user_billing_status[user_id] = BillingStatus()
        
        user_billing_status[user_id].apply_transaction(transaction, monetary_columns)
    
    return user_billing_status

# Example usage:
monetary_columns = ('ad_delivery_pennies', 'payment_pennies')
transactions = {
    'ff8bc1c2-8d45-11e9-bc42-526af7764f64': {'user_id': 1, 'ad_delivery_pennies': 1000, 'transaction_timestamp': 1500000001},
    'ff8bc2e4-8d45-11e9-bc42-526af7764f64': {'user_id': 1, 'ad_delivery_pennies': 1000, 'transaction_timestamp': 1500000002},
    'ff8bc4ec-8d45-11e9-bc42-526af7764f64': {'user_id': 1, 'payment_pennies': 500, 'transaction_timestamp': 1500000003},
    'fv24z4ec-8d45-11e9-bc42-526af7764f64': {'user_id': 1, 'ad_delivery_pennies': 1000, 'payment_pennies': 500, 'transaction_timestamp': 1500000004}
}

user_billing_status = process_transactions(transactions, monetary_columns)
print(user_billing_status)


# part2 Handling Overwrite Transactions
class BillingStatus:
    def __init__(self):
        self.ad_delivery_pennies = 0
        self.payment_pennies = 0
    
    def apply_transaction(self, transaction, monetary_columns):
        for col in monetary_columns:
            if col in transaction:
                if transaction.get('overwrite', False):
                    setattr(self, col, transaction[col])
                else:
                    setattr(self, col, getattr(self, col) + transaction[col])
    
    def __repr__(self):
        return f"BillingStatus(ad_delivery_pennies={self.ad_delivery_pennies}, payment_pennies={self.payment_pennies})"

# part 3 Handling Undo and Redo Transactions
class BillingStatus:
    def __init__(self):
        self.ad_delivery_pennies = 0
        self.payment_pennies = 0
        self.transaction_history = []
    
    def apply_transaction(self, transaction, monetary_columns):
        if transaction.get('undo_last', False):
            self.undo_last_transaction()
        elif transaction.get('redo_last', False):
            self.redo_last_transaction()
        else:
            for col in monetary_columns:
                if col in transaction:
                    if transaction.get('overwrite', False):
                        setattr(self, col, transaction[col])
                    else:
                        setattr(self, col, getattr(self, col) + transaction[col])
            self.transaction_history.append(transaction)
    
    def undo_last_transaction(self):
        if self.transaction_history:
            last_transaction = self.transaction_history.pop()
            for col in last_transaction:
                if col.endswith('_pennies'):
                    if 'overwrite' in last_transaction:
                        continue
                    setattr(self, col, getattr(self, col) - last_transaction[col])
    
    def redo_last_transaction(self):
        # Implement this based on specific requirements
        pass
    
    def __repr__(self):
        return f"BillingStatus(ad_delivery_pennies={self.ad_delivery_pennies}, payment_pennies={self.payment_pennies})"

def process_transactions(transactions, monetary_columns):
    user_billing_status = {}
    
    for transaction in transactions.values():
        user_id = transaction['user_id']
        if user_id not in user_billing_status:
            user_billing_status[user_id] = BillingStatus()
        
        user_billing_status[user_id].apply_transaction(transaction, monetary_columns)
    
    return user_billing_status

# Example usage:
monetary_columns = ('ad_delivery_pennies', 'payment_pennies')
transactions = {
    'ff8bc1c2-8d45-11e9-bc42-526af7764f64': {'user_id': 1, 'ad_delivery_pennies': 1000, 'transaction_timestamp': 1500000001},
    'ff8bc2e4-8d45-11e9-bc42-526af7764f64': {'user_id': 1, 'undo_last': True, 'transaction_timestamp': 1500000002},
    'ff8bc4ec-8d45-11e9-bc42-526af7764f64': {'user_id': 1, 'payment_pennies': 500, 'transaction_timestamp': 1500000003},
    'fv24z4ec-8d45-11e9-bc42-526af7764f64': {'user_id': 1, 'ad_delivery_pennies': 1000, 'payment_pennies': 500, 'transaction_timestamp': 1500000004}
}

user_billing_status = process_transactions(transactions, monetary_columns)
print(user_billing_status)
