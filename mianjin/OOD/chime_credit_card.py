'''
Imagine that you're writing software for a credit card provider. Your task is to implement a program that will add new credit card accounts, process charges and credits against them, and display summary information.

You are given a list of commands:

Add <card_holder> <card_number> $: Add command will create a new credit card for the given card_holder, card_number, and limit. 
It is guaranteed that the given card_holder didn't have a credit card before this operation.
New cards start with a $0 balance.
Cards numbers should be validated using basic validation.
(Bonus) Card numbers should be validated using the Luhn 10 algorithm.

Charge <card_holder> $: Charge command will increase the balance of the card associated with the provided name by the amount specified.
Charges that would raise the balance over the limit are ignored as if they were declined.
Charges against invalid cards are ignored.

Credit <card_holder> $: Credit command will decrease the balance of the card associated with the provided name by the amount specified.
Credits that would drop the balance below $0 will create a negative balance.
Credits against invalid cards are ignored.

Credit Card validation
In order to ensure the credit card number is valid, we want to run some very basic validation.
You need to ensure the string is only composed of digits [0-9] and is between 12 and 16 characters long (although most cards are 15 to 16, let's keep it simple).

(Bonus) How the Luhn algorithm works:

Starting with the rightmost digit, which is the check digit, and moving left, double the value of every second digit. If the result of this doubling operation is greater than 9 (e.g., 8 * 2 = 16), then add the digits of the product (e.g., 16: 1 + 6 = 7, 18: 1 + 8 = 9).
Take the sum of all the digits.
If the total modulo 10 is equal to 0 (if the total ends in zero) then the number is valid according to the Luhn algorithm, otherwise it is not valid.
The last Unit Test will be testing for the Luhn algorithm.

Luhn(number) = 7 + 9 + 9 + 4 + 7 + 6 + 9 + 7 + 7 = 65 = 5 (mod 10) != 0

Your Challenge

Return the card holder names with the balance of the card associated with the provided name. 
The names in output should be displayed in lexicographical order.
Display "error" instead of the balance if the credit card number does not pass validation.

Example

For

operations = [["Add", "Tom", "4111111111111111", "$1000"],
["Add", "Lisa", "5454545454545454", "$3000"],
["Add", "Quincy", "12345678901234", "$2000"],
["Charge", "Tom", "$500"],
["Charge", "Tom", "$800"],
["Charge", "Lisa", "$7"],
["Credit", "Lisa", "$100"],
["Credit", "Quincy", "$200"]]

the output should be

creditCardProvider(operations) = [["Lisa", "$-93"],
["Quincy", "error"],
["Tom", "$500"]]
Input/Output
'''

class credit_card_process:
    def __init__(self):
        self.user={} # dict 里面包一个dict

    def process_ops(self,operations):
        for op in operations:
            
            if op[0]=='Add':
                name, card,limit=op[1],op[2],op[3]
                if self.validate(card)==False or self.validate_2(card)==False:
                    self.user[name]={"card number":"error","limit":"error","balance":"error"}
                else:
                    self.user[name]={"card number":op[2],"limit":int(limit[1:]),"balance":0}

            elif op[0]=='Charge':
                name, charge=op[1],op[2]
                if self.user[name]['card number']!='error':
                    if int(charge[1:])+self.user[name]['balance']<self.user[name]['limit']:
                        self.user[name]['balance']+=int(charge[1:])

            elif op[0]=='Credit':
                name,credit=op[1],op[2]
                if self.user[name]['card number']!='error':
                    self.user[name]['balance']-=int(credit[1:])
                        
            
        res=[]
        #print(self.user)
        for u, u_dict in self.user.items():
            res.append([u,str(u_dict['balance'])])
        res.sort()
        return res
    
    def validate(self,number):
        #digits [0-9] and is between 12 and 16 characters long (although most cards are 15 to 16
        if len(number)<12 and len(number)>16:
            return False
        for c in number:
            #print(c)
            if c.isdigit()==False:
                return False
        return True

    def validate_2(self,number):
        # https://stackoverflow.com/questions/21079439/implementation-of-luhn-formula/21079551#21079551
        total_sum=0
        number_list=[]
        for i in range(len(number)):
            number_list.append(int(number[i]))

        # selected digits
        list_1=number_list[-1::-2]
        total_sum+=sum(list_1)

        list_2=number_list[-2::-2]
        for num in list_2:
            r=2*num
            total_sum+=sum([int(d) for d in str(r)])
            
        
        print(total_sum)
        return total_sum % 10==0



system=credit_card_process()


inp = [["Add", "Tom", "4111111111111111", "$1000"],
["Add", "Lisa", "5454545454545454", "$3000"],
["Add", "Quincy", "12345678901234", "$2000"],
["Charge", "Tom", "$500"],
["Charge", "Tom", "$800"],
["Charge", "Lisa", "$7"],
["Credit", "Lisa", "$100"],
["Credit", "Quincy", "$200"]]

output=system.process_ops(inp)

print(output)


# number='6011514433546201'
# output2=system.validate_2(number)
# print(output2)


