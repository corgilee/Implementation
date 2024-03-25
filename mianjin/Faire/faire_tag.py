'''
Given a string, return true/false if it's a valid tagged element.
 The tags use {{ and }} as brackets, and leading # indicate a open tag, leading / indicate a closing tag. 
(我的理解， open tag word must be companied by a close tag word)
Examples:
Return true: {{ #abc }} {{ #cba }} hello world {{ /cba }} {{ /abc }}
Return true （单 { 视为正常input）: {{ #abc }} {{ #cba }} hello { world {{ /cba }} {{ /abc }}
Return false: {{ #abc }} hello world  {{ /abc
Return false: {{ #abc }} {{ #cba }} hello world {{ /cba }}
Return false: {{ #abc }} {{ #c‍‌‌‌‍‍‌‍‌‌‌‍‍‍‌‌‌‌ba }} hello world {{ /abc }} {{ /cba }}
'''


# if there is a  "{{", find the corresponding "}}", if there is no "}}", return False
# for the word in the brackets, if it is a open tag, push it into the stack, if it is a close tag, check if the top
# of the stack is a close tag, if not , return false


def check_string(string):
    i=0
    tag_stack=[]
    while i<len(string):
        if string[i:i+2]=="{{":
            end_index=string.find("}}",i)
            if end_index==-1:
                return False
            c_string=string[i+2:end_index].strip()
            #print(c_string)
            if c_string.startswith("#"):
                tag_stack.append(c_string[1:])
                #print(tag_stack)

            elif c_string.startswith("/"):
                if len(tag_stack)==0 or tag_stack[-1]!=c_string[1:]:
                    return False
                tag_stack.pop()
            i=end_index+2
        else:
            i+=1
    return len(tag_stack)==0

    
print(check_string('{{ #abc }} {{ #cba }} hello world {{ /cba }} {{ /abc }}'))
print(check_string('{{ #abc }} {{ #cba }} hello { world {{ /cba }} {{ /abc }}'))
print(check_string('{{ #abc }} hello world  {{ /abc'))
print(check_string('{{ #abc }} {{ #cba }} hello world {{ /cba }}'))
print(check_string('{{ #abc }} {{ #c‍‌‌‌‍‍‌‍‌‌‌‍‍‍‌‌‌‌ba }} hello world {{ /abc }} {{ /cba }}'))

        

