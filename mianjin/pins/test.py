class Solution:
    def minimizeResult(self, expression: str) -> str:
        '''
        根据题目直接解
        把两个数字parse出来，然后遍历进行分析
        遍历的时候要注意 边界条件
        '''
        new_exp=expression.split("+")
        num1=new_exp[0] #str
        num2=new_exp[1] #str
        #print(num1,num2)
        n1=len(num1)
        n2=len(num2) 

        res=float(inf)
        for i in range(-1,n1-1):
            #左括号必须包进去一个数字
            for j in range(n2):
                #右括号必须包进去一个数字
                # 这里的 i,j 是cut的index,所以要用+1
                n11=num1[:i+1]
                n12=num1[i+1:]
                if len(n11)==0:
                    first=1
                else:
                    first=int(n11)
                
                n21=num2[:j+1]
                n22=num2[j+1:]
                if len(n22)==0:
                    fourth=1
                else:
                    fourth=int(n22)
                
                #print(n11,n12,n21,n22)             
                c=first*(int(n12)+int(n21))*fourth
                if c<res:
                    res=c
                    output=n11+"("+n12+"+"+n21+")"+n22
                    # print(c,output)
                    # print(n11,n12,n21,n22)
                    
        return output