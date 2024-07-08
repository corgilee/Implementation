class Solution:
    def shortestWay(self, source: str, target: str) -> int:
        '''
        遍历target的第i个字母，如果这个字母在source里面能找到，指针移到source下一个字母，然后再遍历source
        '''
        target_index=0
        res=0
        while target_index<len(target):
            prev_index=target_index
            for char in source:
                if target_index<len(target) and char==target[target_index]:
                    target_index+=1
            if prev_index==target_index:
                # 说明target_index 没有变化，说明source里面char里面找不到配得上当前target的
                return -1
            res+=1
        return res


        