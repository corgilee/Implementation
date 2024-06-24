class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        #option1: divide and conquer
        #option 2: heap

        # divide and conquer 就是把lists里面的list 两两排序，放在一个while 循环里面，直到全部排序完成 合并成一个list

        # n*logk, every single merge take O(n), "log k" depth of recursion 
        if len(lists)==0:
            return None
        
        def sort2lists(list1,list2):
            dummy=ListNode(0)
            cur=dummy
            while list1 and list2:
                if list1.val<=list2.val:
                    cur.next=list1
                    list1=list1.next
                    cur=cur.next
                else:
                    cur.next=list2
                    list2=list2.next
                    cur=cur.next
            if list1:
                cur.next=list1
            if list2:
                cur.next=list2
            return dummy.next

        def merge_sort(lists, start, end):
            if start==end:
                return lists[start]
            mid=(start+end)//2
            left=merge_sort(lists,start,mid)
            right=merge_sort(lists,mid+1,end)
            return sort2lists(left,right)

        return merge_sort(lists,0,len(lists)-1)
