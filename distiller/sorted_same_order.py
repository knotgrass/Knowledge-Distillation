import torch
import numpy as np

def get_order(nums):
    return [idx for idx, _ in sorted(enumerate(nums), key=lambda x: x[-1])]

def check_same_order(num1:list, num2:list) -> bool:
    def compare_list(l1:list, l2:list) -> bool:
        if len(l1) != len(l2):
            return False
        
        for x1, x2 in zip(l1, l2):
            if x1 != x2:
                return False
        return True
    
    if len(num1) != len(num2): 
        return False
    
    x2_pair = [x for _, x in sorted(zip(num1, num2))]
    x2 = sorted(num2)
    if not compare_list(x2_pair, x2): 
        return False
    
    x1_pair = [x for _, x in sorted(zip(num2, num1))]
    x1 = sorted(num1)
    if not compare_list(x1_pair, x1): 
        return False
    
    def check_order(nums) -> bool:
        # check theo đảm bảo các phan tử theo thứ tự từ nhỏ đến lớn
        order = get_order(nums)
        left = nums[order[0]]
        for idx in order[1:]:
            right = nums[idx]
            if left > right:
                return False
            left = right
        return True
    
    if not check_order(num1):
        return False
    if not check_order(num2):
        return False
    
    return True


def sort_by_order(nums1, nums2):
    assert len(nums1) == len(nums2)
    # order = get_order(nums1)
    order = [idx for idx, _ in sorted(enumerate(nums1), key=lambda x: x[-1])]
    # [3, 7, 0, 1, 4, 5, 6, 2, 8]
    nums2 = sorted(nums2)
    nums2 = [x for _, x in sorted(zip(order, nums2))]
    return nums2


num1 = [ 1, 1, 4,  0, 1, 2,  2, 0, 5]
num2 = [ 0, 1, 1,  0, 1, 2,  2, 0, 1]

# num1 = torch.tensor(num1)
# num2 = torch.tensor(num2)
rs1 = sort_by_order(num1, num2)
print(rs1)
print(check_same_order(num1, rs1))
