"""
1. init one-hot encoding 
generate normal distribution, sắp xếp lại, lấy giá trị lớn nhất cho y_label
swap elem tại `torch.Tensor.argmax()` sang index của `y_label`
để đảm bảo y_label có giá trị lớn nhất (`assert:torch.Tensor[y_label] == torch.Tensor.argmax()`)

2. sau mỗi epoch 
elem tại mỗi `index != y_label` sẽ giảm dần về `0`
`giá trị giảm đi sau mỗi epoch` = (`giá trị của elem tại index đó` - `0`) / epochs
`giá trị của `torch.Tensor[y_label]`` sẽ tăng lên >>> TODO tính toán lại sự thay đổi của giá trị này.
vẫn đảm bảo vector `y` tuân theo normal distribution 

3. Tại epoch cuối cùng
`torch.Tensor[y_label] -> 1` tiến đến 1
`các phần từ còn lại -> 0 ` tiến đến 0

4. Train với nhãn có phần tử nhỏ hơn 0. Ví dụ: [0.1, -0.3, 0.2, ]

5. Ý tưởng khác, gen normal distribution với gaussian function, a, b thay đổi, 
sau đó sắp xếp lại theo thứ tự ban đầu
"""
"https://pytorch.org/docs/stable/distributions.html#normal"

import torch, math
from torch.distributions import Normal
import torch.nn.functional as F

mean = torch.tensor(0.1, dtype=torch.float32)
std = torch.tensor(0.2667, dtype=torch.float32)
m = Normal(loc=mean, scale=std)
y = m.sample((10,))

print(y.shape)
print(y)


def mean_std(x:torch.Tensor) -> tuple:
    x = x.to(torch.float32)
    mean = torch.mean(x)
    std = torch.std(x, correction=0)
    print('mean = ',mean)
    print('std = ',std)
    return mean, std

x = F.one_hot(torch.arange(1,), num_classes=10).reshape(-1)
print(x)

onehot_mean, onehot_std = mean_std(x)
y = torch.tensor([0.9, 0.1/9, 0.1/9, 0.1/9, 0.1/9, 0.1/9, 0.1/9, 0.1/9, 0.1/9, 0.1/9], dtype=torch.float32)

print('\n\n')
mean_std(y)
