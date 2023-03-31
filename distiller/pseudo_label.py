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
"""
"https://pytorch.org/docs/stable/distributions.html#normal"


import torch
from torch.distributions import Normal

num_class = 10
mu = 1. / num_class

sigma = 1. # std
m = Normal(loc=mu, scale=sigma)
y = m.sample((10,))

print(y.shape)
print(y)