"""
1. init one-hot encoding
generate normal distribution, sắp xếp lại, lấy giá trị lớn nhất cho y_label
swap elem tại `Tensor.argmax()` sang index của `y_label`
để đảm bảo y_label có giá trị lớn nhất (`assert:Tensor[y_label] == Tensor.argmax()`)

2. sau mỗi epoch
elem tại mỗi `index != y_label` sẽ giảm dần về `0`
`giá trị giảm đi sau mỗi epoch` = (`giá trị của elem tại index đó` - `0`) / epochs
`giá trị của `Tensor[y_label]`` sẽ tăng lên >>> TODO tính toán lại sự thay đổi của giá trị này.
vẫn đảm bảo vector `y` tuân theo normal distribution

3. Tại epoch cuối cùng
`Tensor[y_label] -> 1` tiến đến 1
`các phần từ còn lại -> 0 ` tiến đến 0

4. Train với nhãn có phần tử nhỏ hơn 0. Ví dụ: [0.1, -0.3, 0.2, ]

5. Ý tưởng khác, gen normal distribution với gaussian function, a, b thay đổi,
sau đó sắp xếp lại theo thứ tự ban đầu

6. Sử dụng pretrained để gen ylabel ở epoch đầu tiên. Chỉ sử dụng đối với y đúng
"""
"https://pytorch.org/docs/stable/distributions.html#normal"

from typing import Any
import torch, math
from torch import Tensor
from torch.distributions import Normal
import torch.nn.functional as F

class Y_normal(object):
    """
    generate y label, tuân theo normal distribution
    """
    def __init__(self,
                 num_classes:int = 1000,
                 n_epochs   :int = ...,
                 std        :Any = 0.0316) -> None:
        self.n_epochs = n_epochs
        self.num_classes = num_classes
        self.change_mean_std(torch.tensor(1./num_classes, dtype=torch.float32),
                             torch.tensor(std, dtype=torch.float32))
        self.init_order()

    def change_mean_std(self, mean:Tensor=None,
                              std:Tensor=None):
        if mean is not None:
            self.mean = mean

        if std is not None:
            self.std = std
            onehot = F.one_hot(torch.arange(1, ), num_classes=self.num_classes).view(-1)
        assert 0 < self.std <= torch.std(onehot.to(dtype=torch.float32), correction=0)
        #num_classes= 10  ;   0 <= std <= 0.3000
        #num_classes= 100 ;   0 <= std <= 0.0995
        #num_classes= 1000;   0 <= std <= 0.0316
        self.normal = Normal(loc=self.mean, scale=self.std)

    def init_order(self, idx):
        first = self.normal.sample((self.num_classes,))
        # self.order = torch.argsort(self.order)
        self.order = [idx for idx, _ in sorted(enumerate(first), key=lambda x: x[-1])]
        return

    def sort_by_order(self, y:Tensor) -> Tensor:
        # assert len(self.order) == len(y)
        y = sorted(y)
        y = [x for _, x in sorted(zip(self.order, y))]
        return torch.tensor(y)

    def __call__(self, idx:int) -> Tensor:
        # FIXME thuật toán k chạy với idx, order giữ nguyên tại lần tạo đầu tiên,
        # tuy nhiên mỗi sample lại có order khác nhau.
        # tạo `y_label` ngay từ đầu, lúc train chỉ load lên thôi
        y = self.normal.sample((self.num_classes, ))
        y = self.sort_by_order(y)
        return y


mean = torch.tensor(0.1, dtype=torch.float32)
std = torch.tensor(0.2667, dtype=torch.float32)
m = Normal(loc=mean, scale=std)
y = m.sample((10,))

print(y.shape)
print(y)


def mean_std(x:Tensor) -> tuple:
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
