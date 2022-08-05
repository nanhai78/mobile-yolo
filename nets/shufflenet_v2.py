import sys

import torch
import torch.nn as nn
from pathlib import Path
from collections import OrderedDict


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
weights_path = str(ROOT) + "\\weights"
if weights_path not in sys.path:
    sys.path.append(weights_path)


class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp
        # 右分支
        branch_main = [
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            # pw-linear
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        ]
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            # 左分支
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if self.stride == 1:
            x_proj, x = self.channel_shuffle(old_x)  # 步长为1时先进行Shuffle。
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride == 2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()  # [1,16,112,112]
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)  # [8,2,112 *112]
        x = x.permute(1, 0, 2)  # [2,8,112,112]  # 转置
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]


class ShuffleNetV2(nn.Module):
    def __init__(self, input_size=224, n_class=1000, model_size='1.0x'):
        super(ShuffleNetV2, self).__init__()
        print('model size is ', model_size)

        self.stage_repeats = [4, 8, 4]  # 模块重复次数
        self.model_size = model_size
        if model_size == '0.5x':
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif model_size == '1.0x':
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]  # 每一个stage的输出channel
        elif model_size == '1.5x':
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif model_size == '2.0x':
            self.stage_out_channels = [-1, 24, 244, 488, 976, 2048]
        else:
            raise NotImplementedError

        # building first layer
        # 第一个CBR模块 416,416,3 -> 208,208,24
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
        )
        # 208,208,24 -> 104,104,24
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = []
        # stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        # 经过第一个stage后：104，104,24 -> 52,52,116
        # 经过第二个stage后：52，52，116 -> 26,26,232
        # 经过第三个stage后: 26,26,232 -> 13,13,464

        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]  # 模块重复次数 [4,8,4]
            output_channel = self.stage_out_channels[idxstage + 2]  # stage的输出channel

            for i in range(numrepeat):
                if i == 0:  # stage第一个模块。步长为2,起到了下采样作用 mid_channels为out_channel的一半
                    self.features.append(ShuffleV2Block(input_channel, output_channel,
                                                        mid_channels=output_channel // 2, ksize=3, stride=2))
                else:
                    self.features.append(ShuffleV2Block(input_channel // 2, output_channel,
                                                        mid_channels=output_channel // 2, ksize=3, stride=1))

                input_channel = output_channel

        self.features = nn.Sequential(*self.features)

        self.conv_last = nn.Sequential(
            nn.Conv2d(input_channel, self.stage_out_channels[-1], 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[-1]),
            nn.ReLU(inplace=True)
        )
        self.globalpool = nn.AvgPool2d(7)
        if self.model_size == '2.0x':
            self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], n_class, bias=False))
        self._initialize_weights()

    def forward(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)
        x = self.features(x)
        x = self.conv_last(x)

        x = self.globalpool(x)
        if self.model_size == '2.0x':
            x = self.dropout(x)
        x = x.contiguous().view(-1, self.stage_out_channels[-1])
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def shufflnet_v2(pretrained=False, **kwargs):
    model = ShuffleNetV2(**kwargs)
    if pretrained:
        if model.model_size == '1.0x':
            state_dic = torch.load('D:\\python_all\\WorkSpace001\\mobilenet-yolov4-pytorch-3.1\\weights\\ShuffleNetV2.1.0x.pth')
        elif model.model_size == '1.5x':
            state_dic = torch.load('D:\\python_all\\WorkSpace001\\mobilenet-yolov4-pytorch-3.1\\weights\\SShuffleNetV2.1.5x.pth')
        else:
            state_dic = torch.load('D:\\python_all\\WorkSpace001\\mobilenet-yolov4-pytorch-3.1\\weights\\ShuffleNetV2.2.0x.pth')
        new_dict = OrderedDict()
        for key, v in state_dic['state_dict'].items():
            name = key[7:]
            new_dict[name] = v
        model.load_state_dict(new_dict)
    return model


if __name__ == '__main__':
    model = shufflnet_v2(pretrained=True, model_size='1.0x')

    test_data = torch.rand(1, 3, 224, 224)
    test_outputs = model(test_data)
    print(test_outputs.size())

    # state_dic = torch.load('../weights/ShuffleNetV2.1.0x.pth')
    # new_dict = OrderedDict()
    # for key, v in state_dic['state_dict'].items():
    #     name = key[7:]
    #     new_dict[key] = v
