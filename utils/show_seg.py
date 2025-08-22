""" from __future__ import print_function
from show3d_balls import showpoints
import argparse
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetDenseCls
import matplotlib.pyplot as plt


#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--idx', type=int, default=0, help='model index')
parser.add_argument('--dataset', type=str, default='', help='dataset path')
parser.add_argument('--class_choice', type=str, default='', help='class choice')

opt = parser.parse_args()
print(opt)

d = ShapeNetDataset(
    root=opt.dataset,
    class_choice=[opt.class_choice],
    split='test',
    data_augmentation=False)

idx = opt.idx

print("model %d/%d" % (idx, len(d)))
point, seg = d[idx]
print(point.size(), seg.size())
point_np = point.numpy()

cmap = plt.cm.get_cmap("hsv", 10)
cmap = np.array([cmap(i) for i in range(10)])[:, :3]
gt = cmap[seg.numpy() - 1, :]

state_dict = torch.load(opt.model)
classifier = PointNetDenseCls(k= state_dict['conv4.weight'].size()[0])
classifier.load_state_dict(state_dict)
classifier.eval()

point = point.transpose(1, 0).contiguous()

point = Variable(point.view(1, point.size()[0], point.size()[1]))
pred, _, _ = classifier(point)
pred_choice = pred.data.max(2)[1]
print(pred_choice)

#print(pred_choice.size())
pred_color = cmap[pred_choice.numpy()[0], :]

#print(pred_color.shape)
showpoints(point_np, gt, pred_color)
 """

from __future__ import print_function

# ---------- HEADLESS PATCH：把 OpenCV 弹窗改成存 PNG ----------
# 必须放在 import show3d_balls 之前


import os, time
os.environ.setdefault("HEADLESS", "1")  # 默认无窗
try:
    import cv2
    if os.environ.get("HEADLESS") == "1":
        def _noop(*args, **kwargs): pass
        def _save_img(winname, img):
            outdir = os.path.join(os.path.dirname(__file__), "vis_out")
            os.makedirs(outdir, exist_ok=True)
            fn = os.path.join(outdir, f"show3d_{int(time.time())}.png")
            cv2.imwrite(fn, img)
            print("saved:", fn)

        # 让 waitKey 返回一个整数（模拟按 'q'），避免 None % 256 报错
        def _wait_quit(*args, **kwargs):
            return ord('q')

        # 拦截所有窗口相关 API
        cv2.namedWindow       = _noop
        cv2.imshow            = _save_img
        cv2.waitKey           = _wait_quit
        cv2.moveWindow        = _noop
        cv2.resizeWindow      = _noop
        cv2.setWindowProperty = _noop
        cv2.setWindowTitle    = _noop
        cv2.destroyWindow     = _noop
        cv2.destroyAllWindows = _noop
        cv2.createTrackbar    = _noop
        cv2.setMouseCallback  = _noop
except Exception as e:
    print("Headless patch warning:", e)
# ---------------------------------------------------------------

from show3d_balls import showpoints

import argparse
import numpy as np
import torch
import torch.utils.data
from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetDenseCls
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='model path (.pth)')
parser.add_argument('--idx', type=int, default=0, help='sample index in test split')
parser.add_argument('--dataset', type=str, required=True, help='dataset root path')
parser.add_argument('--class_choice', type=str, required=True, help='class choice, e.g. Chair')
opt = parser.parse_args()
print(opt)

# 加载测试样本
d = ShapeNetDataset(root=opt.dataset, class_choice=[opt.class_choice],
                    split='test', data_augmentation=False)
idx = opt.idx
print("model %d/%d" % (idx, len(d)))

point, seg = d[idx]                    # point: [N,3]  seg: [N]（全局标签）
print(point.size(), seg.size())
point_np = point.numpy()

# ——把全局标签 remap 成“本类局部标签”0..k-1——
parts = np.unique(seg.numpy())                           # 本样本出现的全局部件标签
label2local = {int(p): i for i, p in enumerate(parts)}   # 全局 -> 局部
gt_local = np.array([label2local[int(x)] for x in seg.numpy()], dtype=np.int64)

# 模型
try:
    # 新版 PyTorch 更安全的加载方式
    state_dict = torch.load(opt.model, map_location='cpu', weights_only=True)
except TypeError:
    state_dict = torch.load(opt.model, map_location='cpu')

k = state_dict['conv4.weight'].size(0)                   # 输出通道 = 部件数
use_ft = any(kname.startswith('feat.fstn.') for kname in state_dict.keys())
classifier = PointNetDenseCls(k=k, feature_transform=use_ft)
classifier.load_state_dict(state_dict)
classifier.eval()

# 推理
point = point.transpose(1, 0).contiguous()               # -> [3, N]
point = point.view(1, point.size(0), point.size(1))      # -> [1, 3, N]
with torch.no_grad():
    pred, _, _ = classifier(point)                       # [1, k, N]
    pred_choice = pred.argmax(2).squeeze(0).cpu().numpy()# [N] 0..k-1

# 颜色：覆盖 max(len(parts), k)
n_colors = int(max(len(parts), k))
if hasattr(plt, "colormaps"):                            # Matplotlib 新接口
    cmap = plt.colormaps.get_cmap("hsv")
else:                                                    # 旧接口（可能有弃用警告）
    cmap = plt.cm.get_cmap("hsv")
cmap_arr = cmap(np.linspace(0.0, 1.0, n_colors, endpoint=False))[:, :3]

gt_color   = cmap_arr[gt_local, :]
pred_color = cmap_arr[pred_choice, :]

# 触发渲染（HEADLESS 模式下会写 PNG 到 utils/vis_out/）
showpoints(point_np, gt_color, pred_color)
