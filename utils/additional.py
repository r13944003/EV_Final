import torch
from scene.gaussian_model import GaussianModel

def set_water_style(model, rgb=(0.7, 0.95, 1.1), alpha=0.1, sh_scale=1.2):
    """
    支援 SH 格式為 [N, 15, 3] 的 GaussianModel，將其轉為亮水藍色風格。
    """
    device = model._features_dc.device
    dtype = model._features_dc.dtype

    # 設定 base RGB 顏色
    if model._features_dc.numel() > 0:
        rgb_tensor = torch.tensor(rgb, dtype=dtype, device=device)
        model._features_dc.data.copy_(rgb_tensor.expand_as(model._features_dc))

    # 設定透明度
    if model._opacity.numel() > 0:
        inv_alpha = model.inverse_opacity_activation(torch.tensor(alpha, dtype=dtype, device=device))
        model._opacity.data.copy_(inv_alpha.expand_as(model._opacity))

    # 設定高階 SH 色彩方向（支援 shape: [N, 15, 3]）
    if model._features_rest.numel() > 0 and model._features_rest.ndim == 3:
        model._features_rest.data.mul_(sh_scale)
