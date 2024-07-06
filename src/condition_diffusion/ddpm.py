import torch.nn as nn
import torch
from typing import List, Tuple

class DDPM(nn.Module):
    def __init__(
        self,
        T: int,
        p_cond: float,
        eps_model: nn.Module,
        device: str
    ):
        super().__init__()
        self.T = T
        self.eps_model = eps_model.to(device)
        self.device = device
        self.p_cond = torch.tensor([p_cond]).to(device)
        beta_schedule = torch.linspace(1e-4, 0.02, T + 1, device=device)
        alpha_t_schedule = 1 - beta_schedule
        bar_alpha_t_schedule = torch.cumprod(alpha_t_schedule.detach().cpu(), 0).to(device)
        sqrt_bar_alpha_t_schedule = torch.sqrt(bar_alpha_t_schedule)
        sqrt_minus_bar_alpha_t_schedule = torch.sqrt(1 - bar_alpha_t_schedule)
        self.register_buffer("beta_schedule", beta_schedule)
        self.register_buffer("alpha_t_schedule", alpha_t_schedule)
        self.register_buffer("bar_alpha_t_schedule", bar_alpha_t_schedule)
        self.register_buffer("sqrt_bar_alpha_t_schedule", sqrt_bar_alpha_t_schedule)
        self.register_buffer("sqrt_minus_bar_alpha_t_schedule", sqrt_minus_bar_alpha_t_schedule)
        self.criterion = nn.MSELoss()

    def forward(self, imgs: torch.Tensor, conds: torch.Tensor):
        t = torch.randint(low=1, high=self.T+1, size=(imgs.shape[0],), device=self.device)
        noise = torch.randn_like(imgs, device=self.device)
        batch_size, channels, width, height = imgs.shape
        noise_imgs = self.sqrt_bar_alpha_t_schedule[t].view((batch_size, 1, 1 ,1)) * imgs \
            + self.sqrt_minus_bar_alpha_t_schedule[t].view((batch_size, 1, 1, 1)) * noise
        
        conds = conds.unsqueeze(1)
        mask = torch.rand_like(conds, dtype=torch.float32) > self.p_cond

        pred_noise = self.eps_model(noise_imgs, t.unsqueeze(1), conds, mask)

        return self.criterion(pred_noise, noise)
    
    def sample(self, n_samples: int, size: Tuple[int], classes: List[int], w: float):
        self.eval()
        assert len(classes) == n_samples
        with torch.no_grad():
            x_t = torch.randn(n_samples, *size, device=self.device)
            cond = torch.tensor(classes).unsqueeze(1).to(self.device)
            mask_ones = torch.ones_like(cond)
            mask_zeros = torch.zeros_like(cond)
            for t in range(self.T, 0, -1):
                z = torch.randn_like(x_t, device=self.device) if t > 0 else 0
                t_tensor = torch.tensor([t], device=self.device).repeat(x_t.shape[0], 1)
                pred_noise_cond = self.eps_model(x_t, t_tensor, cond, mask_ones)
                pred_noise_zero = self.eps_model(x_t, t_tensor, cond, mask_zeros)
                pred_noise = (1 + w) * pred_noise_cond - w * pred_noise_zero
                x_t = 1 / torch.sqrt(self.alpha_t_schedule[t]) * \
                    (x_t - pred_noise * (1 - self.alpha_t_schedule[t]) / self.sqrt_minus_bar_alpha_t_schedule[t]) + \
                    torch.sqrt(self.beta_schedule[t]) * z
            return x_t