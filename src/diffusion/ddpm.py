import torch.nn as nn
import torch

class DDPM(nn.Module):
    def __init__(
        self,
        T: int,
        eps_model: nn.Module,
        device: str
    ):
        super().__init__()
        self.T = T
        self.eps_model = eps_model.to(device)
        self.device = device
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

    def forward(self, imgs):
        # random choose some time steps
        t = torch.randint(low=1, high=self.T+1, size=(imgs.shape[0],), device=self.device)

        # get random noise to add it to the images
        noise = torch.randn_like(imgs, device=self.device)

        # get noise image as: sqrt(alpha_t_bar) * x0 + noise * sqrt(1 - alpha_t_bar)
        batch_size, channels, width, height = imgs.shape
        noise_imgs = self.sqrt_bar_alpha_t_schedule[t].view((batch_size, 1, 1 ,1)) * imgs \
            + self.sqrt_minus_bar_alpha_t_schedule[t].view((batch_size, 1, 1, 1)) * noise
        
        # get predicted noise from our model
        pred_noise = self.eps_model(noise_imgs, t.unsqueeze(1))

        # calculate of Loss simple ||noise - pred_noise||^2, which is MSELoss
        return self.criterion(pred_noise, noise)
    
    def sample(self, n_samples, size):
        self.eval()
        with torch.no_grad():
            # get normal noise
            x_t = torch.randn(n_samples, *size, device=self.device)
            # calculate x_(t-1) on every iteration
            for t in range(self.T, 0, -1):
                t_tensor = torch.tensor([t], device=self.device).repeat(x_t.shape[0], 1)
                # get predicted noise from model
                pred_noise = self.eps_model(x_t, t_tensor)

                # get some noise to calculate x_(t-1) as in formula (How to get a Noise)
                # for t = 0, noise should be 0
                z = torch.randn_like(x_t, device=self.device) if t > 0 else 0

                # Formula from How to get sample
                # x_(t-1) = 1 / sqrt(alpha_t) * (x_t - pred_noise * (1 - alpha_t) / sqrt(1 - alpha_t_bar)) + beta_t * eps
                x_t = 1 / torch.sqrt(self.alpha_t_schedule[t]) * \
                    (x_t - pred_noise * (1 - self.alpha_t_schedule[t]) / self.sqrt_minus_bar_alpha_t_schedule[t]) + \
                    torch.sqrt(self.beta_schedule[t]) * z
            return x_t