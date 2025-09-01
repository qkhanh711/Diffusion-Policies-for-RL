import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

# =========================
# Utils: Tanh-squashed Gaussian
# =========================
LOG_STD_MIN, LOG_STD_MAX = -20, 2
EPS = 1e-6

def squash_action(u, max_action):
    # a = tanh(u) * max_action
    return torch.tanh(u) * max_action

def squash_log_prob(base_dist, u, action):
    # log_prob(u) - sum log(1 - tanh(u)^2)
    # action = tanh(u) * max_action  (max_action không ảnh hưởng correction)
    log_prob_u = base_dist.log_prob(u).sum(dim=-1)
    correction = torch.log(1 - torch.tanh(u).pow(2) + EPS).sum(dim=-1)
    return log_prob_u - correction

# =========================
# Actor / Critic
# =========================
class GaussianActor(nn.Module):
    """Gaussian Actor with tanh squash."""
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super().__init__()
        self.max_action = max_action
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU()
        )
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        # init small log_std
        nn.init.constant_(self.log_std.weight, 0.0)
        nn.init.constant_(self.log_std.bias, -1.0)

    def forward(self, state):
        h = self.backbone(state)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        return mu, std

    @torch.no_grad()
    def act(self, state, deterministic=False):
        """Return action (numpy), value-less. Dùng lúc eval."""
        mu, std = self.forward(state)
        if deterministic:
            u = mu
        else:
            noise = torch.randn_like(std)
            u = mu + std * noise
        a = squash_action(u, self.max_action)
        return a.cpu().numpy()

    def dist_action_logp(self, state, action=None):
        """
        Trả về:
          - dist info: (u, a, logp_a)
          - logp được tính đúng với tanh-squash correction.
        Nếu action=None -> sample từ policy (rsample).
        """
        mu, std = self.forward(state)
        base = torch.distributions.Normal(mu, std)
        if action is None:
            # sample u, rồi squash -> a
            eps = torch.randn_like(std)
            u = mu + std * eps  # rsample
            a = squash_action(u, self.max_action)
        else:
            # Nếu đã có a (đã squash), cần recover u bằng atanh(a/max) để tính logp chính xác.
            # clip vào (-1+eps, 1-eps)
            a_clip = torch.clamp(action / self.max_action, -1 + EPS, 1 - EPS)
            u = 0.5 * torch.log((1 + a_clip) / (1 - a_clip))  # atanh
            a = action

        logp = squash_log_prob(base, u, a)
        return u, a, logp

class ValueCritic(nn.Module):
    """State-value function V(s)."""
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.v = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.v(state).squeeze(-1)  # (B,)

# =========================
# On-Policy Rollout Buffer cho PPO
# =========================
class RolloutBuffer:
    """
    Lưu đúng dữ liệu PPO: s, a, r, done, logp_old, v_old
    Sau khi kết thúc rollout (T steps hoặc done), gọi compute_gae_and_returns(last_value).
    """
    def __init__(self, state_dim, action_dim, device):
        self.device = device
        self.clear()
        self.state_dim = state_dim
        self.action_dim = action_dim

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.logps = []

    def add(self, state, action, reward, done, value, logp):
        self.states.append(torch.as_tensor(state))
        self.actions.append(torch.as_tensor(action))
        self.rewards.append(torch.as_tensor(reward))
        self.dones.append(torch.as_tensor(done))
        self.values.append(torch.as_tensor(value))
        self.logps.append(torch.as_tensor(logp))

    def compute_gae_and_returns(self, last_value, gamma=0.99, lam=0.95):
        """
        last_value: V(s_T) nếu chưa done; hoặc 0 nếu episode kết thúc.
        Trả về tensors: states, actions, logp_old, value_old, returns, advantages
        """
        rewards = torch.stack(self.rewards).to(self.device).float()
        dones = torch.stack(self.dones).to(self.device).float()
        values = torch.stack(self.values).to(self.device).float()

        # Append last_value for bootstrap
        values_plus = torch.cat([values, torch.as_tensor([last_value], device=self.device)])

        T = rewards.shape[0]
        adv = torch.zeros(T, device=self.device)
        gae = 0.0
        for t in reversed(range(T)):
            delta = rewards[t] + gamma * values_plus[t+1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            adv[t] = gae
        returns = adv + values

        # pack tensors
        states = torch.stack(self.states).to(self.device).float()
        actions = torch.stack(self.actions).to(self.device).float()
        logp_old = torch.stack(self.logps).to(self.device).float()
        value_old = values

        # clear for next rollout
        self.clear()
        return states, actions, logp_old, value_old, returns, adv

    def iterate_minibatches(self, *tensors, minibatch_size=64, shuffle=True):
        N = tensors[0].size(0)
        idxs = torch.randperm(N, device=tensors[0].device) if shuffle else torch.arange(N, device=tensors[0].device)
        for start in range(0, N, minibatch_size):
            end = start + minibatch_size
            mb_idx = idxs[start:end]
            yield [t[mb_idx] for t in tensors]

# =========================
# PPO Agent
# =========================
class Gaussian_PPO(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 discount=0.99,
                 tau=0.005,            # không dùng trong PPO, giữ để "giữ mẫu"
                 lr=3e-4,
                 lr_decay=False,
                 lr_maxt=1000,
                 grad_norm=1.0,
                 clip_ratio=0.2,
                 value_clip_ratio=0.2,
                 norm_adv=True,
                 horizon_steps=2048,   # số bước rollout mặc định
                 ent_coef=0.0,         # entropy bonus hệ số dương: total_loss = policy + 0.5*value - ent_coef*entropy
                 ):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        self.discount = discount
        self.grad_norm = grad_norm
        self.clip_ratio = clip_ratio
        self.value_clip_ratio = value_clip_ratio
        self.norm_adv = norm_adv
        self.horizon_steps = horizon_steps
        self.ent_coef = ent_coef

        self.actor = GaussianActor(state_dim, action_dim, max_action).to(device)
        self.critic = ValueCritic(state_dim).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.lr_decay = lr_decay
        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=lr_maxt, eta_min=0.)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=lr_maxt, eta_min=0.)

    @torch.no_grad()
    def sample_action(self, state, deterministic=False):
        """
        Lấy action để tương tác env:
          - Trả về np.array action
          - Đồng thời trả về value(s), logp(a|s) để buffer.add(...)
        """
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        v = self.critic(state_t).squeeze(0)            # V(s)
        # Lấy a và logp đúng chuẩn tanh-squash
        u, a_t, logp = self.actor.dist_action_logp(state_t, action=None)
        if deterministic:
            mu, _ = self.actor.forward(state_t)
            a_t = squash_action(mu, self.max_action)
            # logp không dùng khi deterministic, nhưng để nhất quán ta vẫn tính theo mu ~ delta
            base = torch.distributions.Normal(mu, torch.zeros_like(mu)+EPS)
            logp = squash_log_prob(base, mu, a_t)
        return a_t.cpu().numpy()[0], v.cpu().item(), logp.cpu().item()

    def _update_one_epoch(self, states, actions, logp_old, value_old, returns, advantages,
                          minibatch_size=64):
        # Chuẩn hóa advantage
        if self.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # iterate mini-batches
        policy_loss_epoch = 0.0
        value_loss_epoch = 0.0
        entropy_epoch = 0.0
        n_mb = 0

        for s_mb, a_mb, logp_old_mb, v_old_mb, ret_mb, adv_mb in RolloutBuffer.iterate_minibatches(
            RolloutBuffer, states, actions, logp_old, value_old, returns, advantages, minibatch_size=minibatch_size, shuffle=True
        ):
            # Policy
            _, _, logp_new = self.actor.dist_action_logp(s_mb, action=a_mb)
            ratio = torch.exp(logp_new - logp_old_mb)          # quan trọng!
            surr1 = ratio * adv_mb
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv_mb
            policy_loss = -torch.min(surr1, surr2).mean()

            # Entropy (dùng mẫu lại)
            mu, std = self.actor.forward(s_mb)
            base = torch.distributions.Normal(mu, std)
            entropy = base.entropy().sum(dim=-1).mean()

            # Value loss (clipped around v_old)
            v_pred = self.critic(s_mb)
            if self.value_clip_ratio is not None:
                v_clipped = v_old_mb + torch.clamp(v_pred - v_old_mb, -self.value_clip_ratio, self.value_clip_ratio)
                v_loss_unclipped = (v_pred - ret_mb).pow(2)
                v_loss_clipped = (v_clipped - ret_mb).pow(2)
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
            else:
                value_loss = 0.5 * (v_pred - ret_mb).pow(2).mean()

            total_loss = policy_loss + value_loss - self.ent_coef * entropy

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()

            if self.grad_norm and self.grad_norm > 0:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_norm)

            self.actor_optimizer.step()
            self.critic_optimizer.step()

            policy_loss_epoch += policy_loss.item()
            value_loss_epoch += value_loss.item()
            entropy_epoch += entropy.item()
            n_mb += 1

        if n_mb == 0: n_mb = 1
        return {
            "policy_loss": policy_loss_epoch / n_mb,
            "value_loss": value_loss_epoch / n_mb,
            "entropy": entropy_epoch / n_mb
        }

    def train(self, rollout_buffer, update_epochs=10, minibatch_size=64, gamma=None, lam=0.95):
        """
        rollout_buffer: đối tượng RolloutBuffer đã add() hết T bước, đã compute_gae_and_returns(...)
        """
        if gamma is None:
            gamma = self.discount

        # TÍNH returns/advantages từ buffer
        # Ở ngoài, bạn truyền last_value = 0 nếu done, hoặc V(s_T) nếu chưa done.
        # Ở đây giả định đã gọi buffer.compute_gae_and_returns trước khi train.
        # Nếu chưa, bạn có thể gọi ở ngoài như:
        # last_value = critic(last_state) if not done else 0.0
        # states, actions, logp_old, value_old, returns, advantages = buffer.compute_gae_and_returns(last_value, gamma, lam)
        raise_if_not = ("Use buffer.compute_gae_and_returns(...) before calling train().")
        # Để "giữ mẫu" gọn, ta cho phép train nhận trực tiếp buffer đã được compute:
        if not hasattr(rollout_buffer, "_cached_pack"):
            # bạn có thể cache trước khi gọi train để tránh compute nhiều lần
            pass

        # Ở đây, để đơn giản, ta mong caller đã làm:
        # states, actions, logp_old, value_old, returns, advantages = rollout_buffer.compute_gae_and_returns(...)
        try:
            states, actions, logp_old, value_old, returns, advantages = rollout_buffer._last_pack
        except AttributeError:
            raise RuntimeError(
                "Hãy gọi: "
                "states, actions, logp_old, value_old, returns, advantages = buffer.compute_gae_and_returns(last_value)\n"
                "buffer._last_pack = (states, actions, logp_old, value_old, returns, advantages)\n"
                "rồi mới gọi ppo.train(buffer, ...)"
            )

        metrics = {}
        for _ in range(update_epochs):
            out = self._update_one_epoch(states, actions, logp_old, value_old, returns, advantages,
                                         minibatch_size=minibatch_size)
            for k, v in out.items():
                metrics[k] = metrics.get(k, 0.0) + v

        for k in metrics:
            metrics[k] /= max(update_epochs, 1)

        if self.lr_decay:
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

        return metrics

    # Giữ API save/load giống bạn
    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/ppo_actor_{id}.pth')
            torch.save(self.critic.state_dict(), f'{dir}/ppo_critic_{id}.pth')
        else:
            torch.save(self.actor.state_dict(), f'{dir}/ppo_actor.pth')
            torch.save(self.critic.state_dict(), f'{dir}/ppo_critic.pth')

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/ppo_actor_{id}.pth', map_location=self.device))
            self.critic.load_state_dict(torch.load(f'{dir}/ppo_critic_{id}.pth', map_location=self.device))
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/ppo_actor.pth', map_location=self.device))
            self.critic.load_state_dict(torch.load(f'{dir}/ppo_critic.pth', map_location=self.device))
