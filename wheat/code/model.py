"""
model.py -- 식물 높이 예측을 위한 Latent (Neural) ODE.

    genotype id ─► Embedding / kinship 행 ─┬────────────┐
                                          │            │
   기온 시계열 T, 시간 J ─► LSTM Encoder ─┴─► z0        │
                                                │       │
                                                ▼       ▼
                            dz/dτ = f(z, T(τ), g, τ)   Neural ODE (RK4)
                                                │
                                                ▼
                                       latent trajectory z(τ)
                                                │
                                                ▼
                                        Decoder (MLP) ─► ŷ(t)

  입력은 시간과 기온, 그리고 유전자형뿐이다. 예측 대상 연도의 높이 관측은
  어떤 경로로도 모델에 들어가지 않는다.

핵심 설계 결정
--------------
1. 시간은 τ ∈ [0, 1] 로 정규화해 적분한다 (170일 = 1.0). 원 단위(day)로 h=1을
   쓰면 170 스텝 누적으로 잠재상태가 발산하기 쉽다. 물리 손실을 계산할 때만
   dy/dt = (dy/dτ) / days_per_tau 로 되돌린다.
2. torchdiffeq 의존성 없이 고정 스텝 RK4로 적분한다. 관측 자체가 일 단위이고
   격자가 균일하므로 adaptive solver의 이점이 거의 없고, 역전파가 단순해진다.
3. 물리 손실(선택)은 잠재 미분에서 연쇄법칙으로 바로 얻는다.
       dŷ/dτ = (∂dec/∂z) · f(z, env, g, τ)
   논문처럼 시간에 대해 autograd를 다시 돌릴 필요가 없다.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def mlp(sizes, act=nn.Tanh, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    if out_act is not None:
        layers.append(out_act())
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------- #
# 환경 입력 보간 (RK4의 중간 시점 τ+h/2 에서 env 값이 필요하다)
# --------------------------------------------------------------------------- #
class Interp:
    """[B, T] 또는 [B, T, C] 격자 위 시계열을 τ∈[0,1]에서 선형 보간한다.

    RK4 는 τ+h/2 같은 격자 사이 시점의 값을 요구하므로 필요하다.
    """

    def __init__(self, x: torch.Tensor):
        self.x = x if x.dim() == 3 else x.unsqueeze(-1)
        self.n_t = self.x.shape[1]

    def __call__(self, tau) -> torch.Tensor:
        pos = float(tau) * (self.n_t - 1)
        pos = min(max(pos, 0.0), self.n_t - 1.0)
        i0 = int(pos)
        i1 = min(i0 + 1, self.n_t - 1)
        w = pos - i0
        return (1.0 - w) * self.x[:, i0] + w * self.x[:, i1]


EnvInterpolator = Interp  # 이전 이름 호환


# --------------------------------------------------------------------------- #
# 구성 요소
# --------------------------------------------------------------------------- #
class GenotypeEncoder(nn.Module):
    """유전자형 -> 잠재 유전 표현.

    encoding='one_hot' : nn.Embedding (논문의 one-hot + FC 와 동등)
    encoding='kinship' : kinship 행렬의 해당 행을 입력으로 하는 FC
    """

    def __init__(self, n_genotypes, embed_dim, encoding="one_hot", kinship=None):
        super().__init__()
        self.encoding = encoding
        if encoding == "one_hot":
            self.embed = nn.Embedding(n_genotypes, embed_dim)
            nn.init.normal_(self.embed.weight, std=0.1)
        elif encoding == "kinship":
            if kinship is None:
                raise ValueError("encoding='kinship' 에는 kinship 행렬이 필요합니다.")
            self.register_buffer("kinship", torch.as_tensor(kinship, dtype=torch.float32))
            self.embed = nn.Sequential(nn.Linear(n_genotypes, embed_dim), nn.Tanh())
        else:
            raise ValueError(f"알 수 없는 encoding: {encoding}")

    def forward(self, g_idx: torch.Tensor) -> torch.Tensor:
        if self.encoding == "one_hot":
            return self.embed(g_idx)
        return self.embed(self.kinship[g_idx])


class EnvEncoder(nn.Module):
    """환경 시계열(기온) + 시간 + 유전자형을 읽어 z0 를 만든다.

    높이 관측은 **입력하지 않는다.** 테스트 연도에서 사용할 수 있는 정보가
    시간과 기온뿐이라는 논문의 예측 설정을 그대로 지킨다.
    """

    def __init__(self, env_dim, g_dim, hidden_size, latent_dim, num_layers=1, variational=False):
        super().__init__()
        self.variational = variational
        in_dim = env_dim + 1  # env, tau
        self.lstm = nn.LSTM(in_dim, hidden_size, num_layers=num_layers, batch_first=True)
        out_dim = latent_dim * 2 if variational else latent_dim
        self.head = nn.Linear(hidden_size + g_dim, out_dim)
        self.latent_dim = latent_dim

    def forward(self, env_hist, t_hist, g_emb):
        """t_hist: [B, T'] 시간 특징 (달력 τ 또는 열시간 s)."""
        x = torch.cat([env_hist, t_hist.unsqueeze(-1)], dim=-1)
        # 시간을 거꾸로 읽어 마지막 은닉이 t=0 근처 정보를 담게 한다 (Latent ODE 관례)
        out, _ = self.lstm(torch.flip(x, dims=[1]))
        h = torch.cat([out[:, -1], g_emb], dim=-1)
        out = self.head(h)
        if self.variational:
            mu, logvar = out.chunk(2, dim=-1)
            return mu, logvar.clamp(-8.0, 4.0)
        return out, None


class ODEFunc(nn.Module):
    """dz/dτ = f(z, env(τ), g, τ)"""

    def __init__(self, latent_dim, env_dim, g_dim, hidden_size, num_layers=2, out_scale=3.0):
        super().__init__()
        sizes = [latent_dim + env_dim + g_dim + 1] + [hidden_size] * num_layers + [latent_dim]
        self.net = mlp(sizes, act=nn.Tanh)
        # 마지막 층을 작게 시작시켜 초기 궤적이 폭주하지 않게 한다
        nn.init.uniform_(self.net[-1].weight, -1e-2, 1e-2)
        nn.init.zeros_(self.net[-1].bias)
        self.out_scale = out_scale
        self.nfe = 0

    def forward(self, t_feat, z, env_t, g_emb):
        """t_feat: [B, 1] 시간 특징. 달력 시간 τ 이거나 열시간 s."""
        self.nfe += 1
        h = torch.cat([z, env_t, g_emb, t_feat], dim=-1)
        return self.out_scale * torch.tanh(self.net(h))


def odeint_rk4(func, z0, n_t, env_interp, g_emb, n_substeps=1,
               t_interp=None, rate_interp=None):
    """τ∈[0,1] 위 균일 격자에서 고정 스텝 RK4 적분. 반환: [B, n_t, latent]

    t_interp    : None 이면 시간 특징으로 달력 시간 τ 를 쓴다.
                  주어지면 그 값(열시간 s)을 대신 쓴다.
    rate_interp : 주어지면 dz/dτ 에 ds/dτ 를 곱한다. 즉 발육이 달력이 아니라
                  열시간에 따라 진행하고, 추운 날에는 잠재상태가 거의 멈춘다.
    """
    B = z0.shape[0]

    def deriv(t, z):
        if t_interp is None:
            tf = torch.full((B, 1), float(t), device=z.device, dtype=z.dtype)
        else:
            tf = t_interp(t)
        d = func(tf, z, env_interp(t), g_emb)
        if rate_interp is not None:
            d = d * rate_interp(t)
        return d

    h_day = 1.0 / (n_t - 1)
    h = h_day / n_substeps
    z = z0
    traj = [z]
    for i in range(n_t - 1):
        for j in range(n_substeps):
            t = i * h_day + j * h
            k1 = deriv(t, z)
            k2 = deriv(t + h / 2, z + h / 2 * k1)
            k3 = deriv(t + h / 2, z + h / 2 * k2)
            k4 = deriv(t + h, z + h * k3)
            z = z + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj.append(z)
    return torch.stack(traj, dim=1)


class ODEParameterHead(nn.Module):
    """유전 표현 -> 로지스틱 ODE 파라미터 (논문 Fig. 2의 PINN-specific module).

    r     = sigmoid(.)        ∈ (0, 1)
    y_max = tanh(.) + 1       ∈ (0, 2)  [m]
    """

    def __init__(self, g_dim, hidden=8, r_init=0.15, ymax_init=0.8):
        super().__init__()
        self.net = mlp([g_dim, hidden, 2], act=nn.Tanh)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            # sigmoid(b0)=r_init, tanh(b1)+1=ymax_init 이 되도록 편향 초기화
            b0 = torch.log(torch.tensor(r_init / (1 - r_init)))
            b1 = torch.atanh(torch.tensor(ymax_init - 1.0))
            self.net[-1].bias.copy_(torch.stack([b0, b1]))

    def forward(self, g_emb):
        raw = self.net(g_emb)
        r = torch.sigmoid(raw[..., 0])
        y_max = torch.tanh(raw[..., 1]) + 1.0
        return r, y_max


# --------------------------------------------------------------------------- #
# 전체 모델
# --------------------------------------------------------------------------- #
class LatentODEHeightModel(nn.Module):
    """학습에 쓰지 않은 연도의 생육 곡선 전체를 **시간과 기온만으로** 예측한다.

    어떤 경로로도 예측 대상 연도의 높이 관측이 모델에 들어가지 않는다.

    init_mode
      'env_encoder' (기본): z0 = LSTM(기온 시계열, 시간) + 유전자형.
                    연도별 기온 조건이 초기 잠재상태에 반영된다.
      'genotype'   : z0 = MLP(유전자형). 연도 정보가 z0 에 전혀 들어가지 않고
                     기온은 ODE forcing 으로만 작용한다 (ablation 용).

    time_mode  — 발육을 무엇이 끌고 가는가
      'calendar'        (기본): 시간 특징 = 달력 시간 τ. 기존 동작.
      'thermal_feature' : 시간 특징을 열시간 s 로 **교체**. 적분은 여전히 균일 τ.
      'thermal_rate'    : 위에 더해 dz/dτ 에 ds/dτ 를 곱한다. 완전한 열시간
                          재매개화 — 추운 날에는 잠재상태가 거의 정지한다.

    'calendar' 에서 GDD 를 환경 채널로 *추가* 만 하면 모델은 여전히 τ 를 쓸 수
    있어 열시간을 무시할 수 있다. 그래서 '교체' 가 핵심이다.
    """

    def __init__(
        self,
        n_genotypes,
        env_dim=1,
        latent_dim=16,
        g_embed_dim=4,
        ode_hidden=32,
        ode_layers=2,
        dec_hidden=32,
        enc_hidden=16,
        init_mode="env_encoder",
        time_mode="calendar",
        genetic_encoding="one_hot",
        kinship=None,
        variational=False,
        n_substeps=1,
        days_per_tau=169.0,
        use_physics=True,
    ):
        super().__init__()
        if time_mode not in ("calendar", "thermal_feature", "thermal_rate"):
            raise ValueError(f"알 수 없는 time_mode: {time_mode}")
        self.init_mode = init_mode
        self.time_mode = time_mode
        self.variational = variational and init_mode == "env_encoder"
        self.n_substeps = n_substeps
        self.days_per_tau = days_per_tau
        self.latent_dim = latent_dim
        self.use_physics = use_physics

        self.g_enc = GenotypeEncoder(n_genotypes, g_embed_dim, genetic_encoding, kinship)
        self.ode_func = ODEFunc(latent_dim, env_dim, g_embed_dim, ode_hidden, ode_layers)
        self.decoder = mlp([latent_dim, dec_hidden, 1], act=nn.Tanh)
        self.out_act = nn.LeakyReLU(0.01)  # 논문과 동일: 음수 높이를 강하게 억제

        if init_mode == "genotype":
            self.z0_net = mlp([g_embed_dim, dec_hidden, latent_dim], act=nn.Tanh)
        elif init_mode == "env_encoder":
            self.encoder = EnvEncoder(
                env_dim, g_embed_dim, enc_hidden, latent_dim, variational=self.variational
            )
        else:
            raise ValueError(f"알 수 없는 init_mode: {init_mode}")

        self.ode_param_head = ODEParameterHead(g_embed_dim) if use_physics else None

    # ---------------------------------------------------------------- #
    def decode(self, z):
        return self.out_act(self.decoder(z)).squeeze(-1)

    def get_z0(self, g_emb, env=None, t_feat=None, encoder_days=0):
        if self.init_mode == "genotype":
            return self.z0_net(g_emb), None, None
        h = slice(0, encoder_days) if encoder_days > 0 else slice(None)
        mu, logvar = self.encoder(env[:, h], t_feat[:, h], g_emb)
        if self.variational and self.training:
            z0 = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z0 = mu
        return z0, mu, logvar

    def time_features(self, env, s=None):
        """[B, T] 시간 특징과, 필요하면 [B, T] 발육속도를 돌려준다."""
        B, n_t = env.shape[0], env.shape[1]
        if self.time_mode == "calendar":
            tau = torch.linspace(0.0, 1.0, n_t, device=env.device, dtype=env.dtype)
            return tau.unsqueeze(0).expand(B, -1), None
        if s is None:
            raise ValueError(f"time_mode={self.time_mode} 에는 열시간 s 가 필요합니다.")
        return s, None

    def forward(self, g_idx, env, s=None, ds=None, encoder_days=0):
        """
        g_idx        : [B]        long
        env          : [B, T, E]  표준화된 환경 입력 (기온)
        s, ds        : [B, T]     정규화된 열시간과 그 진행 속도 (time_mode 가
                                  'thermal_*' 일 때 필요)
        encoder_days : 0 이면 인코더가 기온 전 구간을 읽는다. 양수면 앞 N일만 읽어
                       인과적(causal) 예측 설정이 된다.
        반환         : dict(pred [B,T], z [B,T,L], r [B], y_max [B], mu, logvar)
        """
        n_t = env.shape[1]
        g_emb = self.g_enc(g_idx)
        t_feat, _ = self.time_features(env, s)

        z0, mu, logvar = self.get_z0(g_emb, env, t_feat, encoder_days)

        t_interp = Interp(t_feat) if self.time_mode != "calendar" else None
        rate_interp = None
        if self.time_mode == "thermal_rate":
            if ds is None:
                raise ValueError("time_mode='thermal_rate' 에는 ds 가 필요합니다.")
            rate_interp = Interp(ds)

        z_traj = odeint_rk4(
            self.ode_func, z0, n_t, Interp(env), g_emb, self.n_substeps,
            t_interp=t_interp, rate_interp=rate_interp,
        )
        pred = self.decode(z_traj)

        out = {"pred": pred, "z": z_traj, "g_emb": g_emb, "mu": mu, "logvar": logvar,
               "t_feat": t_feat, "ds": ds}
        if self.use_physics:
            r, y_max = self.ode_param_head(g_emb)
            out["r"], out["y_max"] = r, y_max
        return out

    # ---------------------------------------------------------------- #
    def dydt(self, z_traj, env, g_emb, t_feat=None, ds=None):
        """연쇄법칙으로 dŷ/dt (단위: m/day) 를 구한다.

        dŷ/dτ = (∂dec/∂z) · dz/dτ,  dŷ/dt = dŷ/dτ / days_per_tau
        thermal_rate 모드에서는 dz/dτ 에 ds/dτ 가 곱해진 것을 반영한다.
        격자점에서만 평가하므로 보간이 필요 없다.
        """
        B, T, L = z_traj.shape
        if t_feat is None:
            t_feat = torch.linspace(
                0.0, 1.0, T, device=z_traj.device, dtype=z_traj.dtype
            ).unsqueeze(0).expand(B, -1)

        z_flat = z_traj.reshape(B * T, L)
        env_flat = env.reshape(B * T, env.shape[-1])
        g_flat = g_emb.unsqueeze(1).expand(B, T, g_emb.shape[-1]).reshape(B * T, -1)
        t_flat = t_feat.reshape(B * T, 1)

        h = torch.cat([z_flat, env_flat, g_flat, t_flat], dim=-1)
        dz = self.ode_func.out_scale * torch.tanh(self.ode_func.net(h))  # [B*T, L]
        if self.time_mode == "thermal_rate":
            if ds is None:
                raise ValueError("thermal_rate 모드의 물리 손실에는 ds 가 필요합니다.")
            dz = dz * ds.reshape(B * T, 1)

        y = self.decode(z_flat)                                          # [B*T]
        (grad_z,) = torch.autograd.grad(y.sum(), z_traj, create_graph=True)
        grad_z = grad_z.reshape(B * T, L)

        dy_dtau = (grad_z * dz).sum(dim=-1)
        return (dy_dtau / self.days_per_tau).reshape(B, T), y.reshape(B, T)


# --------------------------------------------------------------------------- #
# 손실
# --------------------------------------------------------------------------- #
def masked_rmse_loss(pred, y, mask):
    """논문 Eq.(3): 시퀀스별 RMSE 후 시퀀스 평균."""
    err = ((pred - y) ** 2) * mask
    denom = mask.sum(dim=-1).clamp_min(1e-8)
    return torch.sqrt(err.sum(dim=-1) / denom).mean()


def physics_losses(model, out, env, weights):
    """물리 기반 정칙화 항.

    L_mono : 단조 증가 제약. dŷ/dt < 0 인 구간만 벌점. 로지스틱 ODE보다 훨씬
             단순하지만 "줄기 신장은 되돌아가지 않는다"는 더 확실한 사전지식이다.
             (주의: 후기 도복이나 측정 노이즈로 실제 곡선도 국소적으로 감소할 수
             있으므로 hard constraint 가 아니라 약한 벌점으로 둔다.)
    L_m    : 로지스틱 ODE 미분 손실 (논문 Eq. 5)
    L_r    : 성장률 r 이 양수를 유지하도록 하는 벌점 (논문 Eq. 6)
    L_y    : 예측 최대 높이와 유전자형별 y_max 의 잔차 (논문 Eq. 7)
    """
    z, g_emb = out["z"], out["g_emb"]
    dy_dt, y_hat = model.dydt(z, env, g_emb, out.get("t_feat"), out.get("ds"))

    losses = {}
    if weights.get("mono", 0.0) > 0:
        losses["L_mono"] = weights["mono"] * torch.relu(-dy_dt).mean()

    if "r" in out:
        r, y_max = out["r"], out["y_max"]
        rhs = r.unsqueeze(1) * y_hat * (1.0 - y_hat / y_max.unsqueeze(1).clamp_min(1e-3))
        losses["L_m"] = weights["physic"] * ((dy_dt - rhs) ** 2).mean()

        alpha = 1e4
        losses["L_r"] = weights["r"] * (1.0 / (10.0 * alpha ** r)).mean()
        losses["L_y"] = weights["ymax"] * (y_hat.max(dim=1).values - y_max).abs().mean()
    return losses


def kl_divergence(mu, logvar):
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(dim=-1).mean()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# 스모크 테스트: python model.py
#   실제 데이터 없이 형상/기울기 흐름/물리 손실 계산이 되는지 확인한다.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, E, G = 6, 170, 1, 19
    g_idx = torch.randint(0, G, (B,))
    env = torch.randn(B, T, E)
    y = torch.rand(B, T) * 0.8
    mask = (torch.rand(B, T) > 0.7).float()
    kin = torch.randn(G, G).numpy()

    # 합성 열시간: 대략 0→1 로 단조 증가, 진행 속도는 후반에 빠름
    daily = torch.rand(B, T) * 0.5 + torch.linspace(0.2, 1.8, T).unsqueeze(0)
    s_t = torch.cumsum(daily, dim=1); s_t = s_t / s_t[:, -1:]
    ds_t = daily * (T - 1) / torch.cumsum(daily, dim=1)[:, -1:]

    for init_mode, enc, tmode, enc_days in [
        ("env_encoder", "one_hot", "calendar", 0),
        ("env_encoder", "kinship", "thermal_feature", 60),
        ("env_encoder", "one_hot", "thermal_rate", 0),
        ("genotype", "one_hot", "thermal_rate", 0),
    ]:
        m = LatentODEHeightModel(
            n_genotypes=G, env_dim=E, init_mode=init_mode, time_mode=tmode,
            genetic_encoding=enc, kinship=kin,
            variational=(init_mode == "env_encoder"),
        )
        out = m(g_idx, env, s_t, ds_t, enc_days)
        assert out["pred"].shape == (B, T), out["pred"].shape
        assert out["z"].shape == (B, T, m.latent_dim)

        loss = masked_rmse_loss(out["pred"], y, mask)
        pl = physics_losses(m, out, env,
                            {"physic": 2.0, "r": 1.0, "ymax": 0.1, "mono": 10.0})
        total = loss + sum(pl.values())
        if m.variational:
            total = total + 1e-3 * kl_divergence(out["mu"], out["logvar"])
        total.backward()

        n_no_grad = [n for n, p in m.named_parameters()
                     if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
        print(f"[{init_mode:11s}/{enc:7s}/{tmode:15s}] params={count_parameters(m):5d} "
              f"data={loss.item():.4f} "
              + " ".join(f"{k}={v.item():.5f}" for k, v in pl.items())
              + f" nfe={m.ode_func.nfe} "
              f"grad 없는 파라미터: {n_no_grad if n_no_grad else '없음'}")
    print("스모크 테스트 통과")