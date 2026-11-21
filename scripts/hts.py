"""HTS - Hierarchical Thermal-Survival model with campaign-level latent effects.

Fixed (transferable) part
-------------------------
Temperature T plays the role of "time"; a unit of dry feed moves through the states

    solid --devolatilisation--> volatiles --phi--> tar/oil --secondary cracking--> gas
      +-- refractory ("cured") fraction pi never leaves the solid state

    Y_char(T, x) = 100 [ pi + (1 - pi) S_d(T) ],      S_d = 1 / (1 + exp((ln T - mu_d) / sigma_d))
    Y_oil (T, x) = 100 (1 - pi)(1 - S_d) phi S_k(T),  S_k = 1 / (1 + exp((ln T - mu_k) / sigma_k))
    Y_gas        = 100 - Y_char - Y_oil >= 0           (mass balance closed by construction)

(pi, mu_d, sigma_d, phi, mu_k, sigma_k) are neural functions of the non-temperature inputs x.

Random (laboratory / campaign) part
-----------------------------------
A standardised latent vector u_c ~ N(0, I) per experimental campaign (one feedstock sample in
one study) shifts the link-scale survival parameters:

    link(pi)     += s1 u1   char retention / refractory fraction offset
    link(T50_d)  += s2 u2   thermal lag (thermocouple position, heat-transfer limitation)
    link(phi)    += s3 u3   condensate recovery of the rig
    link(T50_k)  += s4 u4   vapour residence / secondary-cracking severity

All six survival parameters use bounded links (physically admissible ranges, see T50D ... SIGK)
so that an out-of-distribution feedstock cannot push a parameter to a non-physical value.

Training   : joint MAP estimation of network weights and campaign latents.
Zero-shot  : u = 0 for an unseen campaign.
k-shot     : u estimated by MAP from k anchor runs with the network frozen (calibrate_batch).

structure="plain" replaces the survival structure by an unconstrained network (T is an ordinary
input) with additive campaign offsets on the two outputs - the structure ablation.
"""
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(2)
RE_INDEX = {"pi": 0, "mu_d": 1, "phi": 3, "mu_k": 4}
RE_SCALE = {"pi": 0.6, "mu_d": 0.4, "phi": 0.8, "mu_k": 0.5}      # link-scale sd of each latent
# Physically admissible parameter ranges (bounded links prevent runaway extrapolation):
#   T50 of primary devolatilisation 180-460 degC, of secondary cracking 450-1400 degC,
#   log-temperature breadths 0.12-0.60 and 0.08-0.80, refractory fraction and oil selectivity in (0, 1).
T50D, T50K = (180.0, 460.0), (450.0, 1400.0)
SIGD, SIGK = (0.12, 0.60), (0.08, 0.80)
BIAS0 = [-1.1, 0.60, -1.0, 0.4, -0.20, -0.75]   # pi~0.25, T50_d~330 C, sigma_d~0.25, phi~0.6, T50_k~750 C, sigma_k~0.30


def surv_params(h):
    """Map unconstrained link values to (pi, ln T50_d, sigma_d, phi, ln T50_k, sigma_k)."""
    pi = torch.sigmoid(h[:, 0])
    mu_d = np.log(T50D[0]) + np.log(T50D[1] / T50D[0]) * torch.sigmoid(h[:, 1])
    sig_d = SIGD[0] + (SIGD[1] - SIGD[0]) * torch.sigmoid(h[:, 2])
    phi = torch.sigmoid(h[:, 3])
    mu_k = np.log(T50K[0]) + np.log(T50K[1] / T50K[0]) * torch.sigmoid(h[:, 4])
    sig_k = SIGK[0] + (SIGK[1] - SIGK[0]) * torch.sigmoid(h[:, 5])
    return pi, mu_d, sig_d, phi, mu_k, sig_k


class _HNet(nn.Module):
    def __init__(self, d_in, n_camp, hidden=64, depth=2, dropout=0.10,
                 re=("pi", "mu_d", "phi", "mu_k"), structure="surv"):
        super().__init__()
        self.structure = structure
        layers, d = [], d_in + (1 if structure == "plain" else 0)
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU(), nn.Dropout(dropout)]
            d = hidden
        self.trunk = nn.Sequential(*layers)
        if structure == "surv":
            self.head = nn.Linear(d, 6)
            self.register_buffer("bias0", torch.tensor(BIAS0, dtype=torch.float32))
            self.re_idx = [RE_INDEX[r] for r in re]
            self.register_buffer("re_scale", torch.tensor([RE_SCALE[r] for r in re], dtype=torch.float32))
            nn.init.normal_(self.head.weight, std=0.01)
        else:
            self.head = nn.Linear(d, 2)
            self.register_buffer("bias0", torch.tensor([33.0, 35.0], dtype=torch.float32))
            self.re_idx = [0, 1] if len(re) else []
            self.register_buffer("re_scale", torch.tensor([6.0, 8.0][:len(self.re_idx)], dtype=torch.float32))
        nn.init.zeros_(self.head.bias)
        self.u = nn.Embedding(max(n_camp, 1), max(len(self.re_idx), 1))
        nn.init.zeros_(self.u.weight)

    def base(self, x, lnT):
        if self.structure == "plain":
            x = torch.cat([x, ((lnT - 6.2) / 0.25).unsqueeze(1)], 1)
        return self.head(self.trunk(x)) + self.bias0

    def add_u(self, h, u):
        if u is None or not len(self.re_idx):
            return h
        add = torch.zeros_like(h)
        add[:, self.re_idx] = u * self.re_scale
        return h + add

    def yields(self, h, lnT):
        if self.structure == "plain":
            return h[:, 0], h[:, 1]
        pi, mu_d, sig_d, phi, mu_k, sig_k = surv_params(h)
        s_d = torch.sigmoid(-(lnT - mu_d) / sig_d)
        s_k = torch.sigmoid(-(lnT - mu_k) / sig_k)
        return 100.0 * (pi + (1 - pi) * s_d), 100.0 * (1 - pi) * (1 - s_d) * phi * s_k


class HTS:
    def __init__(self, hidden=64, depth=2, dropout=0.15, lr=3e-3, weight_decay=1e-3, epochs=300,
                 n_ens=5, batch=256, seed=0, prior=4.0, re=("pi", "mu_d", "phi", "mu_k"), delta=8.0,
                 structure="surv", clip=2.5):
        self.kw = dict(hidden=hidden, depth=depth, dropout=dropout, re=tuple(re), structure=structure)
        self.lr, self.wd, self.epochs, self.n_ens = lr, weight_decay, epochs, n_ens
        self.batch, self.seed, self.prior, self.delta, self.clip = batch, seed, prior, delta, clip

    @property
    def n_re(self):
        return len(self.nets_[0].re_idx)

    # ---- fold-internal preprocessing --------------------------------------
    def _impute(self, X):
        return np.hstack([np.where(np.isnan(X), self.med_, X), np.isnan(X)[:, self.has_nan_].astype(float)])

    def _tensors(self, X, fit=False):
        X = np.asarray(X, float).copy()
        for c in self.log_cols_:
            X[:, c] = np.log1p(X[:, c])
        lnT = np.log(X[:, self.t_col_])
        Xo = np.delete(X, self.t_col_, axis=1)
        if fit:
            self.med_ = np.nanmedian(Xo, axis=0)
            self.has_nan_ = np.isnan(Xo).any(axis=0)
            Zi = self._impute(Xo)
            self.mean_, self.std_ = Zi.mean(0), Zi.std(0) + 1e-8
        Z = np.clip((self._impute(Xo) - self.mean_) / self.std_, -self.clip, self.clip)   # winsorised inputs
        return torch.tensor(Z, dtype=torch.float32), torch.tensor(lnT, dtype=torch.float32)

    def _loss(self, pc, po, yc, yo, mc, mo):
        loss = torch.zeros(())
        if mc.any():
            loss = loss + nn.functional.huber_loss(pc[mc], yc[mc], delta=self.delta, reduction="sum")
        if mo.any():
            loss = loss + nn.functional.huber_loss(po[mo], yo[mo], delta=self.delta, reduction="sum")
        return loss

    @staticmethod
    def _targets(y_char, y_oil):
        y_char, y_oil = np.asarray(y_char, float), np.asarray(y_oil, float)
        return (torch.tensor(np.nan_to_num(y_char), dtype=torch.float32),
                torch.tensor(np.nan_to_num(y_oil), dtype=torch.float32),
                torch.tensor(~np.isnan(y_char)), torch.tensor(~np.isnan(y_oil)))

    def fit(self, X, y_char, y_oil, camp, t_col, log_cols=()):
        self.t_col_, self.log_cols_ = t_col, tuple(log_cols)
        Z, lnT = self._tensors(X, fit=True)
        codes = np.unique(camp, return_inverse=True)[1]
        cidx = torch.tensor(codes, dtype=torch.long)
        yc, yo, mc, mo = self._targets(y_char, y_oil)
        n = len(Z)
        self.nets_ = []
        for e in range(self.n_ens):
            torch.manual_seed(self.seed * 1000 + e)
            net = _HNet(Z.shape[1], int(codes.max()) + 1, **self.kw)
            has_re = len(net.re_idx) > 0
            wparams = [p for nme, p in net.named_parameters() if not nme.startswith("u.")]
            opt = torch.optim.AdamW([{"params": wparams, "weight_decay": self.wd},
                                     {"params": net.u.parameters(), "weight_decay": 0.0}], lr=self.lr)
            steps = self.epochs * int(np.ceil(n / self.batch))
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps)
            g = torch.Generator().manual_seed(self.seed * 1000 + e)
            for _ in range(self.epochs):
                net.train()
                perm = torch.randperm(n, generator=g)
                for i in range(0, n, self.batch):
                    b = perm[i:i + self.batch]
                    h = net.add_u(net.base(Z[b], lnT[b]), net.u(cidx[b]) if has_re else None)
                    pc, po = net.yields(h, lnT[b])
                    loss = self._loss(pc, po, yc[b], yo[b], mc[b], mo[b]) / len(b)
                    if has_re:      # Gaussian prior on the latents (objective: sum Huber + prior * |u|^2)
                        loss = loss + self.prior * (net.u.weight ** 2).sum() / n
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                    opt.step()
                    sched.step()
            net.eval()
            self.nets_.append(net)
        self.train_codes_ = codes
        return self

    @torch.no_grad()
    def predict(self, X, U=None):
        """U: optional list (one per ensemble member) of (n_rows, n_re) standardised latents."""
        Z, lnT = self._tensors(X)
        outs = []
        for j, net in enumerate(self.nets_):
            u = None if U is None else torch.tensor(U[j], dtype=torch.float32)
            outs.append(torch.stack(net.yields(net.add_u(net.base(Z, lnT), u), lnT), 1).numpy())
        m = np.mean(outs, 0)
        return m[:, 0], m[:, 1]

    def calibrate_batch(self, X_anchor, yc_anchor, yo_anchor, pid, n_prob, steps=200, lr=0.05):
        """MAP latents for many independent campaigns at once (network frozen).
        pid[i] is the campaign/problem index of anchor row i.  Returns a list (per ensemble member)
        of (n_prob, n_re) arrays; problems without anchors keep u = 0."""
        k = self.n_re
        if k == 0 or len(X_anchor) == 0:
            return [np.zeros((n_prob, max(k, 1)), np.float32) for _ in self.nets_]
        Z, lnT = self._tensors(X_anchor)
        yc, yo, mc, mo = self._targets(yc_anchor, yo_anchor)
        pid_t = torch.tensor(np.asarray(pid), dtype=torch.long)
        out = []
        for net in self.nets_:
            with torch.no_grad():
                base = net.base(Z, lnT)
            U = torch.zeros(n_prob, k, requires_grad=True)
            opt = torch.optim.Adam([U], lr=lr)
            for _ in range(steps):
                pc, po = net.yields(net.add_u(base, U[pid_t]), lnT)
                loss = self._loss(pc, po, yc, yo, mc, mo) + self.prior * (U ** 2).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
            out.append(U.detach().numpy())
        return out

    def train_latents(self):
        """Fitted latents of the training campaigns, averaged over ensemble members."""
        return np.mean([net.u.weight.detach().numpy() for net in self.nets_], 0)

    @torch.no_grad()
    def descriptors(self, X, U=None):
        """pi, T50_d (deg C), sigma_d, phi, T50_k (deg C), sigma_k  (survival structure only)."""
        Z, lnT = self._tensors(X)
        P = []
        for j, net in enumerate(self.nets_):
            u = None if U is None else torch.tensor(U[j], dtype=torch.float32)
            h = net.add_u(net.base(Z, lnT), u)
            pi, mu_d, sig_d, phi, mu_k, sig_k = surv_params(h)
            P.append(torch.stack([pi, torch.exp(mu_d), sig_d, phi, torch.exp(mu_k), sig_k], 1).numpy())
        return np.mean(P, 0)
