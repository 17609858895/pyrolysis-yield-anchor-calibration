"""HTS-Boost - hierarchical thermal-survival model learned by Gauss-Newton gradient boosting.

Same structural equations as hts.py (mixture-cure AFT for char, competing-risk split with
secondary cracking for oil, mass balance closed by construction, bounded physical links), but
the six link functions are additive ensembles of shallow regression trees instead of a neural
network.  Trees are piecewise constant, so an out-of-distribution feedstock receives the
parameters of its nearest leaf rather than an extrapolated value.

Per boosting round m (functional Gauss-Newton step in link space):
    r      = y_hat(F + S u_c) - y                        residuals of the observed products
    J      = d y_hat / d link                             (n x 6) for char and for oil
    g_j    = sum_products J_j * psi(r),  h_j = sum_products J_j^2
    tree_j : weighted least squares fit to -g_j / h_j with weights h_j   (Newton leaf values)
    F_j   += eta * tree_j(x)
    u_c   <- damped Gauss-Newton step on the campaign latents with Gaussian prior
psi is the Huber influence function (clipped residual).

Zero-shot: u = 0.  k-shot: u from damped Gauss-Newton MAP on the anchors with F frozen.
"""
import numpy as np
import torch
from sklearn.tree import DecisionTreeRegressor
from hts import RE_INDEX, RE_SCALE, BIAS0, surv_params

torch.set_num_threads(2)


def _yields_np(H, lnT, need_jac=True):
    """Yields and Jacobians w.r.t. the six link values (row-wise independent)."""
    Ht = torch.tensor(H, dtype=torch.float64, requires_grad=need_jac)
    lt = torch.tensor(lnT, dtype=torch.float64)
    pi, mu_d, sig_d, phi, mu_k, sig_k = surv_params(Ht)
    s_d = torch.sigmoid(-(lt - mu_d) / sig_d)
    s_k = torch.sigmoid(-(lt - mu_k) / sig_k)
    yc = 100.0 * (pi + (1 - pi) * s_d)
    yo = 100.0 * (1 - pi) * (1 - s_d) * phi * s_k
    if not need_jac:
        return yc.detach().numpy(), yo.detach().numpy()
    Jc = torch.autograd.grad(yc.sum(), Ht, retain_graph=True)[0].numpy()
    Jo = torch.autograd.grad(yo.sum(), Ht)[0].numpy()
    return yc.detach().numpy(), yo.detach().numpy(), Jc, Jo


class HTSBoost:
    def __init__(self, n_rounds=300, lr=0.10, max_depth=3, min_leaf=15, subsample=0.8, n_ens=3, seed=0,
                 prior=4.0, re=("pi", "mu_d", "phi", "mu_k"), delta=10.0, lr_u=0.3, max_features=0.8):
        self.M, self.lr, self.depth, self.min_leaf = n_rounds, lr, max_depth, min_leaf
        self.subsample, self.n_ens, self.seed, self.prior = subsample, n_ens, seed, prior
        self.re, self.delta, self.lr_u, self.max_features = tuple(re), delta, lr_u, max_features
        self.re_idx = [RE_INDEX[r] for r in self.re]
        self.re_scale = np.array([RE_SCALE[r] for r in self.re])

    @property
    def n_re(self):
        return len(self.re_idx)

    # ---- helpers -----------------------------------------------------------
    def _split(self, X, fit=False):
        X = np.asarray(X, float).copy()
        for c in self.log_cols_:
            X[:, c] = np.log1p(X[:, c])
        lnT = np.log(X[:, self.t_col_])
        Xo = np.delete(X, self.t_col_, axis=1)
        if fit:
            self.med_ = np.nanmedian(Xo, axis=0)
        # median imputation + missing indicators (fitted on the training fold)
        Xo = np.hstack([np.where(np.isnan(Xo), self.med_, Xo), np.isnan(Xo).astype(float)])
        return Xo, lnT

    def _add_u(self, F, U_rows):
        if U_rows is None or not self.n_re:
            return F
        H = F.copy()
        H[:, self.re_idx] += U_rows * self.re_scale
        return H

    def _psi(self, r):
        return np.clip(r, -self.delta, self.delta)

    def _latent_step(self, F, lnT, yc, yo, mc, mo, pid, U, lr_u):
        """One damped Gauss-Newton MAP step for all campaigns at once."""
        H = self._add_u(F, U[pid])
        pc, po, Jc, Jo = _yields_np(H, lnT)
        rc = np.where(mc, self._psi(pc - yc), 0.0)
        ro = np.where(mo, self._psi(po - yo), 0.0)
        Ju_c = Jc[:, self.re_idx] * self.re_scale * mc[:, None]
        Ju_o = Jo[:, self.re_idx] * self.re_scale * mo[:, None]
        k = self.n_re
        A = np.zeros((len(U), k, k))
        b = np.zeros((len(U), k))
        np.add.at(A, pid, Ju_c[:, :, None] * Ju_c[:, None, :] + Ju_o[:, :, None] * Ju_o[:, None, :])
        np.add.at(b, pid, Ju_c * rc[:, None] + Ju_o * ro[:, None])
        A += 2.0 * self.prior * np.eye(k)
        b += 2.0 * self.prior * U
        return U - lr_u * np.linalg.solve(A, b[:, :, None])[:, :, 0]

    # ---- training ----------------------------------------------------------
    def fit(self, X, y_char, y_oil, camp, t_col, log_cols=()):
        self.t_col_, self.log_cols_ = t_col, tuple(log_cols)
        Xo, lnT = self._split(X, fit=True)
        yc, yo = np.asarray(y_char, float), np.asarray(y_oil, float)
        mc, mo = ~np.isnan(yc), ~np.isnan(yo)
        yc0, yo0 = np.nan_to_num(yc), np.nan_to_num(yo)
        pid = np.unique(camp, return_inverse=True)[1]
        n_camp, n = pid.max() + 1, len(Xo)
        self.members_ = []
        for e in range(self.n_ens):
            rng = np.random.default_rng(self.seed * 1000 + e)
            F = np.tile(np.array(BIAS0, float), (n, 1))
            U = np.zeros((n_camp, max(self.n_re, 1)))
            trees = []
            for m in range(self.M):
                H = self._add_u(F, U[pid] if self.n_re else None)
                pc, po, Jc, Jo = _yields_np(H, lnT)
                rc = np.where(mc, self._psi(pc - yc0), 0.0)
                ro = np.where(mo, self._psi(po - yo0), 0.0)
                g = Jc * rc[:, None] + Jo * ro[:, None]
                h = Jc ** 2 * mc[:, None] + Jo ** 2 * mo[:, None]
                rows = rng.random(n) < self.subsample
                round_trees = []
                for j in range(6):
                    hj = h[:, j] + 1e-6
                    if hj[rows].sum() < 1e-3:               # parameter not identified by the data yet
                        round_trees.append(None)
                        continue
                    target = np.clip(-g[:, j] / hj, -2.0, 2.0)
                    t = DecisionTreeRegressor(max_depth=self.depth, min_samples_leaf=self.min_leaf,
                                              max_features=self.max_features,
                                              random_state=int(rng.integers(1 << 31)))
                    t.fit(Xo[rows], target[rows], sample_weight=hj[rows])
                    F[:, j] += self.lr * t.predict(Xo)
                    round_trees.append(t)
                trees.append(round_trees)
                if self.n_re:
                    U = self._latent_step(F, lnT, yc0, yo0, mc, mo, pid, U, self.lr_u)
            self.members_.append({"trees": trees, "U": U})
        self.train_codes_ = pid
        return self

    def _links(self, Xo, member):
        F = np.tile(np.array(BIAS0, float), (len(Xo), 1))
        for round_trees in member["trees"]:
            for j, t in enumerate(round_trees):
                if t is not None:
                    F[:, j] += self.lr * t.predict(Xo)
        return F

    # ---- inference ---------------------------------------------------------
    def predict(self, X, U=None):
        Xo, lnT = self._split(X)
        outs = []
        for e, mem in enumerate(self.members_):
            H = self._add_u(self._links(Xo, mem), None if U is None else U[e])
            outs.append(np.column_stack(_yields_np(H, lnT, need_jac=False)))
        m = np.mean(outs, 0)
        return m[:, 0], m[:, 1]

    def calibrate_batch(self, X_anchor, yc_anchor, yo_anchor, pid, n_prob, steps=25, lr_u=0.5):
        k = max(self.n_re, 1)
        if not self.n_re or len(X_anchor) == 0:
            return [np.zeros((n_prob, k)) for _ in self.members_]
        Xo, lnT = self._split(X_anchor)
        yc, yo = np.asarray(yc_anchor, float), np.asarray(yo_anchor, float)
        mc, mo = ~np.isnan(yc), ~np.isnan(yo)
        yc0, yo0 = np.nan_to_num(yc), np.nan_to_num(yo)
        pid = np.asarray(pid)
        out = []
        for mem in self.members_:
            F = self._links(Xo, mem)
            U = np.zeros((n_prob, k))
            for _ in range(steps):
                U = self._latent_step(F, lnT, yc0, yo0, mc, mo, pid, U, lr_u)
            out.append(U)
        return out

    def train_latents(self):
        return np.mean([m["U"] for m in self.members_], 0)

    def descriptors(self, X, U=None):
        Xo, _ = self._split(X)
        P = []
        for e, mem in enumerate(self.members_):
            H = torch.tensor(self._add_u(self._links(Xo, mem), None if U is None else U[e]))
            pi, mu_d, sig_d, phi, mu_k, sig_k = surv_params(H)
            P.append(torch.stack([pi, torch.exp(mu_d), sig_d, phi, torch.exp(mu_k), sig_k], 1).numpy())
        return np.mean(P, 0)
