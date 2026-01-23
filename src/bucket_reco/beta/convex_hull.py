from __future__ import annotations

import logging
from math import gamma as gamma_func
from math import pi
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog

H_MIN = 1e-12


def sphere_area(dim: int) -> float:
    return float(2.0 * (pi ** (dim / 2.0)) / gamma_func(dim / 2.0))


def sample_sphere_uniform(dim: int, M: int, rng: np.random.Generator) -> np.ndarray:
    G = rng.normal(size=(M, dim))
    norms = np.linalg.norm(G, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return G / norms


def dual_lp_support_theta(
    A: np.ndarray, theta: np.ndarray
) -> Tuple[float, Optional[np.ndarray], bool]:
    d = A.shape[1]
    c = -theta.astype(float)
    A_ub = A.astype(float)
    b_ub = np.ones(A.shape[0], dtype=float)
    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * d, method="highs")
    if res.status == 0:
        y_opt = res.x
        h = float(theta @ y_opt)
        return h, y_opt, True
    if res.status == 3:
        return np.inf, None, False
    return np.inf, None, False


def pivoted_qr_init_indices(X: np.ndarray, dim: int) -> List[int]:
    try:
        from scipy.linalg import qr as scipy_qr

        _, _, piv = scipy_qr(X.T, mode="economic", pivoting=True)
        k = min(dim, len(piv))
        return list(piv[:k])
    except Exception:
        N = X.shape[0]
        idxs: List[int] = []
        V = np.empty((0, dim))
        for _ in range(min(dim, N)):
            best_i, best_score = -1, -1.0
            for i in range(N):
                if i in idxs:
                    continue
                v = X[i]
                if V.shape[0] > 0:
                    coef, *_ = np.linalg.lstsq(V.T, v, rcond=None)
                    proj = V.T @ coef
                    resid = v - proj
                else:
                    resid = v
                sc = float(np.linalg.norm(resid))
                if sc > best_score:
                    best_score = sc
                    best_i = i
            if best_i >= 0:
                idxs.append(best_i)
                V = X[best_i : best_i + 1] if V.shape[0] == 0 else np.vstack([V, X[best_i]])
        return idxs


class RadialVolumeAccumulator:
    def __init__(self, dim: int, rho: np.ndarray):
        self.dim = int(dim)
        self.rho = np.asarray(rho, dtype=float)
        self.M = self.rho.size
        self.rho_pow = self.rho**self.dim
        self.mean_rho_pow = float(self.rho_pow.mean())
        self.area_over_d = sphere_area(self.dim) / self.dim

    def volume(self) -> float:
        return self.area_over_d * self.mean_rho_pow

    def delta_from_update(self, idx: np.ndarray, new_rho_vals: np.ndarray) -> float:
        if np.size(idx) == 0:
            return 0.0
        new_pow = (np.asarray(new_rho_vals, float)) ** self.dim
        old_pow = self.rho_pow[idx]
        sum_diff = float((new_pow - old_pow).sum())
        return self.area_over_d * (sum_diff / self.M)

    def apply_update(self, idx: np.ndarray, new_rho_vals: np.ndarray) -> None:
        if np.size(idx) == 0:
            return
        new_pow = (np.asarray(new_rho_vals, float)) ** self.dim
        old_pow = self.rho_pow[idx]
        sum_diff = float((new_pow - old_pow).sum())
        self.rho[idx] = np.asarray(new_rho_vals, float)
        self.rho_pow[idx] = new_pow
        self.mean_rho_pow += sum_diff / self.M


def _get_logger(logger: Optional[logging.Logger], debug: bool) -> logging.Logger:
    if logger is not None:
        return logger
    log = logging.getLogger(__name__)
    log.propagate = True
    log.setLevel(logging.DEBUG if debug else logging.NOTSET)
    return log


def select_representatives(
    X: np.ndarray,
    epsilon: float,
    *,
    M: int = 4096,
    rng_seed: int = 42,
    topk_per_iter: Optional[int] = 64,
    violation_tol: float = 1e-9,
    max_iters: Optional[int] = None,
    clip_rhopow: Optional[float] = None,
    clip_viol: Optional[float] = None,
    debug: bool = False,
    logger: Optional[logging.Logger] = None,
    log_topk: int = 5,
    diversity_beta: float = 1.5,
    whitelist_idx: Optional[List[int]] = None,
    nms_cos_thresh: Optional[float] = 0.98,
    labels: Optional[List[str]] = None,
) -> np.ndarray:
    log = _get_logger(logger, debug)

    assert X.ndim == 2 and X.shape[0] >= 1
    N0, d = X.shape
    rng = np.random.default_rng(rng_seed)

    norms = np.linalg.norm(X, axis=1)
    keep = norms > 0.0
    orig_idx_map = np.arange(N0)[keep]
    X = X[keep]
    if X.shape[0] == 0:
        return np.zeros((0,), dtype=int)

    thetas = sample_sphere_uniform(d, M, rng)

    init_idxs = pivoted_qr_init_indices(X, d)
    S_idx: List[int] = list(init_idxs)
    S = X[S_idx]
    A = S.copy()
    used_mask = np.zeros(X.shape[0], dtype=bool)
    used_mask[S_idx] = True
    disabled_mask = np.zeros(X.shape[0], dtype=bool)

    if whitelist_idx is not None and len(whitelist_idx) > 0:
        wl_orig = [int(i) for i in whitelist_idx if 0 <= int(i) < N0]
        inv_map = {int(oi): int(fi) for fi, oi in enumerate(orig_idx_map.tolist())}
        wl_filtered: List[int] = []
        seen = set()
        for oi in wl_orig:
            if oi in inv_map:
                fi = inv_map[oi]
                if fi not in seen and not used_mask[fi]:
                    wl_filtered.append(fi)
                    seen.add(fi)

        if debug and len(wl_filtered) != len(wl_orig):
            log.debug(
                f"Whitelist: requested={len(wl_orig)}, accepted={len(wl_filtered)} "
                f"(invalid/zero/dup skipped)"
            )

        if wl_filtered:
            for fi in wl_filtered:
                S_idx.append(int(fi))
                used_mask[fi] = True
            S = X[S_idx]
            A = S.copy()

    rho = np.zeros(M, dtype=float)
    Y = np.zeros((M, d), dtype=float)
    bounded = np.zeros(M, dtype=bool)
    for m in range(M):
        if A.size == 0:
            rho[m] = 0.0
            bounded[m] = False
            continue
        h, y_opt, ok = dual_lp_support_theta(A, thetas[m])
        if ok:
            rho[m] = 1.0 / max(h, H_MIN)
            Y[m] = y_opt
            bounded[m] = True
        else:
            rho[m] = 0.0
            bounded[m] = False

    acc = RadialVolumeAccumulator(d, rho)
    vol = acc.volume()
    if debug and (whitelist_idx is not None and len(whitelist_idx) > 0):
        log.debug(
            f"After whitelist: |S|={len(S_idx)} bounded={bounded.sum()} "
            f"unbounded={(~bounded).sum()} vol≈{vol:.6g}"
        )
    log.debug(
        f"Init: N={X.shape[0]} d={d} M={M} rank_init=|S|={len(S_idx)} "
        f"bounded={bounded.sum()} unbounded={(~bounded).sum()} vol≈{vol:.6g}"
    )

    if nms_cos_thresh is not None:
        remain = np.where(~used_mask & ~disabled_mask)[0]
        if remain.size and len(S_idx) > 0:
            S_unit = S / (np.linalg.norm(S, axis=1, keepdims=True) + H_MIN)
            U = X[remain]
            U_unit = U / (np.linalg.norm(U, axis=1, keepdims=True) + H_MIN)
            maxcos = (U_unit @ S_unit.T).max(axis=1)
            kill = remain[maxcos > nms_cos_thresh]
            if kill.size:
                disabled_mask[kill] = True
                if debug:
                    log.debug(f"NMS after whitelist: suppressed={kill.size} @cos>{nms_cos_thresh}")

    iter_count = 0
    while True:
        iter_count += 1
        if max_iters is not None and iter_count > max_iters:
            log.debug("Stop: reached max_iters")
            break

        cand_idx = np.where(~used_mask & ~disabled_mask)[0]
        if cand_idx.size == 0:
            log.debug("Stop: no remaining candidates")
            break

        U = X[cand_idx]
        dots = U @ Y.T
        viol = np.maximum(dots - (1.0 + violation_tol), 0.0)

        rho_pow_d1 = acc.rho_pow * acc.rho
        if clip_rhopow is not None:
            rho_pow_d1 = np.minimum(rho_pow_d1, float(clip_rhopow))
        if clip_viol is not None:
            viol = np.minimum(viol, float(clip_viol))
        score_bounded = (viol * rho_pow_d1[None, :]).sum(axis=1)

        ub_mask = ~bounded
        if np.any(ub_mask):
            U_norm = np.linalg.norm(U, axis=1, keepdims=True) + H_MIN
            cos_align = np.maximum((U @ thetas[ub_mask].T) / U_norm, 0.0)
            score_unbd_raw = cos_align.sum(axis=1)
            lam = float(np.median(acc.rho_pow[acc.rho_pow > 0])) if np.any(acc.rho_pow > 0) else 1.0
            score = score_bounded + lam * score_unbd_raw
        else:
            score_unbd_raw = np.zeros(U.shape[0], dtype=float)
            score = score_bounded

            U_norm = np.linalg.norm(U, axis=1, keepdims=True) + H_MIN
            U_unit = U / U_norm
            S_unit = S / (np.linalg.norm(S, axis=1, keepdims=True) + H_MIN)
            cos_US = U_unit @ S_unit.T
            cos_US_pos = np.clip(cos_US, 0.0, 1.0)
            maxcos = cos_US_pos.max(axis=1) if cos_US_pos.size else 0.0
            w_div = (1.0 - maxcos) ** diversity_beta + H_MIN
            score *= w_div

        score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max / 4, neginf=0.0)

        order_main = np.argsort(-score)
        reserve_unbounded = 0 if not np.any(ub_mask) else max(5, (topk_per_iter or 64) // 4)
        order_unbd = np.argsort(-score_unbd_raw)[:reserve_unbounded]

        order_list = np.concatenate([order_unbd, order_main]).tolist()
        seen_order: set[int] = set()
        unique_order: list[int] = []
        for i in order_list:
            if i in seen_order:
                continue
            seen_order.add(i)
            unique_order.append(i)
        order = np.array(unique_order)

        if topk_per_iter is not None:
            order = order[: min(topk_per_iter, order.size)]

        cand_idx_ordered = cand_idx[order]

        if debug:
            kshow = min(log_topk, cand_idx_ordered.size)
            log.debug(
                f"[Iter {iter_count}] candidates={cand_idx.size} "
                f"bounded={bounded.sum()} unbounded={(~bounded).sum()} "
                f"vol≈{vol:.6g}"
            )
            if kshow > 0:
                top_ids = cand_idx_ordered[:kshow]
                sb = score_bounded[order][:kshow]
                st = score[order][:kshow]
                log.debug(
                    "  TopK (idx_in_X, score_bounded, total): "
                    + ", ".join(
                        [f"({int(i)}, {b:.3g}, {t:.3g})" for i, b, t in zip(top_ids, sb, st)]
                    )
                )

        if cand_idx_ordered.size == 0 and not np.any(ub_mask):
            log.debug("Stop: no candidate passes coarse screen and no unbounded directions")
            break

        best_delta = -np.inf
        best_i = None
        best_idx_m = None
        best_new_rho_m = None
        best_new_Y_m = None
        best_new_bounded_flags = None

        total_lp = 0
        for idx_i in cand_idx_ordered:
            u = X[idx_i]
            dots_u = u @ Y.T
            mask_vio = dots_u > (1.0 + violation_tol)
            if np.any(ub_mask):
                ub_idx = np.where(ub_mask)[0]
                u_norm = np.linalg.norm(u) + 1e-12
                align_mask = (thetas[ub_idx] @ u) / u_norm > 0.1
                if np.any(align_mask):
                    mask_extra = np.zeros(M, dtype=bool)
                    mask_extra[ub_idx[align_mask]] = True
                    mask_vio = np.logical_or(mask_vio, mask_extra)

            if not np.any(mask_vio):
                continue

            idx_m = np.where(mask_vio)[0]
            A_aug = np.vstack([A, u.reshape(1, -1)])
            new_rho_m = np.empty(idx_m.size, dtype=float)
            new_Y_m = np.empty((idx_m.size, d), dtype=float)
            new_bounded_flags = np.zeros(idx_m.size, dtype=bool)

            if debug:
                nb_v = int((mask_vio & bounded).sum())
                nu_v = int((mask_vio & (~bounded)).sum())

            for j, m in enumerate(idx_m):
                h2, y2, ok2 = dual_lp_support_theta(A_aug, thetas[m])
                if ok2:
                    new_rho_m[j] = 1.0 / max(h2, H_MIN)
                    new_Y_m[j] = y2
                    new_bounded_flags[j] = True
                else:
                    new_rho_m[j] = 0.0
                    new_Y_m[j] = Y[m]
                    new_bounded_flags[j] = False
            total_lp += idx_m.size

            delta = acc.delta_from_update(idx_m, new_rho_m)

            if debug:
                log.debug(
                    f"    cand={int(idx_i)}  LPs={idx_m.size} "
                    f"(bounded_cut={nb_v}, unbounded_try={nu_v})  Δ≈{delta:.6g}"
                )

            if delta > best_delta:
                best_delta = float(delta)
                best_i = int(idx_i)
                best_idx_m = idx_m
                best_new_rho_m = new_rho_m
                best_new_Y_m = new_Y_m
                best_new_bounded_flags = new_bounded_flags

        if best_i is None or best_delta <= 0.0:
            log.debug("Stop: no candidate yields positive Δ in fine evaluation")
            break
        if (
            best_idx_m is None
            or best_new_rho_m is None
            or best_new_Y_m is None
            or best_new_bounded_flags is None
        ):
            log.debug("Stop: missing candidate state")
            break

        rel_gain = best_delta / max(vol, H_MIN)
        if labels is not None and 0 <= best_i < orig_idx_map.size:
            orig_idx = int(orig_idx_map[best_i])
            label = labels[orig_idx] if 0 <= orig_idx < len(labels) else None
        else:
            label = None
        pick_msg = (
            f"[Iter {iter_count}] pick idx={best_i}"
            f"{'' if label is None else f' ({label})'}  Δ≈{best_delta:.6g} "
            f"rel_gain≈{rel_gain:.3%}  LP_solved={total_lp}"
        )
        log.info(pick_msg)

        if rel_gain < epsilon:
            log.debug(f"Stop: relative gain {rel_gain:.3%} < epsilon {epsilon:.3%}")
            break

        used_mask[best_i] = True
        S_idx.append(best_i)
        S = np.vstack([S, X[best_i]])
        A = S.copy()
        acc.apply_update(best_idx_m, best_new_rho_m)
        Y[best_idx_m] = best_new_Y_m
        bounded[best_idx_m] = best_new_bounded_flags
        vol = acc.volume()

        ub_rest = np.where(~bounded)[0]
        if ub_rest.size:
            new_rho_rest = np.empty(ub_rest.size, dtype=float)
            new_Y_rest = np.empty((ub_rest.size, d), dtype=float)
            new_bounded_rest = np.zeros(ub_rest.size, dtype=bool)
            for j, m in enumerate(ub_rest):
                h3, y3, ok3 = dual_lp_support_theta(A, thetas[m])
                if ok3:
                    new_rho_rest[j] = 1.0 / max(h3, H_MIN)
                    new_Y_rest[j] = y3
                    new_bounded_rest[j] = True
                else:
                    new_rho_rest[j] = 0.0
                    new_Y_rest[j] = Y[m]
                    new_bounded_rest[j] = False

            acc.apply_update(ub_rest, new_rho_rest)
            Y[ub_rest] = new_Y_rest
            bounded[ub_rest] = new_bounded_rest
            vol = acc.volume()
            if debug:
                n_fixed = int(new_bounded_rest.sum())
                log.debug(
                    f"    refreshed unbounded={ub_rest.size}, now fixed={n_fixed}, "
                    f"bounded={bounded.sum()}"
                )
        log.debug(f"    |S|={len(S_idx)}  new_vol≈{vol:.6g}")

        if nms_cos_thresh is not None:
            remain = np.where(~used_mask & ~disabled_mask)[0]
            if remain.size:
                S_unit = S / (np.linalg.norm(S, axis=1, keepdims=True) + H_MIN)
                U = X[remain]
                U_unit = U / (np.linalg.norm(U, axis=1, keepdims=True) + H_MIN)
                maxcos = (U_unit @ S_unit.T).max(axis=1)
                kill = remain[maxcos > nms_cos_thresh]
                if kill.size:
                    disabled_mask[kill] = True
                    if debug:
                        log.debug(f"    NMS: suppressed={kill.size} @cos>{nms_cos_thresh}")

    log.debug(f"Done. selected |S|={len(S_idx)}")

    selected_idx = np.asarray(orig_idx_map[np.array(S_idx, dtype=int)], dtype=int)
    return selected_idx
