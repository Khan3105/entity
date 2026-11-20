#!/usr/bin/env python3
"""
Read and plot the on-the-fly VDFs written by the KAW2D problem generator.

The simulation writes (see vdf_output.hpp)
    <sim>/vdf/vdf.<step>.npy    one file per dump (a single structured record)
    <sim>/vdf/vdf_meta.json     description: species labels, boxes, units, ...

Two file formats exist, and both are read:
    hist3d  [nsp, nbox, nb, nb, nb]  full 3D histogram over (v_x, v_y, v_z) (current)
    hist    [nsp, nbox, 3, nb, nb]   only the reduced planes xy, xz, yz (first test runs)
Reduced VDFs work for both; 3D VDFs and slices need hist3d.

Command line
------------
    python plot_vdf.py KAW2D_046                        # last dump, e-, box 0, (vx, vy)
    python plot_vdf.py KAW2D_046 --step 41000 --species 2 --box 1 --plane xz
    python plot_vdf.py KAW2D_046 --all                  # all boxes x all planes
    python plot_vdf.py KAW2D_046 --slice 0.25           # slice |v_z| < 0.125 instead of reduced
    python plot_vdf.py KAW2D_046 --slice 0.25 --slice-centre 1.0   # slice around v_z = 1
    python plot_vdf.py KAW2D_046 --diff                 # f(t) - f(first dump)
    python plot_vdf.py KAW2D_046 --log --vlim 4         # log10 f over the whole grid
    python plot_vdf.py KAW2D_046 --moments              # N, drift, temperature vs time
    python plot_vdf.py KAW2D_046 --save vdf.png         # write a file instead of showing

From Python
-----------
    from plot_vdf import list_dumps, load_dump, reduced_vdf, slice_vdf, vdf_3d, box_moments
    dumps = list_dumps("KAW2D_046")                     # [(step, path), ...]
    d = load_dump(dumps[-1][1])
    edges, f = reduced_vdf(d, species=1, box=0, plane="xy")   # f[i_vx, i_vy]
    edges, f3 = vdf_3d(d, species=1, box=0)                   # f3[i_vx, i_vy, i_vz]
    edges, fs, vz_range = slice_vdf(d, 1, 0, "xy", width=0.25)

Velocities are in units of the initial thermal speed v_th = sqrt(T0/m) of each
species. Strictly the code bins u = gamma*v, which equals v for v_th << c.
Times in the plots are shown as t * Omega_ci (see OMEGA_CI below); the dumps
themselves, and load_dump(...)["time"], stay in code units.
"""

import argparse
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np

# plane name -> (index in old-format files, first component, second component)
PLANES = {"xy": (0, "x", "y"), "xz": (1, "x", "z"), "yz": (2, "y", "z")}
# velocity component -> axis of the 3D histogram f3[i_x, i_y, i_z]
AXIS = {"x": 0, "y": 1, "z": 2}

# ion cyclotron frequency in code units (omegaCi0 in pgen.hpp: 0.004 for KAW2D
# with cAi0 = 0.02, mi = 25); every plotted time is t * OMEGA_CI
OMEGA_CI = 0.004


# ----------------------------------------------------------------- reading ---
def list_dumps(sim_dir):
    """Sorted list of (step, path) for all VDF dumps of a simulation."""
    vdf_dir = os.path.join(sim_dir, "vdf")
    dumps = []
    for name in os.listdir(vdf_dir):
        m = re.fullmatch(r"vdf\.(\d+)\.npy", name)
        if m:
            dumps.append((int(m.group(1)), os.path.join(vdf_dir, name)))
    return sorted(dumps)


def load_meta(sim_dir):
    """Contents of vdf_meta.json, or None if it is missing."""
    path = os.path.join(sim_dir, "vdf", "vdf_meta.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def load_dump(path, fields=None):
    """
    One dump as a dict of numpy arrays. Fields: step, time, species, vth, vmax,
    boxes, moments, and hist3d (or hist in old files; see vdf_output.hpp).
    With `fields`, only those are read (e.g. fields=("moments", "vth")).

    The 3D histogram is not read into memory here: d["hist3d"] is a read-only
    memory-mapped view of the file, so vdf_3d(d, species, box) reads only that
    species and box from disk.
    """
    arr = np.load(path, mmap_mode="r")
    names = arr.dtype.names if fields is None else ("step", "time", "species", *fields)
    d = {}
    for name in dict.fromkeys(names):
        col = arr[name][0]
        d[name] = col if name == "hist3d" else np.array(col)
    d["step"], d["time"] = int(d["step"]), float(d["time"])
    return d


def _isp(d, species):
    """Position of an Entity species index (1 = e-, 2 = i+) inside a dump."""
    recorded = [int(s) for s in d["species"]]
    if species not in recorded:
        raise ValueError(f"species {species} was not recorded (have {recorded})")
    return recorded.index(species)


# ---------------------------------------------------------------- analysis ---
def has_3d(d):
    """True if the dump holds the full 3D histogram (needed for slices)."""
    return "hist3d" in d


def velocity_edges(d, species):
    """Bin edges of the velocity grid, in units of v_th of that species."""
    vmax = d["vmax"][_isp(d, species)]
    nb = (d["hist3d"] if has_3d(d) else d["hist"]).shape[-1]
    return np.linspace(-vmax, vmax, nb + 1)


def _third_axis(plane):
    """Axis of f3[i_x, i_y, i_z] that is not in the plane ('xy' -> 2)."""
    _, a, b = PLANES[plane]
    return 3 - AXIS[a] - AXIS[b]


def vdf_3d(d, species, box):
    """
    Full 3D VDF f(v_x, v_y, v_z), velocities in units of v_th. Normalised so that
    sum(f) * dv**3 is the fraction of the particles in the box that fall inside
    the velocity grid (~1); a Maxwellian peaks at (2 pi)^(-3/2) = 0.0635.
    Returns (edges, f) with f[i_x, i_y, i_z]. Needs a dump with hist3d.
    """
    if not has_3d(d):
        raise ValueError("this dump only has the reduced 2D planes (old format): "
                         "3D VDFs and slices need a dump with hist3d")
    isp = _isp(d, species)
    counts = np.asarray(d["hist3d"][isp, box], dtype=np.float64)
    edges = velocity_edges(d, species)
    dv = edges[1] - edges[0]
    ntot = d["moments"][isp, box, 0]
    f = counts / (ntot * dv**3) if ntot > 0 else np.zeros_like(counts)
    return edges, f


def reduced_vdf(d, species, box, plane="xy"):
    """
    2D reduced VDF f(v_a, v_b) = integral of f over v_c, with velocities in units
    of v_th. Normalised so that sum(f) * dv**2 is the fraction of the particles in
    the box that fall inside the velocity grid (~1); a Maxwellian peaks at 1/(2 pi).
    Returns (edges, f) with f[i_a, i_b]. Works for both file formats.
    """
    if has_3d(d):
        edges, f3 = vdf_3d(d, species, box)
        return edges, f3.sum(axis=_third_axis(plane)) * (edges[1] - edges[0])
    isp = _isp(d, species)
    counts = d["hist"][isp, box, PLANES[plane][0]]
    edges = velocity_edges(d, species)
    dv = edges[1] - edges[0]
    ntot = d["moments"][isp, box, 0]
    f = counts / (ntot * dv**2) if ntot > 0 else np.zeros_like(counts)
    return edges, f


def slice_vdf(d, species, box, plane="xy", width=0.25, centre=0.0):
    """
    Slice of the 3D VDF: f(v_a, v_b, v_c) averaged over the v_c bins whose
    centres lie within centre +/- width/2 (at least the bin nearest to centre).
    Same units as vdf_3d, so a Maxwellian sliced through its centre peaks at
    (2 pi)^(-3/2) = 0.0635 (the reduced VDF peaks at 1/(2 pi)). For a slice
    through the bulk flow use centre = box_moments(d, species, box)["U"][axis].
    Returns (edges, f, (vc_lo, vc_hi)) with f[i_a, i_b] and the v_c range used.
    """
    edges, f3 = vdf_3d(d, species, box)
    c = _third_axis(plane)
    vc = 0.5 * (edges[1:] + edges[:-1])
    idx = np.flatnonzero(np.abs(vc - centre) <= 0.5 * width + 1e-9)
    if idx.size == 0:
        idx = np.array([np.argmin(np.abs(vc - centre))])
    f = np.take(f3, idx, axis=c).mean(axis=c)
    return edges, f, (edges[idx[0]], edges[idx[-1] + 1])


def reduced_vdf_1d(d, species, box, axis="x"):
    """1D reduced VDF f(v_axis), same normalisation, integrated from a 2D plane."""
    plane, other = {"x": ("xy", 1), "y": ("xy", 0), "z": ("xz", 0)}[axis]
    edges, f = reduced_vdf(d, species, box, plane)
    return edges, f.sum(axis=other) * (edges[1] - edges[0])


def box_moments(d, species, box):
    """
    Exact (unbinned) moments of the particles in a box:
      N : number of particles (sum of weights)
      U : bulk velocity [Ux, Uy, Uz] in units of v_th
      T : temperature tensor (3x3) in units of the initial temperature T0 = m v_th^2
    """
    isp = _isp(d, species)
    m = d["moments"][isp, box]
    n = m[0]
    if n <= 0:
        return {"N": 0.0, "U": np.full(3, np.nan), "T": np.full((3, 3), np.nan)}
    vth = d["vth"][isp]
    U = m[1:4] / n
    S = np.array([[m[4], m[7], m[8]], [m[7], m[5], m[9]], [m[8], m[9], m[6]]]) / n
    return {"N": n, "U": U / vth, "T": (S - np.outer(U, U)) / vth**2}


def smooth(f, passes=1):
    """1-2-1 filter in both directions, `passes` times (reduces counting noise)."""
    for _ in range(passes):
        g = np.pad(f, 1, mode="edge")
        f = 0.25 * g[:-2, 1:-1] + 0.5 * g[1:-1, 1:-1] + 0.25 * g[2:, 1:-1]
        g = np.pad(f, 1, mode="edge")
        f = 0.25 * g[1:-1, :-2] + 0.5 * g[1:-1, 1:-1] + 0.25 * g[1:-1, 2:]
    return f


# ---------------------------------------------------------------- plotting ---
def species_subscript(meta, species):
    """'e' / 'i' for the axis labels, taken from the species label."""
    for sp in (meta or {}).get("species", []):
        if sp["index"] == species:
            return sp["label"].rstrip("+-") or sp["label"]
    return str(species)


def box_title(d, box):
    x0, x1, y0, y1 = d["boxes"][box]
    return f"box {box}: x=[{x0:g}, {x1:g}], y=[{y0:g}, {y1:g}]"


def vdf_2d(d, species, box, plane="xy", slice_width=None, slice_centre=0.0):
    """reduced_vdf, or slice_vdf if slice_width is given. Returns (edges, f)."""
    if slice_width is None:
        return reduced_vdf(d, species, box, plane)
    edges, f, _ = slice_vdf(d, species, box, plane, slice_width, slice_centre)
    return edges, f


def plot_vdf(ax, d, species, box, plane="xy", vlim=2.5, nlevels=20, log=False,
             nsmooth=0, ref=None, sub="e", cmap="jet", slice_width=None,
             slice_centre=0.0):
    """
    Filled contours of f(v_a, v_b) on `ax`: the reduced VDF, or with
    `slice_width` a slice of the 3D VDF (see slice_vdf). Values below the lowest
    level are left white. With `ref` (another dump) the difference f - f_ref is
    shown. Returns the contour set (for a colorbar), or None if the box is empty.
    """
    sl = dict(slice_width=slice_width, slice_centre=slice_centre)
    edges, f = vdf_2d(d, species, box, plane, **sl)
    f = smooth(f, nsmooth)
    v = 0.5 * (edges[1:] + edges[:-1])
    _, a, b = PLANES[plane]
    cs = None
    if f.max() <= 0:
        ax.text(0.5, 0.5, "no particles", transform=ax.transAxes, ha="center")
    elif ref is not None:
        df = f - smooth(vdf_2d(ref, species, box, plane, **sl)[1], nsmooth)
        lim = np.abs(df).max()
        if lim > 0:
            levels = np.linspace(-lim, lim, 2 * nlevels + 1)
            cs = ax.contourf(v, v, df.T, levels=levels, cmap="RdBu_r")
    else:
        if log:
            with np.errstate(divide="ignore"):
                z = np.log10(f)
            levels = np.linspace(z.max() - 4.0, z.max(), nlevels + 1)  # 4 decades
        else:
            z = f
            levels = np.linspace(f.max() / nlevels, f.max(), nlevels)
        cs = ax.contourf(v, v, z.T, levels=levels, cmap=cmap)
        ax.contour(v, v, z.T, levels=levels, colors="k", linewidths=0.3)
    ax.set_xlim(-vlim, vlim)
    ax.set_ylim(-vlim, vlim)
    ax.set_aspect("equal")
    ax.set_xlabel(rf"$v_{a}/v_{{\rm th,{sub}}}$")
    ax.set_ylabel(rf"$v_{b}/v_{{\rm th,{sub}}}$")
    return cs


def plot_moments(dumps, species, sub):
    """Particle number, bulk velocity and temperature in every box vs time."""
    ds = [load_dump(p, fields=("moments", "vth")) for _, p in dumps]
    t = np.array([d["time"] for d in ds]) * OMEGA_CI
    fig, axes = plt.subplots(3, 1, figsize=(6, 7), sharex=True, layout="constrained")
    for b in range(ds[0]["moments"].shape[1]):
        mom = [box_moments(d, species, b) for d in ds]
        N = np.array([m["N"] for m in mom])
        U = np.array([m["U"] for m in mom])
        T = np.array([np.diag(m["T"]) for m in mom])
        axes[0].plot(t, N / N[0] if N[0] > 0 else N, f"C{b}", label=f"box {b}")
        for k, ls in enumerate(["-", "--", ":"]):
            axes[1].plot(t, U[:, k], f"C{b}", ls=ls)
            axes[2].plot(t, T[:, k], f"C{b}", ls=ls)
    axes[0].set_ylabel(r"$N / N(t_0)$")
    axes[1].set_ylabel(rf"$U/v_{{\rm th,{sub}}}$   (x –, y --, z ··)")
    axes[2].set_ylabel(r"$T/T_0$   (xx –, yy --, zz ··)")
    axes[2].set_xlabel(r"$t\,\Omega_{ci}$")
    axes[0].legend(fontsize=8)
    return fig


# --------------------------------------------------------------------- CLI ---
def main():
    p = argparse.ArgumentParser(description="Plot VDFs written by the KAW2D pgen.")
    p.add_argument("sim", help="simulation output directory (= simulation.name)")
    p.add_argument("--step", type=int, help="step of the dump to plot (default: last)")
    p.add_argument("--species", type=int, default=1, help="1 = e-, 2 = i+ (default 1)")
    p.add_argument("--box", type=int, default=0, help="box index (default 0)")
    p.add_argument("--plane", choices=list(PLANES), default="xy")
    p.add_argument("--all", action="store_true", help="grid of all boxes x all planes")
    p.add_argument("--diff", action="store_true", help="plot f - f(first dump)")
    p.add_argument("--log", action="store_true", help="plot log10 f")
    p.add_argument("--vlim", type=float, default=2.5, help="axis limit in v_th")
    p.add_argument("--levels", type=int, default=20, help="number of contour levels")
    p.add_argument("--smooth", type=int, default=0, help="1-2-1 smoothing passes")
    p.add_argument("--slice", type=float, metavar="WIDTH",
                   help="plot a slice of thickness WIDTH (in v_th) in the third "
                        "velocity component instead of the reduced VDF (3D dumps only)")
    p.add_argument("--slice-centre", type=float, default=0.0,
                   help="centre of the slice in the third component, in v_th (default 0)")
    p.add_argument("--moments", action="store_true", help="box moments vs time")
    p.add_argument("--save", help="write the figure to this file instead of showing it")
    args = p.parse_args()

    dumps = list_dumps(args.sim)
    if not dumps:
        raise SystemExit(f"no VDF dumps found in {os.path.join(args.sim, 'vdf')}")
    sub = species_subscript(load_meta(args.sim), args.species)

    if args.moments:
        fig = plot_moments(dumps, args.species, sub)
    else:
        steps = [s for s, _ in dumps]
        if args.step is None:
            path = dumps[-1][1]
        elif args.step in steps:
            path = dumps[steps.index(args.step)][1]
        else:
            raise SystemExit(f"no dump at step {args.step}; available: {steps}")
        d = load_dump(path)
        if args.slice is not None and not has_3d(d):
            raise SystemExit("--slice needs a dump with the 3D histogram (hist3d); "
                             "this one only has the reduced planes")
        kw = dict(vlim=args.vlim, nlevels=args.levels, log=args.log,
                  nsmooth=args.smooth, sub=sub,
                  ref=load_dump(dumps[0][1]) if args.diff else None,
                  slice_width=args.slice, slice_centre=args.slice_centre)
        if args.all:
            nbox = len(d["boxes"])
            fig, axes = plt.subplots(nbox, 3, figsize=(9, 3.1 * nbox), squeeze=False,
                                     layout="constrained")
            for b in range(nbox):
                for j, plane in enumerate(PLANES):
                    plot_vdf(axes[b, j], d, args.species, b, plane, **kw)
                axes[b, 1].set_title(box_title(d, b), fontsize=9)
        else:
            fig, ax = plt.subplots(figsize=(3.8, 3.4), layout="constrained")
            cs = plot_vdf(ax, d, args.species, args.box, args.plane, **kw)
            if cs is not None:
                fig.colorbar(cs, ax=ax, shrink=0.85,
                             label=r"$\delta f$" if args.diff else
                             (r"$\log_{10} f$" if args.log else r"$f$"))
            ax.set_title(box_title(d, args.box), fontsize=8)
        what = "f - f(t0)" if args.diff else "f"
        if args.slice is None:
            what += " reduced"
        else:   # third velocity component (for --all: of each plane); the range
            # actually used is snapped to the bin edges
            third = ("v_3" if args.all else
                     "v_" + "xyz"[_third_axis(args.plane)])
            _, _, (lo, hi) = slice_vdf(d, args.species, args.box, args.plane,
                                       args.slice, args.slice_centre)
            what += f" slice, {third} in [{lo:.3g}, {hi:.3g}] v_th"
        fig.suptitle(f"{what}, species {args.species}\n"
                     rf"$t\,\Omega_{{ci}}$ = {d['time'] * OMEGA_CI:.2f} (step {d['step']})",
                     fontsize=9)

    if args.save:
        fig.savefig(args.save, dpi=200)
        print(f"saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
