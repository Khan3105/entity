/**
 * @file vdf_output.hpp
 * @brief On-the-fly velocity distribution functions (VDFs) in small spatial boxes
 *
 * How it works
 * ------------
 * Every `vdf_interval_time` (or `vdf_interval` steps) each MPI rank loops over
 * its own particles on the device, keeps the ones that sit inside one of the
 * user-defined boxes, and histograms their velocities on a full 3D grid
 * (ux, uy, uz). Nothing is integrated out: reduced 2D/1D distributions, slices
 * and other cuts are all made afterwards from the cube (see plot_vdf.py). The
 * partial histograms of all ranks are summed onto rank 0 (a box may straddle
 * several subdomains), and rank 0 writes one self-describing .npy file per
 * dump. No particle data is ever written to disk.
 *
 * Output
 * ------
 *   <simulation.name>/vdf/vdf.<step>.npy   (same step numbering as the fields)
 *   <simulation.name>/vdf/vdf_meta.json    (human-readable description)
 *
 * Each .npy holds one structured record (load with np.load(fname)[0]):
 *   step    int64                            simulation step of the dump
 *   time    float64                          simulation time of the dump
 *   species int64   [nsp]                    1-based species indices (as in the toml)
 *   vth     float64 [nsp]                    normalisation speed of each species, units of c
 *   vmax    float64 [nsp]                    half-width of the velocity grid, units of vth
 *   boxes   float64 [nbox, 4]                [xmin, xmax, ymin, ymax], units of grid.extent
 *   moments float64 [nsp, nbox, 10]          exact (unbinned) sums, see below
 *   hist3d  float32 [nsp, nbox, nb, nb, nb]  number of particles per velocity bin
 *
 * hist3d[s, b, i, j, k]: i bins ux, j bins uy and k bins uz; bins are uniform
 * in u/vth with edges = linspace(-vmax, vmax, nb + 1) in every direction. A
 * particle is binned only if all three components lie on the grid (it is
 * always counted in the moments). With particles.use_weights = true, counts
 * are sums of weights. The histogram is summed in float64 and stored as
 * float32, which is exact for counts below 2^24 per bin.
 *
 * Size: nsp * nbox * nb^3 * 4 bytes per dump, e.g. 1.05 MB per species and box
 * for nb = 64, 8.4 MB for nb = 128. With ~1e6 particles per box, nb = 64
 * (dv = 0.125 vth for vmax = 4) already holds ~100 particles per bin in the
 * core; much finer grids mostly store counting noise.
 *
 * Dumps of the older format (field `hist`, [nsp, nbox, 3, nb, nb], the three
 * reduced planes) are still read by plot_vdf.py.
 *
 * moments[s, b, :] = sum over the particles in the box of
 *   w * { 1, ux, uy, uz, ux^2, uy^2, uz^2, ux*uy, ux*uz, uy*uz }
 * with u in units of c (not normalised by vth).
 *
 * u = gamma * v is the spatial part of the 4-velocity in units of c, which is
 * what Entity stores. For vth << c (here vth_e = 0.05 c) u and v are the same.
 *
 * Parameters, all in [setup]
 * --------------------------
 *   vdf_enable        = true                     (default: false)
 *   vdf_boxes         = [[xc, yc, wx, wy], ...]  centre and full width of each box
 *   vdf_species       = [1, 2]                   species to record (default: all)
 *   vdf_nbins         = 64                       bins per velocity axis (grid has nbins^3 bins)
 *   vdf_vmax          = [4.0, 4.0]               grid half-width in vth (one value, or one per recorded species)
 *   vdf_interval_time = 700.0                    output interval in time units
 *   vdf_interval      = 0                        output interval in steps (used if no interval_time)
 * Without vdf_interval_time / vdf_interval the field output interval is used.
 * Write all box coordinates as floats (with a decimal point).
 */

#ifndef KAW2D_VDF_OUTPUT_HPP
#define KAW2D_VDF_OUTPUT_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/mpi_aliases.h"
#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/log.h"
#include "utils/numeric.h"
#include "utils/tools.h"

#include "framework/containers/species.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"
#include "framework/parameters/parameters.h"

#include <Kokkos_Core.hpp>

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif

#include <algorithm>
#include <any>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

namespace user {
  using namespace ntt;

  namespace vdf {

    inline constexpr std::size_t kMaxBoxes = 32;  // max number of boxes
    inline constexpr std::size_t kNMoments = 10;  // see the file header
    // upper limit on the total number of histogram bins (nsp * nbox * nb^3):
    // 2^28 doubles = 2 GiB on every GPU and every host
    inline constexpr std::size_t kMaxBins  = std::size_t { 1 } << 28;
    // Every particle in a box adds to the same 10 moment sums. To avoid
    // thousands of GPU threads doing atomics on the same address, particle p
    // adds to copy (p % kNLanes); the copies are summed on the host.
    inline constexpr std::size_t kNLanes   = 256;

    // Box limits in the *local* cell coordinates of one subdomain. Small and
    // trivially copyable, so it is passed to the kernel by value.
    struct BoxesCode {
      int    n { 0 };
      real_t x1min[kMaxBoxes] {};
      real_t x1max[kMaxBoxes] {};
      real_t x2min[kMaxBoxes] {};
      real_t x2max[kMaxBoxes] {};
    };

    // Index of the velocity bin containing u, or -1 if u is outside the grid
    Inline auto BinIndex(double u, double u_lo, double inv_du, int nb) -> int {
      const double xi = (u - u_lo) * inv_du;
      if (not(xi >= 0.0 and xi < static_cast<double>(nb))) { // also rejects NaN
        return -1;
      }
      const auto i = static_cast<int>(xi);
      return (i < nb) ? i : nb - 1;
    }

    /* reading [setup] values: toml integers arrive as `int`, floats as `real_t` */
    inline auto ReadNumber(const SimulationParams& params,
                           const std::string&      key,
                           double                  def) -> double {
      if (not params.contains(key)) {
        return def;
      }
      const auto& val = params.allVars().at(key);
      if (val.type() == typeid(real_t)) {
        return static_cast<double>(std::any_cast<real_t>(val));
      } else if (val.type() == typeid(int)) {
        return static_cast<double>(std::any_cast<int>(val));
      }
      raise::Error(fmt::format("VDF: [setup] %s must be a number", key.c_str()),
                   HERE);
      return def;
    }

    // A list of numbers; a single number is returned as a list of length 1
    inline auto ReadNumberList(const SimulationParams& params,
                               const std::string& key) -> std::vector<double> {
      std::vector<double> out;
      if (not params.contains(key)) {
        return out;
      }
      const auto& val = params.allVars().at(key);
      if (val.type() == typeid(std::vector<real_t>)) {
        for (const auto& v : std::any_cast<std::vector<real_t>>(val)) {
          out.push_back(static_cast<double>(v));
        }
      } else if (val.type() == typeid(std::vector<int>)) {
        for (const auto& v : std::any_cast<std::vector<int>>(val)) {
          out.push_back(static_cast<double>(v));
        }
      } else {
        out.push_back(ReadNumber(params, key, 0.0));
      }
      return out;
    }

    // A list of boxes, each box being a list of numbers
    inline auto ReadBoxes(const SimulationParams& params, const std::string& key)
      -> std::vector<std::vector<double>> {
      std::vector<std::vector<double>> out;
      if (not params.contains(key)) {
        return out;
      }
      const auto& val = params.allVars().at(key);
      if (val.type() == typeid(std::vector<std::vector<real_t>>)) {
        for (const auto& box :
             std::any_cast<std::vector<std::vector<real_t>>>(val)) {
          out.emplace_back(box.begin(), box.end());
        }
      } else if (val.type() == typeid(std::vector<std::vector<int>>)) {
        for (const auto& box : std::any_cast<std::vector<std::vector<int>>>(val)) {
          out.emplace_back(box.begin(), box.end());
        }
      } else {
        // a single box written as a flat list: vdf_boxes = [0.0, 56.0, 4.0, 4.0]
        out.push_back(ReadNumberList(params, key));
      }
      return out;
    }

    /* small formatting helpers for the .npy header and the json file */
    inline auto NpyShape(const std::vector<std::size_t>& shape) -> std::string {
      std::string s = "(";
      for (std::size_t i = 0; i < shape.size(); ++i) {
        s += std::to_string(shape[i]);
        if (shape.size() == 1) {
          s += ","; // python 1-tuple: (n,)
        } else if (i + 1 < shape.size()) {
          s += ", ";
        }
      }
      return s + ")";
    }

    inline auto JsonString(const std::string& str) -> std::string {
      std::string out = "\"";
      for (const char c : str) {
        if (c == '"' or c == '\\') {
          out += '\\';
        }
        out += c;
      }
      return out + "\"";
    }

    inline auto JsonNumber(double x) -> std::string {
      return fmt::format("%.10g", x);
    }

  } // namespace vdf

  template <SimEngine::type S, class M>
  class VDFOutput {
    static constexpr auto D { M::Dim };

    bool                             m_enabled { false };
    std::vector<std::vector<double>> m_boxes;   // [xmin, xmax, ymin, ymax]
    std::vector<int>                 m_species; // 1-based species indices
    std::vector<std::string>         m_labels;
    std::vector<double>              m_mass, m_charge;
    std::vector<double>              m_vth;     // units of c
    std::vector<double>              m_vmax;    // units of vth
    int                              m_nbins { 0 };
    bool                             m_use_weights { false };
    double                           m_dt { 0.0 };
    timestep_t                       m_interval { 0 };
    simtime_t                        m_interval_time { -1.0 };
    std::filesystem::path            m_dir;
    tools::Tracker                   m_tracker;

    // device buffers (flattened)
    array_t<double*>        m_hist;   // [nsp, nbox, i(ux), j(uy), k(uz)]
    array_t<double*>        m_mom;    // [lane, nsp, nbox, moment]
    array_mirror_t<double*> m_hist_h; // host copy of m_hist, allocated once

  public:
    VDFOutput() = default;

    [[nodiscard]]
    auto enabled() const -> bool {
      return m_enabled;
    }

    /**
     * @brief Read the [setup] parameters, allocate buffers, write vdf_meta.json
     * @param vth_per_species normalisation speed of every species (units of c),
     * in the order of [[particles.species]]
     */
    void Init(const SimulationParams&    params,
              const Metadomain<S, M>&    global_domain,
              const std::vector<double>& vth_per_species) {
      m_enabled = params.contains("setup.vdf_enable") and
                  params.template get<bool>("setup.vdf_enable");
      if (not m_enabled) {
        return;
      }
      raise::ErrorIf(D != Dim::_2D, "VDF output is only implemented in 2D", HERE);

      /* species ------------------------------------------------------------ */
      const auto species = params.template get<std::vector<ParticleSpecies>>(
        "particles.species");
      raise::ErrorIf(vth_per_species.size() < species.size(),
                     "VDF: a thermal speed is needed for every species",
                     HERE);
      const auto sp_list = vdf::ReadNumberList(params, "setup.vdf_species");
      if (sp_list.empty()) {
        for (std::size_t s = 1; s <= species.size(); ++s) {
          m_species.push_back(static_cast<int>(s));
        }
      } else {
        for (const auto v : sp_list) {
          const auto s = static_cast<int>(v);
          raise::ErrorIf(s < 1 or s > static_cast<int>(species.size()),
                         fmt::format("VDF: invalid species index %d", s),
                         HERE);
          m_species.push_back(s);
        }
      }
      const auto nsp = m_species.size();

      /* velocity grid ------------------------------------------------------ */
      m_nbins = static_cast<int>(
        vdf::ReadNumber(params, "setup.vdf_nbins", 64.0));
      raise::ErrorIf(m_nbins < 2 or m_nbins > 512,
                     "VDF: vdf_nbins must be between 2 and 512",
                     HERE);
      const auto vmax_list = vdf::ReadNumberList(params, "setup.vdf_vmax");
      raise::ErrorIf(vmax_list.size() > 1 and vmax_list.size() != nsp,
                     "VDF: vdf_vmax needs one value, or one per recorded species",
                     HERE);
      for (std::size_t s = 0; s < nsp; ++s) {
        const auto& sp   = species[static_cast<std::size_t>(m_species[s] - 1)];
        const auto  vmax = vmax_list.empty()
                             ? 4.0
                             : vmax_list[(vmax_list.size() == 1) ? 0 : s];
        const auto  vth = vth_per_species[static_cast<std::size_t>(m_species[s] - 1)];
        raise::ErrorIf(vmax <= 0.0, "VDF: vdf_vmax must be positive", HERE);
        raise::ErrorIf(vth <= 0.0, "VDF: thermal speeds must be positive", HERE);
        m_labels.push_back(sp.label());
        m_mass.push_back(static_cast<double>(sp.mass()));
        m_charge.push_back(static_cast<double>(sp.charge()));
        m_vth.push_back(vth);
        m_vmax.push_back(vmax);
      }

      /* boxes (physical coordinates, clipped to the domain) ---------------- */
      const auto ext1  = global_domain.mesh().extent(in::x1);
      const auto ext2  = global_domain.mesh().extent(in::x2);
      const auto cell1 = static_cast<double>(ext1.second - ext1.first) /
                         static_cast<double>(global_domain.mesh().n_active(in::x1));
      const auto cell2 = static_cast<double>(ext2.second - ext2.first) /
                         static_cast<double>(global_domain.mesh().n_active(in::x2));
      const auto boxes_in = vdf::ReadBoxes(params, "setup.vdf_boxes");
      raise::ErrorIf(boxes_in.empty(),
                     "VDF: vdf_enable = true but no vdf_boxes are given",
                     HERE);
      raise::ErrorIf(boxes_in.size() > vdf::kMaxBoxes,
                     fmt::format("VDF: at most %zu boxes are supported",
                                 vdf::kMaxBoxes),
                     HERE);
      for (const auto& box : boxes_in) {
        raise::ErrorIf(box.size() != 4,
                       "VDF: each box must be [x_centre, y_centre, width_x, width_y]",
                       HERE);
        raise::ErrorIf(box[2] <= 0.0 or box[3] <= 0.0,
                       "VDF: box widths must be positive",
                       HERE);
        const auto xmin = std::max(box[0] - 0.5 * box[2],
                                   static_cast<double>(ext1.first));
        const auto xmax = std::min(box[0] + 0.5 * box[2],
                                   static_cast<double>(ext1.second));
        const auto ymin = std::max(box[1] - 0.5 * box[3],
                                   static_cast<double>(ext2.first));
        const auto ymax = std::min(box[1] + 0.5 * box[3],
                                   static_cast<double>(ext2.second));
        raise::ErrorIf(xmin >= xmax or ymin >= ymax,
                       "VDF: a box lies outside of the domain",
                       HERE);
        if (xmax - xmin < box[2] or ymax - ymin < box[3]) {
          raise::Warning("VDF: a box was clipped to the domain boundaries", HERE);
        }
        m_boxes.push_back({ xmin, xmax, ymin, ymax });
      }
      const auto nbox = m_boxes.size();

      /* output cadence ----------------------------------------------------- */
      const auto int_time  = vdf::ReadNumber(params, "setup.vdf_interval_time", -1.0);
      const auto int_steps = vdf::ReadNumber(params, "setup.vdf_interval", 0.0);
      if (int_time > 0.0 or int_steps >= 1.0) {
        m_interval      = static_cast<timestep_t>(std::max(int_steps, 0.0));
        m_interval_time = static_cast<simtime_t>(int_time);
      } else {
        m_interval = params.template get<timestep_t>("output.fields.interval");
        m_interval_time = params.template get<simtime_t>(
          "output.fields.interval_time");
      }
      raise::ErrorIf(m_interval_time <= 0.0 and m_interval == 0,
                     "VDF: the output interval must be positive",
                     HERE);
      // no walltime trigger: every rank must take the same decision
      m_tracker.init("vdf", m_interval, m_interval_time);

      m_dt = static_cast<double>(
        params.template get<real_t>("algorithms.timestep.dt"));
      m_use_weights = params.template get<bool>("particles.use_weights");

      /* buffers ------------------------------------------------------------ */
      const auto nb    = static_cast<std::size_t>(m_nbins);
      const auto nhist = nsp * nbox * nb * nb * nb;
      raise::ErrorIf(
        nhist > vdf::kMaxBins,
        fmt::format("VDF: %zu x %zu x %d^3 = %zu bins is more than the limit of "
                    "%zu (reduce vdf_nbins or the number of boxes)",
                    nsp,
                    nbox,
                    m_nbins,
                    nhist,
                    vdf::kMaxBins),
        HERE);
      m_hist   = array_t<double*> { "vdf_hist", nhist };
      m_hist_h = Kokkos::create_mirror_view(m_hist);
      m_mom    = array_t<double*> { "vdf_mom",
                                    vdf::kNLanes * nsp * nbox * vdf::kNMoments };

      /* output directory, metadata and a short report (rank 0 only) -------- */
      m_dir = std::filesystem::path(
                params.template get<std::string>("simulation.name")) /
              "vdf";
      CallOnce([&]() {
        try {
          std::filesystem::create_directories(m_dir);
          WriteMeta(params.template get<std::string>("simulation.name"),
                    cell1,
                    cell2);
        } catch (const std::exception& e) {
          raise::Error(fmt::format("VDF: cannot create %s : %s",
                                   m_dir.string().c_str(),
                                   e.what()),
                       HERE);
        }
      });

      std::string report = "\nVDF output -> " + m_dir.string() + "/vdf.<step>.npy\n";
      if (m_interval_time > 0.0) {
        report += fmt::format("  interval        : every %.6g time units\n",
                              m_interval_time);
      } else {
        report += fmt::format("  interval        : every %u steps\n",
                              static_cast<unsigned int>(m_interval));
      }
      report += fmt::format(
        "  velocity grid   : %d x %d x %d bins in (ux, uy, uz), %.3g MB per "
        "dump\n",
        m_nbins,
        m_nbins,
        m_nbins,
        static_cast<double>(nhist * sizeof(float)) / 1.0e6);
      for (std::size_t s = 0; s < nsp; ++s) {
        report += fmt::format(
          "  species %d (%s)  : vth = %.6g c, grid = +/- %.3g vth\n",
          m_species[s],
          m_labels[s].c_str(),
          m_vth[s],
          m_vmax[s]);
      }
      for (std::size_t b = 0; b < nbox; ++b) {
        report += fmt::format(
          "  box %zu           : x = [%.6g, %.6g], y = [%.6g, %.6g]  (%.1f x %.1f "
          "cells)\n",
          b,
          m_boxes[b][0],
          m_boxes[b][1],
          m_boxes[b][2],
          m_boxes[b][3],
          (m_boxes[b][1] - m_boxes[b][0]) / cell1,
          (m_boxes[b][3] - m_boxes[b][2]) / cell2);
      }
      info::Print(report, false);
    }

    /**
     * @brief Called after every step from CustomPostStep (on every rank)
     * @note All ranks take the same output decision, so the MPI reduction
     * inside is always matched.
     */
    void Process(timestep_t step, simtime_t time, Domain<S, M>& domain) {
      if (not m_enabled or not m_tracker.shouldWrite(step, time)) {
        return;
      }
      // CustomPostStep runs at the end of step `step`: particles are now at
      // step + 1, which is also the number used for the field output files.
      const auto out_step = static_cast<timestep_t>(step + 1);
      const auto out_time = static_cast<simtime_t>(time + m_dt);

      Compute(domain);

      // this rank's partial histograms -> host (the large histogram buffer is
      // allocated once in Init; the moments are small)
      auto mom_h = Kokkos::create_mirror_view(m_mom);
      Kokkos::deep_copy(m_hist_h, m_hist);
      Kokkos::deep_copy(mom_h, m_mom);

      // add up the lanes of the moment sums
      const auto          nmom = m_species.size() * m_boxes.size() * vdf::kNMoments;
      std::vector<double> mom(nmom, 0.0);
      for (std::size_t lane = 0; lane < vdf::kNLanes; ++lane) {
        for (std::size_t i = 0; i < nmom; ++i) {
          mom[i] += mom_h(lane * nmom + i);
        }
      }
      double*    hist  = m_hist_h.data();
      const auto nhist = static_cast<std::size_t>(m_hist_h.extent(0));

#if defined(MPI_ENABLED)
      // sum the contributions of all ranks onto the root rank
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank == MPI_ROOT_RANK) {
        MPI_Reduce(MPI_IN_PLACE, hist, static_cast<int>(nhist), MPI_DOUBLE,
                   MPI_SUM, MPI_ROOT_RANK, MPI_COMM_WORLD);
        MPI_Reduce(MPI_IN_PLACE, mom.data(), static_cast<int>(nmom), MPI_DOUBLE,
                   MPI_SUM, MPI_ROOT_RANK, MPI_COMM_WORLD);
      } else {
        MPI_Reduce(hist, nullptr, static_cast<int>(nhist), MPI_DOUBLE, MPI_SUM,
                   MPI_ROOT_RANK, MPI_COMM_WORLD);
        MPI_Reduce(mom.data(), nullptr, static_cast<int>(nmom), MPI_DOUBLE,
                   MPI_SUM, MPI_ROOT_RANK, MPI_COMM_WORLD);
        return;
      }
#endif
      WriteNpy(out_step, out_time, mom, hist, nhist);
    }

    /**
     * @brief Fill m_hist and m_mom with the contribution of this rank
     * @note Public on purpose: CUDA does not allow device lambdas inside
     * private member functions.
     */
    void Compute(Domain<S, M>& domain) {
      Kokkos::deep_copy(m_hist, 0.0);
      Kokkos::deep_copy(m_mom, 0.0);

      // box limits in local cell coordinates: x_cell = (x - xmin_local) / dx,
      // which is directly comparable with i1 + dx1 of the particles
      const auto ext1  = domain.mesh.extent(in::x1);
      const auto ext2  = domain.mesh.extent(in::x2);
      const auto n1    = static_cast<double>(domain.mesh.n_active(in::x1));
      const auto n2    = static_cast<double>(domain.mesh.n_active(in::x2));
      const auto cell1 = static_cast<double>(ext1.second - ext1.first) / n1;
      const auto cell2 = static_cast<double>(ext2.second - ext2.first) / n2;

      vdf::BoxesCode boxes;
      boxes.n       = static_cast<int>(m_boxes.size());
      bool overlaps = false;
      for (std::size_t b = 0; b < m_boxes.size(); ++b) {
        const auto x1min = (m_boxes[b][0] - static_cast<double>(ext1.first)) / cell1;
        const auto x1max = (m_boxes[b][1] - static_cast<double>(ext1.first)) / cell1;
        const auto x2min = (m_boxes[b][2] - static_cast<double>(ext2.first)) / cell2;
        const auto x2max = (m_boxes[b][3] - static_cast<double>(ext2.first)) / cell2;
        boxes.x1min[b]   = static_cast<real_t>(x1min);
        boxes.x1max[b]   = static_cast<real_t>(x1max);
        boxes.x2min[b]   = static_cast<real_t>(x2min);
        boxes.x2max[b]   = static_cast<real_t>(x2max);
        overlaps = overlaps or
                   ((x1max > 0.0) and (x1min < n1) and (x2max > 0.0) and (x2min < n2));
      }
      if (not overlaps) {
        return; // no box touches this subdomain: this rank contributes zeros
      }

      const auto        nsp         = m_species.size();
      const auto        nbox        = m_boxes.size();
      const int         nb          = m_nbins;
      const auto        nbs         = static_cast<std::size_t>(nb);
      const auto        nb3         = nbs * nbs * nbs;
      const auto        nmom        = nsp * nbox * vdf::kNMoments;
      const std::size_t nlanes      = vdf::kNLanes;
      const std::size_t nmoments    = vdf::kNMoments;
      const bool        use_weights = m_use_weights;
      auto              hist        = m_hist;
      auto              mom         = m_mom;

      for (std::size_t s = 0; s < nsp; ++s) {
        auto& prtls = domain.species[static_cast<std::size_t>(m_species[s] - 1)];
        if (prtls.npart() == 0) {
          continue;
        }
        // u -> bin: i = floor((u - u_lo) / du), uniform on [-vmax, vmax] * vth
        const double u_lo   = -m_vmax[s] * m_vth[s];
        const double inv_du = static_cast<double>(nb) / (2.0 * m_vmax[s] * m_vth[s]);

        // local copies: the device lambda must not capture `this`
        const auto i1     = prtls.i1;
        const auto i2     = prtls.i2;
        const auto dx1    = prtls.dx1;
        const auto dx2    = prtls.dx2;
        const auto ux1    = prtls.ux1;
        const auto ux2    = prtls.ux2;
        const auto ux3    = prtls.ux3;
        const auto weight = prtls.weight;
        const auto tag    = prtls.tag;

        Kokkos::parallel_for(
          "VDF_Compute",
          prtls.rangeActiveParticles(),
          Lambda(prtlidx_t p) {
            if (tag(p) != ParticleTag::alive) {
              return;
            }
            const auto x1 = static_cast<real_t>(i1(p)) + static_cast<real_t>(dx1(p));
            const auto x2 = static_cast<real_t>(i2(p)) + static_cast<real_t>(dx2(p));
            for (int b = 0; b < boxes.n; ++b) {
              if (x1 < boxes.x1min[b] or x1 >= boxes.x1max[b] or
                  x2 < boxes.x2min[b] or x2 >= boxes.x2max[b]) {
                continue;
              }
              const double w  = use_weights ? static_cast<double>(weight(p)) : 1.0;
              const double ux = static_cast<double>(ux1(p));
              const double uy = static_cast<double>(ux2(p));
              const double uz = static_cast<double>(ux3(p));
              const auto   sb = s * nbox + static_cast<std::size_t>(b);

              // exact moments
              const auto m0 = (static_cast<std::size_t>(p) % nlanes) * nmom +
                              sb * nmoments;
              Kokkos::atomic_add(&mom(m0 + 0), w);
              Kokkos::atomic_add(&mom(m0 + 1), w * ux);
              Kokkos::atomic_add(&mom(m0 + 2), w * uy);
              Kokkos::atomic_add(&mom(m0 + 3), w * uz);
              Kokkos::atomic_add(&mom(m0 + 4), w * ux * ux);
              Kokkos::atomic_add(&mom(m0 + 5), w * uy * uy);
              Kokkos::atomic_add(&mom(m0 + 6), w * uz * uz);
              Kokkos::atomic_add(&mom(m0 + 7), w * ux * uy);
              Kokkos::atomic_add(&mom(m0 + 8), w * ux * uz);
              Kokkos::atomic_add(&mom(m0 + 9), w * uy * uz);

              // full 3D histogram: bin (ix, iy, iz), iz running fastest
              const int ix = vdf::BinIndex(ux, u_lo, inv_du, nb);
              const int iy = vdf::BinIndex(uy, u_lo, inv_du, nb);
              const int iz = vdf::BinIndex(uz, u_lo, inv_du, nb);
              if (ix >= 0 and iy >= 0 and iz >= 0) {
                const auto ib = (static_cast<std::size_t>(ix) * nbs +
                                 static_cast<std::size_t>(iy)) *
                                  nbs +
                                static_cast<std::size_t>(iz);
                Kokkos::atomic_add(&hist(sb * nb3 + ib), w);
              }
            }
          });
      }
      Kokkos::fence();
    }

  private:
    /**
     * @brief Write one dump as a .npy file holding a single structured record
     * @note Written to <name>.tmp first and renamed, so a crash never leaves a
     * truncated vdf.<step>.npy behind. A failed write only prints a warning.
     */
    void WriteNpy(timestep_t                 step,
                  simtime_t                  time,
                  const std::vector<double>& mom,
                  const double*              hist,
                  std::size_t                nhist) const {
      static_assert(std::endian::native == std::endian::little,
                    "the VDF .npy writer assumes a little-endian machine");
      const auto nsp  = m_species.size();
      const auto nbox = m_boxes.size();
      const auto nb   = static_cast<std::size_t>(m_nbins);

      std::string header =
        "{'descr': [('step', '<i8'), ('time', '<f8'), "
        "('species', '<i8', " +
        vdf::NpyShape({ nsp }) + "), ('vth', '<f8', " + vdf::NpyShape({ nsp }) +
        "), ('vmax', '<f8', " + vdf::NpyShape({ nsp }) + "), ('boxes', '<f8', " +
        vdf::NpyShape({ nbox, 4 }) + "), ('moments', '<f8', " +
        vdf::NpyShape({ nsp, nbox, vdf::kNMoments }) + "), ('hist3d', '<f4', " +
        vdf::NpyShape({ nsp, nbox, nb, nb, nb }) +
        ")], 'fortran_order': False, 'shape': (1,), }";
      // npy v1.0: magic (6) + version (2) + header length (2) + header,
      // padded with spaces and ending in '\n' so the total is a multiple of 64
      const std::size_t preamble = 10;
      header.append((64 - (preamble + header.size() + 1) % 64) % 64, ' ');
      header.push_back('\n');
      const auto hlen = static_cast<std::uint16_t>(header.size());

      try {
        const auto fname = m_dir / fmt::format("vdf.%08u.npy",
                                               static_cast<unsigned int>(step));
        auto tmpname = fname;
        tmpname += ".tmp";
        {
          std::ofstream out(tmpname, std::ios::binary | std::ios::trunc);
          if (not out) {
            throw std::runtime_error("cannot open " + tmpname.string());
          }
          const char magic[6] = { '\x93', 'N', 'U', 'M', 'P', 'Y' };
          const char version[2] = { '\x01', '\x00' };
          const char hlen_le[2] = { static_cast<char>(hlen & 0xFFu),
                                    static_cast<char>((hlen >> 8) & 0xFFu) };
          out.write(magic, 6);
          out.write(version, 2);
          out.write(hlen_le, 2);
          out.write(header.data(), static_cast<std::streamsize>(header.size()));

          const auto put_i64 = [&out](std::int64_t v) {
            out.write(reinterpret_cast<const char*>(&v), sizeof(v));
          };
          const auto put_f64 = [&out](double v) {
            out.write(reinterpret_cast<const char*>(&v), sizeof(v));
          };
          put_i64(static_cast<std::int64_t>(step));
          put_f64(static_cast<double>(time));
          for (const auto sp : m_species) {
            put_i64(static_cast<std::int64_t>(sp));
          }
          for (const auto v : m_vth) {
            put_f64(v);
          }
          for (const auto v : m_vmax) {
            put_f64(v);
          }
          for (const auto& box : m_boxes) {
            for (const auto v : box) {
              put_f64(v);
            }
          }
          out.write(reinterpret_cast<const char*>(mom.data()),
                    static_cast<std::streamsize>(mom.size() * sizeof(double)));
          // histogram: float64 -> float32, a chunk at a time
          constexpr std::size_t kChunk = std::size_t { 1 } << 20;
          std::vector<float>    buf(std::min(kChunk, nhist));
          for (std::size_t i0 = 0; i0 < nhist; i0 += kChunk) {
            const auto n = std::min(kChunk, nhist - i0);
            for (std::size_t i = 0; i < n; ++i) {
              buf[i] = static_cast<float>(hist[i0 + i]);
            }
            out.write(reinterpret_cast<const char*>(buf.data()),
                      static_cast<std::streamsize>(n * sizeof(float)));
          }
          if (not out) {
            throw std::runtime_error("write to " + tmpname.string() + " failed");
          }
        }
        std::filesystem::rename(tmpname, fname);
        info::Print(fmt::format("VDF: wrote %s (t = %.6g)", fname.string().c_str(), time),
                    false);
      } catch (const std::exception& e) {
        raise::Warning(fmt::format("VDF: could not write dump: %s", e.what()),
                       HERE);
      }
    }

    void WriteMeta(const std::string& simname, double cell1, double cell2) const {
      std::string js = "{\n";
      js += "  \"description\": \"On-the-fly 3D velocity distributions in spatial "
            "boxes, see pgens/KAW2D/vdf_output.hpp\",\n";
      js += "  \"format_version\": 2,\n";
      js += "  \"simulation\": " + vdf::JsonString(simname) + ",\n";
      js += "  \"files\": \"vdf.<step>.npy, each holding one structured record "
            "with fields step, time, species, vth, vmax, boxes, moments, "
            "hist3d\",\n";
      js += "  \"dt\": " + vdf::JsonNumber(m_dt) + ",\n";
      js += "  \"interval_steps\": " + std::to_string(m_interval) + ",\n";
      js += "  \"interval_time\": " + vdf::JsonNumber(m_interval_time) + ",\n";
      js += "  \"nbins\": " + std::to_string(m_nbins) + ",\n";
      js += "  \"hist3d_layout\": \"hist3d[species, box, i, j, k] (float32): i bins "
            "ux, j bins uy, k bins uz; bin edges are linspace(-vmax, vmax, "
            "nbins + 1) in units of vth; a particle is binned only if all three "
            "components are on the grid\",\n";
      js += "  \"velocity\": \"u = gamma * v (spatial part of the 4-velocity) in "
            "units of c\",\n";
      js += "  \"moments\": [\"w\", \"w*ux\", \"w*uy\", \"w*uz\", \"w*ux*ux\", "
            "\"w*uy*uy\", \"w*uz*uz\", \"w*ux*uy\", \"w*ux*uz\", \"w*uy*uz\"],\n";
      js += "  \"use_weights\": " + std::string(m_use_weights ? "true" : "false") +
            ",\n";
      js += "  \"cell_size\": [" + vdf::JsonNumber(cell1) + ", " +
            vdf::JsonNumber(cell2) + "],\n";
      js += "  \"species\": [\n";
      for (std::size_t s = 0; s < m_species.size(); ++s) {
        js += "    {\"index\": " + std::to_string(m_species[s]) +
              ", \"label\": " + vdf::JsonString(m_labels[s]) +
              ", \"mass\": " + vdf::JsonNumber(m_mass[s]) +
              ", \"charge\": " + vdf::JsonNumber(m_charge[s]) +
              ", \"vth\": " + vdf::JsonNumber(m_vth[s]) +
              ", \"vmax\": " + vdf::JsonNumber(m_vmax[s]) + "}" +
              ((s + 1 < m_species.size()) ? "," : "") + "\n";
      }
      js += "  ],\n";
      js += "  \"boxes\": [\n";
      for (std::size_t b = 0; b < m_boxes.size(); ++b) {
        js += "    {\"xmin\": " + vdf::JsonNumber(m_boxes[b][0]) +
              ", \"xmax\": " + vdf::JsonNumber(m_boxes[b][1]) +
              ", \"ymin\": " + vdf::JsonNumber(m_boxes[b][2]) +
              ", \"ymax\": " + vdf::JsonNumber(m_boxes[b][3]) + "}" +
              ((b + 1 < m_boxes.size()) ? "," : "") + "\n";
      }
      js += "  ]\n}\n";
      std::ofstream out(m_dir / "vdf_meta.json");
      out << js;
    }
  };

} // namespace user

#endif // KAW2D_VDF_OUTPUT_HPP
