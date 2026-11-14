#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/pgen.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "archetypes/energy_dist.h"
#include "archetypes/utils.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"

#include <random>

#if defined(MPI_ENABLED)
  #include <stdlib.h>
#endif // MPI_ENABLED

namespace user {
  using namespace ntt; 
  inline constexpr real_t PI = static_cast<real_t>(constant::PI);

  // setting up Equilibrium Bx and n for convenience
  template <Dimension D>
  struct Equilibrium {
    Equilibrium(real_t B_min, real_t B_max, real_t Ly, real_t h, real_t r,
                real_t pT0, real_t kTe0, real_t kTi0, real_t Cj, real_t rho0)
      : B_min { B_min }
      , B_max{ B_max }
      , Ly { Ly }
      , h { h }
      , r { r }
      , pT0 { pT0 }
      , kTe0 { kTe0 }
      , kTi0 { kTi0 }
      , alpha{ (B_max - B_min) * r / (2 * math::pow(2 * h, r)) /
              SQR(1 + 1 / math::pow(2 * h, r)) } 
      , Cj { Cj }
      , rho0 { rho0 } {}

    // Bx
    Inline auto Bx_eq(real_t y) const -> real_t {
        return B_min + (B_max - B_min) / (1 + math::pow(((y) / Ly / h), r)) +
        alpha * SQR((y / Ly * 2));
    }
    Inline auto dBxdy(real_t y) const -> real_t {
        return - r / (h * Ly) * math::pow((y)/(h * Ly), r - 1) * (B_max - B_min) / 
        SQR((1 + math::pow(((y) / Ly / h), r))) + 4 * alpha / Ly * (y / Ly * 2);
    }
    // number density n
    Inline auto ns_eq(real_t y) const -> real_t {
        return (pT0 - SQR(Bx_eq(y)) / 2) / (kTe0 + kTi0);
    }
    Inline auto dndy(real_t y) const -> real_t { 
        return -Bx_eq(y) * dBxdy(y) / (kTe0 + kTi0);
    }
    // Jz here for convenience
    Inline auto Jz_eq(real_t y) const -> real_t { return -dBxdy(y) / Cj; }
    real_t Cj, rho0;
  private:
    real_t B_min, B_max, Ly, h, r, pT0, kTe0, kTi0, alpha;
  };

  // initialise B and E fields
  template <Dimension D>
  struct InitFields {
    InitFields(real_t B_min,
             real_t B_max,
             real_t Ly,
             real_t h,
             real_t r,
             real_t pT0,
             real_t kTe0,
             real_t kTi0,
             real_t Ampl,
             real_t Lx,
             real_t kx,
             real_t me,
             real_t miome,
             real_t q_e,
             real_t q_i,
             real_t Cj,
             real_t rho0,
             bool multimode_in = false,
             int n_mode_pairs_in = 0,
             real_t brms_in = 1.0,
             Kokkos::View<int*> mode_i_in = {},
             Kokkos::View<int*> mode_j_in = {},
             Kokkos::View<real_t*> phi1_in = {},
             Kokkos::View<real_t*> phi2_in = {})
    : Ly { Ly }
    , h { h }
    , r { r }
    , kTe0 { kTe0 }
    , Ampl { Ampl }
    , Lx { Lx }
    , kx { kx }
    , me { me }
    , miome { miome }
    , q_e { q_e }
    , fe { miome / (miome - q_e / q_i) }
    , eq { B_min, B_max, Ly, h, r, pT0, kTe0, kTi0, Cj, rho0 } 
    , multimode { multimode_in }
    , n_mode_pairs { n_mode_pairs_in }
    , brms { brms_in }
    , mode_i { mode_i_in }
    , mode_j { mode_j_in }
    , phi1 { phi1_in }
    , phi2 { phi2_in }
    {}

  private:
    real_t Ly, Lx, kx, h, r, kTe0, Ampl, me, miome, q_e, fe;
    Equilibrium<D> eq;

    bool multimode;
    int n_mode_pairs;
    real_t brms;
    Kokkos::View<int*>    mode_i;
    Kokkos::View<int*>    mode_j;
    Kokkos::View<real_t*> phi1;
    Kokkos::View<real_t*> phi2;


    // B-field perturbation
    Inline auto bz_pert(real_t x, real_t y) const -> real_t { 
      if (!multimode) { 
        return Ampl * math::cos(kx / Lx * (x - Lx/2));
     }
     real_t sum = ZERO;
     for (int m = 0; m < n_mode_pairs; m++) {
        real_t kx = 2 * PI * mode_i(m) / Lx;
        real_t ky = 2 * PI * mode_j(m) / Ly;
        real_t arg = kx * x + ky * y;
        sum += math::cos(arg + phi1(m));
        sum += math::cos(arg + phi2(m));
      }
      return sum / brms * Ampl;
    }

    Inline auto dbzdx(real_t x, real_t y) const -> real_t { 
      if (!multimode) {
        return - kx / Lx * Ampl * math::sin(kx / Lx * (x - Lx / 2));
      }
      real_t sum = ZERO;
      for (int m = 0; m < n_mode_pairs; m++) {
        real_t kx = 2 * PI * mode_i(m) / Lx;
        real_t ky = 2 * PI * mode_j(m) / Ly;
        real_t arg = kx * x + ky * y;
        sum += -kx * math::sin(arg + phi1(m));
        sum += -kx * math::sin(arg + phi2(m));
      }
      return sum / brms * Ampl;
    }

    Inline auto dbzdy(real_t x, real_t y) const -> real_t { 
      if (!multimode) {
        return ZERO;
      }
      real_t sum = ZERO;
      for (int m = 0; m < n_mode_pairs; m++) {
        real_t kx = 2 * PI * mode_i(m) / Lx;
        real_t ky = 2 * PI * mode_j(m) / Ly;
        real_t arg = kx * x + ky * y;
        sum += -ky * math::sin(arg + phi1(m));
        sum += -ky * math::sin(arg + phi2(m));
      }
      return sum / brms * Ampl;
    }
    
    // Bulk velocity perturbation
    Inline auto vz_pert(real_t x, real_t y) const -> real_t {
      if (!multimode) {
        return -bz_pert(x, y) / math::sqrt(eq.ns_eq(y) * me * (1 + miome));
      }
      // For multimode: counter-propagating waves.
      // phi1 modes get -1/√ρ (forward), phi2 modes get +1/√ρ (backward).
      // This is the ± sign that seeds turbulence.
      real_t sum = ZERO;
      for (int m = 0; m < n_mode_pairs; m++) {
        real_t kx = 2 * PI * mode_i(m) / Lx;
        real_t ky = 2 * PI * mode_j(m) / Ly;
        real_t arg = kx * x + ky * y;
        sum += math::cos(arg + phi1(m));
        sum += math::cos(arg + phi2(m));
      }
      return -sum / brms * Ampl / math::sqrt(eq.ns_eq(y) * me * (1 + miome));
    }

    // E-field background and perturbation
    Inline auto Ex(real_t x, real_t y) const -> real_t { 
      return -fe * dbzdx(x, y) * bz_pert(x, y) / (eq.Cj * eq.ns_eq(y));
    }
    Inline auto Ey(real_t x, real_t y) const -> real_t { 
      return - vz_pert(x, y) * eq.Bx_eq(y) 
             - fe * eq.dBxdy(y) * eq.Bx_eq(y) / (eq.Cj * eq.ns_eq(y))
             - fe * dbzdy(x, y) * bz_pert(x, y) / (eq.Cj * eq.ns_eq(y))
             + eq.rho0 * kTe0 * eq.dndy(y) / (q_e * eq.ns_eq(y));
    }
    Inline auto Ez(real_t x, real_t y) const -> real_t { 
      return fe * dbzdx(x, y) * eq.Bx_eq(y) / (eq.Cj * eq.ns_eq(y));
    }

  public:
    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t { return eq.Bx_eq(x_Ph[1]); }
    Inline auto bx2(const coord_t<D>&) const -> real_t { return ZERO; }
    Inline auto bx3(const coord_t<D>& x_Ph) const -> real_t { return bz_pert(x_Ph[0], x_Ph[1]); }

    Inline auto ex1(const coord_t<D>& x_Ph) const -> real_t { return Ex(x_Ph[0], x_Ph[1]); }
    Inline auto ex2(const coord_t<D>& x_Ph) const -> real_t { return Ey(x_Ph[0], x_Ph[1]); }
    Inline auto ex3(const coord_t<D>& x_Ph) const -> real_t { return Ez(x_Ph[0], x_Ph[1]); }

  };

  // number density profile
  template <Dimension D>
  struct DensityProfile {
    DensityProfile(real_t B_min,
             real_t B_max,
             real_t Ly,
             real_t h,
             real_t r,
             real_t pT0,
             real_t kTe0,
             real_t kTi0,
             real_t Cj,
             real_t rho0)
    : eq { B_min, B_max, Ly, h, r, pT0, kTe0, kTi0, Cj, rho0 } {}
    Inline auto operator()(const coord_t<D>& x_Ph) const -> real_t { return 2 * eq.ns_eq(x_Ph[1]); }
  private:
    Equilibrium<D> eq;
  };

  // initialising the drifts and bulk perturbation
  template <Dimension D>
  struct DriftingMaxwellianZY {
    DriftingMaxwellianZY(random_number_pool_t& pool, real_t temperature,
                          real_t q_i, real_t q_e, real_t me, real_t miome,
                          real_t Ampl, real_t kx, real_t Lx, real_t Ly, bool is_ion,
                          Equilibrium<D> eq,
                          bool multimode_in = false,
                          int n_mode_pairs_in = 0,
                          real_t brms_in = 1.0,
                          Kokkos::View<int*> mode_i_in = {},
                          Kokkos::View<int*> mode_j_in = {},
                          Kokkos::View<real_t*> phi1_in = {},
                          Kokkos::View<real_t*> phi2_in = {})
      : pool { pool }, temperature { temperature }
      , q_i { q_i }, q_e { q_e }, me { me }, miome { miome }
      , Ampl { Ampl }, kx { kx }, Lx { Lx }, Ly { Ly }, is_ion { is_ion }, eq { eq }
      , multimode { multimode_in }
      , n_mode_pairs { n_mode_pairs_in }
      , brms { brms_in }
      , mode_i { mode_i_in }
      , mode_j { mode_j_in }
      , phi1 { phi1_in }
      , phi2 { phi2_in }
      {}

    Inline void operator()(const coord_t<D>& x_Ph, vec_t<Dim::_3D>& v) const {
      const auto x = x_Ph[0];
      const auto y = x_Ph[1];

      arch::energy_dist::SampleFromMaxwellian<false>(v, pool, temperature);

      if (is_ion) {
        v[0] += vx_ion(x, y);
        v[1] += vy_ion(x, y);
        v[2] += uz_ion(y) + vz_pert(x, y);
      } else {
        v[0] += vx_electron(x, y);
        v[1] += vy_electron(x, y);
        v[2] += uz_electron(y) + vz_pert(x, y);
      }
    }

  private:
    Inline auto bz_pert(real_t x, real_t y) const -> real_t {
      if (!multimode) {
        return Ampl * math::cos(kx / Lx * (x - Lx / 2));
      }
      real_t sum = ZERO;
      for (int m = 0; m < n_mode_pairs; m++) {
        real_t kx = 2 * PI * mode_i(m) / Lx;
        real_t ky = 2 * PI * mode_j(m) / Ly;
        real_t arg = kx * x + ky * y;
        sum += math::cos(arg + phi1(m));
        sum += math::cos(arg + phi2(m));
      }
      return sum / brms * Ampl;
    }
    Inline auto dbzdx(real_t x, real_t y) const -> real_t {
      if (!multimode) {
        return -kx / Lx * Ampl * math::sin(kx / Lx * (x - Lx / 2));
      }
      real_t sum = ZERO;
      for (int m = 0; m < n_mode_pairs; m++) {
        real_t kx = 2 * PI * mode_i(m) / Lx;
        real_t ky = 2 * PI * mode_j(m) / Ly;
        real_t arg = kx * x + ky * y;
        sum += -kx * math::sin(arg + phi1(m));
        sum += -kx * math::sin(arg + phi2(m));
      }
      return sum / brms * Ampl;
    }
    Inline auto dbzdy(real_t x, real_t y) const -> real_t { 
      if (!multimode) {
        return ZERO;
      }
      real_t sum = ZERO;
      for (int m = 0; m < n_mode_pairs; m++) {
        real_t kx = 2 * PI * mode_i(m) / Lx;
        real_t ky = 2 * PI * mode_j(m) / Ly;
        real_t arg = kx * x + ky * y;
        sum += -ky * math::sin(arg + phi1(m));
        sum += -ky * math::sin(arg + phi2(m));
      }
      return sum / brms * Ampl;
    }

    // Bulk perturbation
    Inline auto vz_pert(real_t x, real_t y) const -> real_t {
      if (!multimode) {
        return -bz_pert(x, y) / math::sqrt(eq.ns_eq(y) * me * (1 + miome));
      }
      real_t sum = ZERO;
      for (int m = 0; m < n_mode_pairs; m++) {
        real_t kx = 2 * PI * mode_i(m) / Lx;
        real_t ky = 2 * PI * mode_j(m) / Ly;
        real_t arg = kx * x + ky * y;
        sum += math::cos(arg + phi1(m));
        sum += math::cos(arg + phi2(m));
      }
      return -sum / brms * Ampl / math::sqrt(eq.ns_eq(y) * me * (1 + miome));
    }

    // Equilibrium species drift (total=0)
    Inline auto uz_ion(real_t y) const -> real_t {
      return eq.Jz_eq(y) / (q_i * eq.ns_eq(y) * (1 + miome));
    }
    Inline auto uz_electron(real_t y) const -> real_t {
      return -miome * uz_ion(y);
    }

    // 1st order species drift (total=0)
    Inline auto vx_ion(real_t x, real_t y) const -> real_t {
      const auto dJx  = dbzdy(x, y) / eq.Cj;
      const auto dJxi = dJx / (1 - (q_e / q_i) * miome);
      return dJxi / (q_i * eq.ns_eq(y));
    }
    Inline auto vx_electron(real_t x, real_t y) const -> real_t {
      const auto dJx  = dbzdy(x, y) / eq.Cj;
      const auto dJxe = dJx * miome / (miome - (q_e / q_i));
      return dJxe / (q_e * eq.ns_eq(y));
    }
    Inline auto vy_ion(real_t x, real_t y) const -> real_t {
      const auto dJy  = -dbzdx(x, y) / eq.Cj;
      const auto dJyi = dJy / (1 - (q_e / q_i) * miome);
      return dJyi / (q_i * eq.ns_eq(y));
    }
    Inline auto vy_electron(real_t x, real_t y) const -> real_t {
      const auto dJy  = -dbzdx(x, y) / eq.Cj;
      const auto dJye = dJy * miome / (miome - (q_e / q_i));
      return dJye / (q_e * eq.ns_eq(y));
    }

    random_number_pool_t pool;
    real_t temperature, q_i, q_e, me, miome, Ampl, kx, Lx, Ly;
    bool is_ion;
    Equilibrium<D> eq;
    bool multimode;
    int n_mode_pairs;
    real_t brms;
    Kokkos::View<int*>    mode_i;
    Kokkos::View<int*>    mode_j;
    Kokkos::View<real_t*> phi1;
    Kokkos::View<real_t*> phi2;
  };

  template <SimEngine::type S, class M>
  struct PGen {
    static constexpr auto D { M::Dim };
    // compatibility traits for the problem generator
    static constexpr auto engines = ::traits::pgen::compatible_with<SimEngine::SRPIC> {};
    static constexpr auto metrics =
      ::traits::pgen::compatible_with<Metric::Minkowski> {};
    static constexpr auto dimensions =
      ::traits::pgen::compatible_with<Dim::_2D> {};

    const SimulationParams& params;
    const std::vector<ParticleSpecies> species;

    const real_t                     c, d0, rho0, Cj, kx, betai0, betae0, Bmaxmin_ratio, h;
    const real_t                     r, me, q_e, mi, q_i, cAi0;
    const real_t                     ne0, ni0, n_tot, miome, fe, omegape0, omegapi0, de0, di0, B_max, B_min, Ampl;
    const real_t                     kTi0, kTe0, vthi0, vthe0;
    const real_t                     omegaCi0, omegaCe0, rhoCi0, rhoCe0, omegae_ratio, pT0;
    const real_t                     Lx, Ly;

    bool multimode;
    int  n_modes_shell;     // shell radius in k-space (e.g. 3)
    unsigned int seed;
    int  mm_direction;      // 0 = all, 1 = parallel only (ky=0), 2 = perpendicular only (kx=0)
    int  n_mode_pairs;      // actual number of (i,j) modes passing the filter
    real_t brms_val;        // RMS normalization factor
    Kokkos::View<int*>    mode_i_v;
    Kokkos::View<int*>    mode_j_v;
    Kokkos::View<real_t*> phi1_v;
    Kokkos::View<real_t*> phi2_v;

    InitFields<D>      init_flds;
    DensityProfile<D>  density_prof;

    PGen(const SimulationParams& p, const Metadomain<S, M>& global_domain)
      : params { p }
      , species { params.template get<std::vector<ParticleSpecies>>("particles.species") }
      , c    { ONE }
      , d0   { params.template get<real_t>("scales.skindepth0") }
      , rho0     { params.template get<real_t>("scales.larmor0") }
      , Cj   { rho0 / SQR(d0) }
      , kx   { 2 * PI * params.template get<real_t>("setup.kx", 1) }
      , betai0 { params.template get<real_t>("setup.betai0", HALF) }
      , betae0 { params.template get<real_t>("setup.betae0", HALF) }
      , Bmaxmin_ratio { params.template get<real_t>("setup.Bmaxmin_ratio", TWO) }
      , h { params.template get<real_t>("setup.h", 0.2) }
      , r { params.template get<real_t>("setup.r", 10) }
      , me { species[0].mass() }
      , q_e { species[0].charge() }
      , mi { species[1].mass() }
      , q_i { species[1].charge() }
      , cAi0 { params.template get<real_t>("setup.cAi0", 0.02) }
      // Derived Quantities
      , ne0 { HALF }
      , ni0 { HALF }
      , n_tot { ne0 + ni0 }
      , fe { miome / (miome - q_e / q_i) }
      , miome { mi / me }
      , omegape0 { (c / d0) * math::sqrt(ne0 * SQR(q_e) / me) }
      , omegapi0 { (c / d0) * math::sqrt(ni0 * SQR(q_i) / mi) }
      , de0 { c / omegape0 }
      , di0 { c / omegapi0 }
      , B_max { cAi0 * math::sqrt(ni0 * mi) }
      , B_min { B_max / Bmaxmin_ratio }
      , Ampl { B_max / 10 }
      , kTi0 { betai0 * SQR(B_max) / (TWO * ni0) }
      , kTe0 { betae0 * SQR(B_max) / (TWO * ne0) }
      , vthi0 { math::sqrt(kTi0 / mi) }
      , vthe0 { math::sqrt(kTe0 / me) }
      , omegaCi0 { q_i * B_max / (mi * rho0) }
      , omegaCe0 { math::abs(q_e) * B_max / (me * rho0) }
      , rhoCi0   { vthi0 / omegaCi0 }
      , rhoCe0   { vthe0 / omegaCe0 }
      , omegae_ratio { omegape0 / omegaCe0 }
      , pT0 { SQR(B_max) / TWO + ne0 * (kTe0 + kTi0) }
      // Dimensions
      , Lx { global_domain.mesh().extent(in::x1).second -
             global_domain.mesh().extent(in::x1).first }
      , Ly { global_domain.mesh().extent(in::x2).second -
             global_domain.mesh().extent(in::x2).first }
      // multimode vars
      , multimode { params.template get<bool>("setup.multimode", false) }
      , n_modes_shell { params.template get<int>("setup.n_modes", 3) }
      , seed { static_cast<unsigned int>(params.template get<int>("setup.seed", 42)) }
      , mm_direction { params.template get<int>("setup.mm_direction", 0) }
      , n_mode_pairs { 0 }
      , brms_val { ONE }
      // Default-construct views (empty, size 0)
      , mode_i_v {}
      , mode_j_v {}
      , phi1_v {}
      , phi2_v {}
      // Initial conditions
      , init_flds { B_min, B_max, Ly, h, r, 
              pT0, kTe0, kTi0, Ampl, Lx, kx, 
              me, miome, q_e, q_i, Cj, rho0 }
      , density_prof { B_min, B_max, Ly, h, r, pT0, kTe0, kTi0, Cj, rho0 }
      {

        if (multimode) {

          //count the number of (i,j) pairs
          int count = 0;
          for (int i = -n_modes_shell; i <= n_modes_shell; i++) {
              for (int j = -n_modes_shell; j <= n_modes_shell; j++) {
                  int r2 = i * i + j * j;
                  if (r2 > 0 && r2 <= n_modes_shell * n_modes_shell) {
                      if (mm_direction == 0
                          || (mm_direction == 1 && j == 0)
                          || (mm_direction == 2 && i == 0)) {
                          count++;
                      }
                  }
              }
          }
          n_mode_pairs = count;
            
          //allocate arrays on device (GPUs)
          mode_i_v = Kokkos::View<int*>("mode_i", count);
          mode_j_v = Kokkos::View<int*>("mode_j", count);
          phi1_v   = Kokkos::View<real_t*>("phi1", count);
          phi2_v   = Kokkos::View<real_t*>("phi2", count);          

          //create host mirrors
          auto mode_i_h = Kokkos::create_mirror_view(mode_i_v);
          auto mode_j_h = Kokkos::create_mirror_view(mode_j_v);
          auto phi1_h   = Kokkos::create_mirror_view(phi1_v);
          auto phi2_h   = Kokkos::create_mirror_view(phi2_v);

          // create random phases with fixed seed for reproducibility
          std::mt19937 rng(seed);
          std::uniform_real_distribution<double> dist(0.0, 2.0 * M_PI);
  
          int idx = 0;
          for (int i = -n_modes_shell; i <= n_modes_shell; i++) {
              for (int j = -n_modes_shell; j <= n_modes_shell; j++) {
                  int r2 = i * i + j * j;
                  if (r2 > 0 && r2 <= n_modes_shell * n_modes_shell) {
                      // Always draw phases to keep rng in sync
                      real_t p1 = static_cast<real_t>(dist(rng));
                      real_t p2 = static_cast<real_t>(dist(rng));
                      
                      if (mm_direction == 0
                          || (mm_direction == 1 && j == 0)
                          || (mm_direction == 2 && i == 0)) {
                          mode_i_h(idx) = i;
                          mode_j_h(idx) = j;
                          phi1_h(idx) = p1;
                          phi2_h(idx) = p2;
                          idx++;
                      }
                  }
              }
          }

          //send to device
          Kokkos::deep_copy(mode_i_v, mode_i_h);
          Kokkos::deep_copy(mode_j_v, mode_j_h);
          Kokkos::deep_copy(phi1_v,   phi1_h);
          Kokkos::deep_copy(phi2_v,   phi2_h);

          //compute brms on host for normalisation
          const int Nx_brms = 256;
          const int Ny_brms = 256;
          double sum_sq = 0.0;
          for (int ix = 0; ix < Nx_brms; ix++) {
            for (int iy = 0; iy < Ny_brms; iy++) {
              double x = (ix + 0.5) / Nx_brms * Lx;
              double y = -Ly / 2.0 + (iy + 0.5) / Ny_brms * Ly;
              double bz = 0.0;
              for (int m = 0; m < count; m++) {
                double kx = 2.0 * M_PI * mode_i_h(m) / Lx;
                double ky = 2.0 * M_PI * mode_j_h(m) / Ly;
                double arg = kx * x + ky * y;
                bz += std::cos(arg + static_cast<double>(phi1_h(m)));
                bz += std::cos(arg + static_cast<double>(phi2_h(m)));
              }
              sum_sq += bz * bz;
            }
          }
          brms_val = static_cast<real_t>(std::sqrt(2 * sum_sq / (Nx_brms * Ny_brms)));

          // Print phases for Python reconstruction
          printf("=== MULTIMODE PHASES (n_mode_pairs=%d, brms=%.10e, seed=%u, dir=%d) ===\n", count, brms_val, seed, mm_direction);
          for (int m = 0; m < count; m++) {
              printf("MODE %3d: i=%+2d j=%+2d phi1=%.15e phi2=%.15e\n",
                    m, mode_i_h(m), mode_j_h(m),
                    static_cast<double>(phi1_h(m)),
                    static_cast<double>(phi2_h(m)));
          }
          printf("=== END PHASES ===\n");

          //reassign init_flds with random perturbation
          init_flds = InitFields<D>(
            B_min, B_max, Ly, h, r,
            pT0, kTe0, kTi0, Ampl, Lx, kx,
            me, miome, q_e, q_i, Cj, rho0,
            true, n_mode_pairs, brms_val,
            mode_i_v, mode_j_v, phi1_v, phi2_v
          );          
        };
      }

    void InitPrtls(Domain<S, M>& domain) {
      const auto eq = Equilibrium<D>(B_min, B_max, Ly, h, r, pT0, kTe0, kTi0, Cj, rho0);

      const auto edist_e = DriftingMaxwellianZY<D>(
        domain.random_pool(), kTe0 / me,
        q_i, q_e, me, miome,
        Ampl, kx, Lx, Ly, false, eq,
        multimode, n_mode_pairs, brms_val,
        mode_i_v, mode_j_v, phi1_v, phi2_v
      );
      const auto edist_i = DriftingMaxwellianZY<D>(
        domain.random_pool(), kTi0 / mi,
        q_i, q_e, me, miome,
        Ampl, kx, Lx, Ly, true, eq,
        multimode, n_mode_pairs, brms_val,
        mode_i_v, mode_j_v, phi1_v, phi2_v
      );

      arch::InjectNonUniform<S, M, decltype(edist_e), decltype(edist_i), decltype(density_prof)>(
        params,
        domain,
        { 1, 2 },
        { edist_e, edist_i },
        density_prof,
        ONE,
        false);
      }
  };
} // namespace user

#endif // PROBLEM_GENERATOR_H