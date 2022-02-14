import numpy as np
from numpy import ndarray, float64
from scipy.constants import pi, m_e, k, e, epsilon_0

from .abc import Equations, Chemistry, PlasmaParameters
from .plasma_parameters import sanitize_power_series


mtorr = 0.133322  # 1 mTorr in Pa
sccm = 4.485e17  # 1 sccm in particles/s


class ElectronEnergyEquations(Equations):
    """A bases concrete `Equations` class resolving densities of all the
    heavy species in the `Chemistry` set as well as electron energy
    density. The neutral temperature is considered constant by this
    model (not resolved by this ODE system).

    The main purpose of this class is to build a function for the
    right-hand-side of an ODE system solving for the *state vector y*.
    All the `get_*` methods of the class take the state vector *y* as
    a parameter in the form of 1D array (described by the
    `ode_system_unknowns` instance attribute). In this instance,
    the state vector is (n0, ..., nN, rho), where n are densities of
    heavy species [m-3] and rho is the electron energy density [m-3.eV].

    Attributes
    ----------
    mask_sp_positive : ndarray[bool]
    mask_sp_negative : ndarray[bool]
    mask_sp_neutral : ndarray[bool]
    mask_sp_ions : ndarray[bool]
    mask_sp_flow : ndarray[bool]
    mask_r_electron : ndarray[bool]
    mask_r_elastic : ndarray[bool]
    num_species : int
    num_reactions : int
    sp_charges : ndarray[int]
    sp_masses : ndarray[float]
    sp_reduced_mass_matrix : ndarray[float]
    sp_flows : ndarray[float]
    sp_surface_sticking_coefficients : ndarray[float]
    sp_return_matrix : ndarray[float]
    sp_mean_velocities : ndarray[float]
        Mean velocities [SI] of all heavy species
    sp_sigma_sc : ndarray[float]
        2D array of hard-sphere scattering cross sections in [m2] for
        every species-species pair. Diagonal elements are all kept 0, as
        collisions with self do not contribute to diffusion of species.
        Only defined for n-n and n-i collisions.
    r_arrh_a : ndarray[float]
    r_arrh_b : ndarray[float]
    r_arrh_c : ndarray[float]
    r_el_energy_losses : ndarray[float]
    r_col_partner_masses : ndarray[float]
        Masses [kg] of heavy-species collisional partners. Only defined
        for elastic electron collisions, 0.0 otherwise.
    r_rate_coefs : ndarray[float]
        Reaction rate coefficients in SI
    r_stoich_electron_net : ndarray[int]
    r_stoichiomatrix_net : ndarray[int]
    r_stoichiomatrix_all_lhs : ndarray[int]
    temp_n : float
    pressure : float
    power : float
    volume : float
    area : float
    diff_l : float
    mean_cation_mass : float
    sheath_voltage_per_ev : float
        Sheath voltage [V] per 1eV of electron temperature.
    """

    # `Equations` ABC declares a mandatory `ode_system_unknowns`
    # attribute, which needs to be pre-defined here to instantiate the
    # class. However, this will be overridden by an instance attribute.
    ode_system_unknowns = None
    # Set the diffusion model: see documentation on ``get_wall_fluxes``.
    diffusion_model = 1

    def __init__(self, chemistry: Chemistry, plasma_params: PlasmaParameters):
        """"""
        self.chemistry = chemistry
        self.plasma_params = plasma_params
        self.ode_system_unknowns = list(chemistry.species_ids) + ["rho_e"]
        # densities are simply denoted by species names, rho_e is
        # the electron energy density

        # stubs for all the instance attributes:
        self.mask_sp_positive = None
        self.mask_sp_negative = None
        self.mask_sp_neutral = None
        self.mask_sp_ions = None
        self.mask_sp_flow = None
        self.mask_r_electron = None
        self.mask_r_elastic = None
        self.num_species = None
        self.num_reactions = None
        self.sp_charges = None
        self.sp_masses = None
        self.sp_reduced_mass_matrix = None
        self.sp_flows = None
        self.sp_mean_velocities = None
        self.sp_surface_sticking_coefficients = None
        self.sp_return_matrix = None
        self.sp_sigma_sc = None
        self.r_arrh_a = None
        self.r_arrh_b = None
        self.r_arrh_c = None
        self.r_el_energy_losses = None
        self.r_col_partner_masses = None
        self.r_rate_coefs = None
        self.r_stoich_electron_net = None
        self.r_stoichiomatrix_net = None
        self.r_stoichiomatrix_all_lhs = None
        self.temp_n = None
        self.pressure = None
        self.power = None
        self.volume = None
        self.area = None
        self.diff_l = None
        self.mean_cation_mass = None
        self.sheath_voltage_per_ev = None

        self.initialize_equations()

    def initialize_equations(
        self, chemistry: Chemistry = None, plasma_params: PlasmaParameters = None
    ):
        """"""
        if chemistry is None:
            chemistry = self.chemistry
        if plasma_params is None:
            plasma_params = self.plasma_params

        # MASKS filtering through species and reactions:
        self.mask_sp_positive = np.array(chemistry.species_charges) > 0
        self.mask_sp_negative = np.array(chemistry.species_charges) < 0
        self.mask_sp_neutral = np.array(chemistry.species_charges) == 0
        self.mask_sp_ions = self.mask_sp_positive | self.mask_sp_negative
        self.mask_sp_flow = np.array(
            [sp_id in plasma_params.feeds for sp_id in chemistry.species_ids],
            dtype=bool,
        )
        self.mask_r_electron = np.array(chemistry.reactions_electron_stoich_lhs) > 0
        self.mask_r_elastic = np.array(chemistry.reactions_elastic_flags)

        # STATIC PARAMETERS (not changing with the solver iterations)
        self.num_species = len(chemistry.species_ids)
        self.num_reactions = len(chemistry.reactions_ids)
        self.sp_charges = np.array(chemistry.species_charges)
        self.sp_masses = np.array(chemistry.species_masses)
        m_i = self.sp_masses[:, np.newaxis]
        m_k = self.sp_masses[np.newaxis, :]
        self.sp_reduced_mass_matrix = m_i * m_k / (m_i + m_k)
        self.sp_flows = np.array(
            plasma_params.feeds.get(sp_id, 0.0) for sp_id in chemistry.species_ids
        )
        self.sp_surface_sticking_coefficients = np.array(
            chemistry.species_surface_sticking_coefficients
        )
        self.sp_return_matrix = np.array(chemistry.species_surface_return_matrix)
        self.r_arrh_a = np.array(chemistry.reactions_arrh_a)
        self.r_arrh_b = np.array(chemistry.reactions_arrh_b)
        self.r_arrh_c = np.array(chemistry.reactions_arrh_c)
        self.r_col_partner_masses = np.zeros(self.num_reactions)
        for j in range(self.num_reactions):
            if self.mask_r_electron[j] and self.mask_r_elastic[j]:
                stoichiovector_lhs = np.array(
                    chemistry.reactions_species_stoichiomatrix_lhs[j]
                )
                self.r_col_partner_masses[j] = self.sp_masses[stoichiovector_lhs > 0][0]
        self.r_stoich_electron_net = np.array(
            chemistry.reactions_electron_stoich_rhs
        ) - np.array(chemistry.reactions_electron_stoich_lhs)
        self.r_stoichiomatrix_net = np.array(
            chemistry.reactions_species_stoichiomatrix_lhs
        ) - np.array(chemistry.reactions_species_stoichiomatrix_rhs)
        self.r_stoichiomatrix_all_lhs = np.c_[
            chemistry.reactions_electron_stoich_lhs,
            chemistry.reactions_species_stoichiomatrix_lhs,
            chemistry.reactions_arbitrary_stoich_lhs,
        ]  # stoichiomatrix with prepended e- column and appended M column
        self.temp_n = plasma_params.temp_n
        self.pressure = plasma_params.pressure
        t_power, power = sanitize_power_series(
            plasma_params.t_power, plasma_params.power, plasma_params.t_end
        )
        if len(set(power)) == 1:
            # power is constant 0-d scalar:
            self.power = power[0]
        else:
            # power is a function of time t returning a scalar:
            self.power = lambda t: np.interp(t, t_power, power)
        r, z = plasma_params.radius, plasma_params.length
        self.volume = pi * r**2 * z
        self.area = 2 * pi * r * (r + z)
        self.diff_l = ((pi / z) ** 2 + (2.405 / r) ** 2) ** -0.5
        self.mean_cation_mass = (
            self.sp_masses[self.mask_sp_positive].mean()
            if any(self.mask_sp_positive)
            else np.nan
        )
        self.sheath_voltage_per_ev = np.log(
            (self.mean_cation_mass / (2 * pi * m_e)) ** 0.5
        )

        # DYNAMIC PARAMETERS (updated with each `self.ode_system_rhs` call)
        self.r_el_energy_losses = np.array(chemistry.reactions_el_energy_losses)
        self.r_rate_coefs = np.empty(self.num_reactions)
        self.r_rate_coefs[~self.mask_r_electron] = (
            self.r_arrh_a[~self.mask_r_electron]
            * (self.temp_n / 300) ** self.r_arrh_b[~self.mask_r_electron]
            * np.e ** (-self.r_arrh_c[~self.mask_r_electron] / self.temp_n)
        )  # this bit stays static
        self.sp_mean_velocities = np.empty(self.num_species)
        self.sp_mean_velocities[self.mask_sp_neutral] = (
            8 * k * self.temp_n / (pi * self.sp_masses[self.mask_sp_neutral])
        ) ** 0.5  # this bit stays static
        self.sp_sigma_sc = (
            np.array(chemistry.species_lj_sigma_coefficients)[:, np.newaxis]
            + np.array(chemistry.species_lj_sigma_coefficients)[np.newaxis, :]
        ) ** 2
        # placeholder for rutherford scattering:
        self.sp_sigma_sc[np.ix_(~self.mask_sp_neutral, ~self.mask_sp_neutral)] = np.nan
        # diagonal held at zero:
        self.sp_sigma_sc[np.diag(len(self.sp_sigma_sc) * [True])] = 0.0

    @staticmethod
    def get_density_vector(y: ndarray) -> ndarray:
        """Extracts the vector of heavy species densities from the state
        vector y.

        Parameters
        ----------
        y : ndarray

        Returns
        -------
        ndarray
            Vector of heavy species densities in [m-3].
        """
        return y[:-1]

    @staticmethod
    def get_electron_energy_density(y: ndarray) -> float64:
        """Extracts the electron energy density from the state vector y.

        Parameters
        ----------
        y : ndarray

        Returns
        -------
        float64
            Electron energy density in [m-3.eV].
        """
        return y[-1]

    def get_total_density(self, y: ndarray, n: ndarray = None) -> float64:
        """Calculate the total heavy-species density in [m-3]. Uses
        ideal gas equation and the neutral gas temperature.

        Parameters
        ----------
        y : ndarray
        n : ndarray, optional
            Vector of densities [m-3] for all the heavy species.

        Returns
        -------
        float64
            Total heavy-species density scalar in [m-3]
        """
        if n is None:
            n = self.get_density_vector(y)
        n_tot = n.sum()
        return n_tot

    def get_total_pressure(self, y: ndarray, n_tot: float64 = None) -> float64:
        """Calculate the total pressure from the state vector y. Uses
        ideal gas equation and the neutral gas temperature.

        Parameters
        ----------
        y : ndarray
        n_tot : float64, optional
            Scalar of total density [m-3] (sum of densities of all
            the heavy species).

        Returns
        -------
        float64
            Instantaneous heavy-species pressure in [Pa].
        """
        if n_tot is None:
            n_tot = self.get_total_density(y)
        p = k * self.temp_n * n_tot
        return p

    def get_ion_temperature(self, y: ndarray, p: float64 = None) -> float64:
        """Calculates the ion temperature.

        The ion temperature is not used to evaluate reaction rate
        coefficients, pressure, etc, but is used only to calculate the
        coefficients of diffusivity.

        Parameters
        ----------
        y : ndarray
        p : float64, optional
            Scalar of total (heavy-species) pressure [Pa].

        Returns
        -------
        float64
            Ion temperature [K] for the diffusivity calculation.
        """
        if p is None:
            p = self.get_total_pressure(y)
        if p > mtorr:
            temp_i = (0.5 * e / k - self.temp_n) / (p / mtorr) + self.temp_n
        else:
            temp_i = 0.5 * e / k
        return temp_i

    def get_electron_density(self, y: ndarray, n: ndarray = None) -> float64:
        """Calculates the electron density from the state vector y.

        Calculated by enforcing the charge neutrality.

        Parameters
        ----------
        y : ndarray
        n : ndarray, optional
            Vector of densities of all the heavy species.

        Returns
        -------
        float64
            The electron density [m-3].
        """
        if n is None:
            n = self.get_density_vector(y)
        n_e = (n * self.sp_charges).sum()
        return n_e

    def get_electron_temperature(
        self, y: ndarray, n_e: float64 = None, rho: float64 = None
    ) -> float64:
        """Calculates the electron temperature from the state vector y,
        with a lower limit set by the gas temperature.

        Parameters
        ----------
        y : ndarray
        n_e : float64, optional
            Electron density [m-3].
        rho : float64, optional
            Electron energy density [eV.m-3].

        Returns
        -------
        float64
            Electron temperature [eV].
        """
        if n_e is None:
            n_e = self.get_electron_density(y)
        if rho is None:
            rho = self.get_electron_energy_density(y)
        temp_e = rho / n_e * 2 / 3
        temp_n_ev = float64(self.temp_n * k / e)
        # noinspection PyTypeChecker
        return max(temp_e, temp_n_ev)

    def get_debye_length(
        self, y: ndarray, n_e: float64 = None, temp_e: float64 = None
    ) -> float64:
        """Calculates the Debye length for this plasma model.

        Parameters
        ----------
        y : ndarray
        n_e : float64, optional
            Electron density [m-3].
        temp_e : float64, optional
            Electron temperature [eV]

        Returns
        -------
        float64
            Debye length in [m]
        """
        if n_e is None:
            n_e = self.get_electron_density(y)
        if temp_e is None:
            temp_e = self.get_electron_temperature(y)
        return (epsilon_0 * temp_e / e / n_e) ** 0.5

    def get_reaction_rate_coefficients(
        self, y: ndarray, temp_e: float64 = None
    ) -> ndarray:
        """Calculate the vector of reaction rate coefficients for all
        the reactions from the state vector y.

        Parameters
        ----------
        y : ndarray
        temp_e : float64, optional
            Electron temperature [eV].

        Returns
        -------
        ndarray
            Rate coefficients [SI] for all the reactions in the set.
        """
        if temp_e is None:
            temp_e = self.get_electron_temperature(y)

        self.r_rate_coefs[self.mask_r_electron] = (
            self.r_arrh_a[self.mask_r_electron]
            * temp_e ** self.r_arrh_b[self.mask_r_electron]
            * np.e ** (-self.r_arrh_c[self.mask_r_electron] / temp_e)
        )
        return self.r_rate_coefs

    def get_reaction_rates(
        self,
        y: ndarray,
        n: ndarray = None,
        n_e: float64 = None,
        n_tot: float64 = None,
        k_r: ndarray = None,
    ):
        """Calculates vector of reaction rates for all the reactions.

        Parameters
        ----------
        y : ndarray
        n : ndarray, optional
            Densities [m-3] for all the heavy species.
        n_e : float64, optional
            Electron density [m-3].
        n_tot : float64, optional
            Total heavy-species density [m-3].
        k_r : ndarray, optional
            Vector of reaction rate coefficients [SI] for all the
            reactions in the set.

        Returns
        -------
        ndarray
            Reaction rates [m-3/s] for all the reactions in the set.
        """
        if n is None:
            n = self.get_density_vector(y)
        if n_e is None:
            n_e = self.get_electron_density(y, n=n)
        if n_tot is None:
            n_tot = self.get_total_density(y, n=n)
        if k_r is None:
            k_r = self.get_reaction_rate_coefficients(y)
        n_all = np.r_[n_e, n, n_tot]
        n_all_prod = np.prod(
            n_all[np.newaxis, :] ** self.r_stoichiomatrix_all_lhs, axis=1
        )
        rates = n_all_prod * k_r
        return rates

    def get_volumetric_source_rates(self, y: ndarray, rates: ndarray = None) -> ndarray:
        """Calculates the contributions to the time derivatives of
        heavy species densities due to volumetric reactions.

        Parameters
        ----------
        y : ndarray
        rates : ndarray, optional
            Reaction rates [m-3/s] for all the reactions in the set.

        Returns
        -------
        ndarray
            Vector of contributions to the time derivatives of heavy
            species densities due to volumetric reactions in [m-3/s].
            Length as number of heavy species.
        """
        if rates is None:
            rates = self.get_reaction_rates(y)
        # volumetric reactions production rates for all species [m-3/s]
        source_vol = np.sum(self.r_stoichiomatrix_net * rates[:, np.newaxis], axis=0)
        return source_vol

    def get_flow_source_rates(
        self, y: ndarray, n: ndarray = None, p: float64 = None
    ) -> ndarray:
        """Calculates the contributions to the time derivatives of
        heavy species densities due to the gas flows (in and out).

        Parameters
        ----------
        y : ndarray
        n : ndarray, optional
            Vector of densities [m-3] of all the heavy species.
        p : float64, optional
            Total instantaneous (heavy species) pressure in [Pa].

        Returns
        -------
        ndarray
            Vector of contributions to the time derivatives of heavy
            species densities due to gas flows, in and out, in [m-3/s].
            Length as number of heavy species.
        """
        if n is None:
            n = self.get_density_vector(y)
        if p is None:
            p = self.get_total_pressure(y)
        t_flow = 1e-3  # pressure recovery time in [s]
        source_flow = np.zeros(self.num_species)
        # loss rate due to the pressure regulation, acting on neutrals:
        source_flow[self.mask_sp_neutral] -= (
            n[self.mask_sp_neutral] * (1 / t_flow) * (p - self.pressure) / self.pressure
        )
        # gain rate due to the inflow:
        source_flow += self.sp_flows * sccm / self.volume
        # loss rate due to the outflow, acting on all neutrals
        source_flow[self.mask_sp_neutral] -= (
            sum(self.sp_flows * sccm / self.volume)
            * n[self.mask_sp_neutral]
            / sum(n[self.mask_sp_neutral])
        )
        return source_flow

    def get_mean_speeds(self, y: ndarray, temp_i: float64 = None) -> ndarray:
        """Calculates the mean thermal speeds for all the heavy species.

        Only the ion values are dynamically calculated, the neutral
        values stay static.

        Parameters
        ----------
        y : ndarray
        temp_i : float64, optional
            Ion temperature [K] for the diffusivity calculation.

        Returns
        -------
        ndarray
            Vector of mean thermal speeds [m/s] for all heavy species.
        """
        if temp_i is None:
            temp_i = self.get_ion_temperature(y)
        self.sp_mean_velocities[~self.mask_sp_neutral] = (
            8 * k * temp_i / (pi * self.sp_masses[~self.mask_sp_neutral])
        ) ** 0.5
        return self.sp_mean_velocities

    def get_sigma_sc(
        self, y: ndarray, v_m: ndarray = None, debye_length: float64 = None
    ) -> ndarray:
        """Calculates the matrix for species-to-species momentum
        transfer scattering cross sections.

        Only the ion-ion pairs are dynamically calculated, the rest stay
        static.

        Parameters
        ----------
        y : ndarray
        v_m : ndarray, optional
            Mean thermal speeds for all the heavy species in [m/s].
        debye_length : float64, optional
            Debye length of the plasma in [m].

        Returns
        -------
        ndarray
            2-d matrix of cross sections [m2] for momentum transfer
            between each pair of species.
        """
        if v_m is None:
            v_m = self.get_mean_speeds(y)
        if debye_length is None:
            debye_length = self.get_debye_length(y)
        # distance of closest approach matrix for each ion-ion pair:
        b_0_ii = (
            e**2
            * abs(
                self.sp_charges[~self.mask_sp_neutral, np.newaxis]
                * self.sp_charges[np.newaxis, ~self.mask_sp_neutral]
            )
            / (2 * pi * epsilon_0)
            / (
                self.sp_reduced_mass_matrix[
                    np.ix_(~self.mask_sp_neutral, ~self.mask_sp_neutral)
                ]
                * v_m[~self.mask_sp_neutral, np.newaxis] ** 2
            )
        )
        self.sp_sigma_sc[np.ix_(~self.mask_sp_neutral, ~self.mask_sp_neutral)] = (
            pi * b_0_ii**2 * np.log(2 * debye_length / b_0_ii)
        )
        self.sp_sigma_sc[np.diag(len(self.sp_sigma_sc) * [True])] = 0.0
        return self.sp_sigma_sc

    def get_mean_free_paths(
        self, y: ndarray, n: ndarray = None, sigma_sc: ndarray = None
    ) -> ndarray:
        """Calculates the vector of mean free paths for all the heavy
        species.

        Parameters
        ----------
        y : ndarray
        n : ndarray, optional
            Vector of densities [m-3] of all the heavy species.
        sigma_sc : ndarray, optional
            2-d matrix of momentum transfer cross sections [m2] for
            each species pair.

        Returns
        -------
        ndarray
            Vector of mean free paths [m] for all the heavy species.
        """
        if n is None:
            n = self.get_density_vector(y)
        if sigma_sc is None:
            sigma_sc = self.get_sigma_sc(y)
        mfp = 1 / np.sum(n[np.newaxis, :] * sigma_sc, axis=1)
        return mfp

    def get_free_diffusivities(
        self, y: ndarray, mfp: ndarray = None, v_m: ndarray = None
    ) -> ndarray:
        """Calculate the free diffusion coefficients for all the heavy
        species.

        Parameters
        ----------
        y : ndarray
        mfp : ndarray, optional
            Mean-free-paths of all the heavy species in [m].
        v_m : ndarray, optional
            Mean thermal speeds of all the heavy species in [m/s]

        Returns
        -------
        ndarray
            Vector of coefficients of free diffusion for the heavy
            species in [SI].
        """
        if mfp is None:
            mfp = self.get_mean_free_paths(y)
        if v_m is None:
            v_m = self.get_mean_speeds(y)
        return pi / 8 * mfp * v_m

    @property
    def ode_system_rhs(self):
        return
