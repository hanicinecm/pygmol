"""A module providing a concrete subclass of the `Equations` ABC, which solves for
the densities of heavy species and the electron energy density.
"""
from typing import Callable, List, Dict

import numpy as np
from numpy import ndarray, float64

from .abc import Equations, Chemistry, PlasmaParameters
from .plasma_parameters import sanitize_power_series


class ElectronEnergyEquations(Equations):
    """A bases concrete `Equations` class resolving densities of all the heavy species
    in the `Chemistry` set as well as electron energy density. The neutral temperature
    is considered constant by this model (not resolved by this ODE system).

    The main purpose of this class is to build a function for the right-hand-side of an
    ODE system solving for the *state vector y*. All the `get_*` methods of the class
    take the state vector *y* as a parameter in the form of 1D array (described by the
    `ode_system_unknowns` instance attribute). In this instance, the state vector is
    (n0, ..., nN, rho), where n are densities of heavy species [m-3] and rho is the
    electron energy density [m-3.eV].

    Attributes
    ----------
    chemistry : Chemistry
    plasma_params : PlasmaParameters
    mask_sp_positive : ndarray[bool]
    mask_sp_negative : ndarray[bool]
    mask_sp_neutral : ndarray[bool]
    mask_r_electron : ndarray[bool]
    mask_r_elastic : ndarray[bool]
    num_species : int
    num_reactions : int
    sp_charges : ndarray[int64]
    sp_masses : ndarray[float64]
    sp_lj_sigma_coefficients : ndarray[float64]
    sp_reduced_mass_matrix : ndarray[float64]
    sp_flows : ndarray[float64]
    sp_surface_sticking_coefficients : ndarray[float64]
    sp_return_matrix : ndarray[float64]
        2D array.
    sp_mean_velocities : ndarray[float64]
        Mean velocities [SI] of all heavy species
    sp_sigma_sc : ndarray[float64]
        2D array of hard-sphere scattering cross sections in [m2] for every
        species-species pair. Diagonal elements are all kept 0, as collisions with self
        do not contribute to diffusion of species. Only defined for n-n and n-i
        collisions.
    r_arrh_a : ndarray[float64]
    r_arrh_b : ndarray[float64]
    r_arrh_c : ndarray[float64]
    r_el_energy_losses : ndarray[float64]
    r_col_partner_masses : ndarray[float64]
        Masses [kg] of heavy-species collisional partners. Only defined for elastic
        electron collisions, 0.0 otherwise.
    r_rate_coefs : ndarray[float64]
        Reaction rate coefficients in [SI].
    r_stoich_electron_net : ndarray[int64]
    r_stoichiomatrix_net : ndarray[int64]
    r_stoichiomatrix_all_lhs : ndarray[int64]
    temp_n : float
    pressure : float
    power : float
    volume : float
    area : float
    diff_l : float
    mean_cation_mass : float64
    sheath_voltage_per_ev : float64
        Sheath voltage [V] per 1eV of electron temperature.
    """

    # Set the diffusion model: see documentation on ``get_wall_fluxes``.
    diffusion_model = 1

    def __init__(self, chemistry: Chemistry, plasma_params: PlasmaParameters):
        """The Equations initializer.

        Parameters
        ----------
        chemistry : Chemistry
            An instance of an `abc.Chemistry` subclass.
        plasma_params : PlasmaParameters
            An instance of an `abc.PlasmaParams` subclass.
        """
        # the ABC constructor just saves the arguments as instance attributes
        super().__init__(chemistry, plasma_params)

        # stubs for all the instance attributes:
        self.mask_sp_positive = None
        self.mask_sp_negative = None
        self.mask_sp_neutral = None
        self.mask_r_electron = None
        self.mask_r_elastic = None
        self.num_species = None
        self.num_reactions = None
        self.sp_charges = None
        self.sp_masses = None
        self.sp_lj_sigma_coefficients = None
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
        """Method initializing all the static and dynamic instance attributes used
        by all the methods ultimately leading the the final `ode_system_rhs` property.

        Parameters
        ----------
        chemistry : Chemistry
            Class encoding the chemistry set, such as species, reactions, etc.
        plasma_params : PlasmaParameters
            Class encoding the plasma parameters, such as pressure, power, etc.
        """
        if chemistry is None:
            chemistry = self.chemistry
        if plasma_params is None:
            plasma_params = self.plasma_params

        # MASKS filtering through species and reactions:
        self.mask_sp_positive = np.array(chemistry.species_charges) > 0
        self.mask_sp_negative = np.array(chemistry.species_charges) < 0
        self.mask_sp_neutral = np.array(chemistry.species_charges) == 0
        self.mask_r_electron = np.array(chemistry.reactions_electron_stoich_lhs) > 0
        self.mask_r_elastic = np.array(chemistry.reactions_elastic_flags)

        # STATIC PARAMETERS (not changing with the solver iterations)
        self.num_species = len(chemistry.species_ids)
        self.num_reactions = len(chemistry.reactions_ids)
        self.sp_charges = np.array(chemistry.species_charges)
        self.sp_masses = np.array(chemistry.species_masses) * self.atomic_mass
        m_i = self.sp_masses[:, np.newaxis]
        m_k = self.sp_masses[np.newaxis, :]
        self.sp_reduced_mass_matrix = m_i * m_k / (m_i + m_k)
        self.sp_lj_sigma_coefficients = (
            np.array(chemistry.species_lj_sigma_coefficients) * 1.0e-10
        )
        self.sp_flows = np.array(
            [plasma_params.feeds.get(sp_id, 0.0) for sp_id in chemistry.species_ids]
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
            chemistry.reactions_species_stoichiomatrix_rhs
        ) - np.array(chemistry.reactions_species_stoichiomatrix_lhs)
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
        self.volume = self.pi * r**2 * z
        self.area = 2 * self.pi * r * (r + z)
        self.diff_l = ((self.pi / z) ** 2 + (2.405 / r) ** 2) ** -0.5
        self.mean_cation_mass = (
            self.sp_masses[self.mask_sp_positive].mean()
            if any(self.mask_sp_positive)
            else np.nan
        )
        self.sheath_voltage_per_ev = np.log(
            (self.mean_cation_mass / (2 * self.pi * self.m_e)) ** 0.5
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
            8 * self.k * self.temp_n / (self.pi * self.sp_masses[self.mask_sp_neutral])
        ) ** 0.5  # this bit stays static
        self.sp_sigma_sc = (
            self.sp_lj_sigma_coefficients[:, np.newaxis]
            + self.sp_lj_sigma_coefficients[np.newaxis, :]
        ) ** 2
        # placeholder for rutherford scattering:
        self.sp_sigma_sc[np.ix_(~self.mask_sp_neutral, ~self.mask_sp_neutral)] = np.nan
        # diagonal held at zero:
        self.sp_sigma_sc[np.diag(len(self.sp_sigma_sc) * [True])] = 0.0

    @staticmethod
    def get_density_vector(y: ndarray) -> ndarray:
        """Extracts the vector of heavy species densities from the state vector y.

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
        """Calculate the total heavy-species density in [m-3]. Uses ideal gas equation
        and the neutral gas temperature.

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
        """Calculate the total pressure from the state vector y. Uses ideal gas equation
        and the neutral gas temperature.

        Parameters
        ----------
        y : ndarray
        n_tot : float64, optional
            Scalar of total density [m-3] (sum of densities of all the heavy species).

        Returns
        -------
        float64
            Instantaneous heavy-species pressure in [Pa].
        """
        if n_tot is None:
            n_tot = self.get_total_density(y)
        p = self.k * self.temp_n * n_tot
        return p

    def get_ion_temperature(self, y: ndarray, p: float64 = None) -> float64:
        """Calculates the ion temperature.

        The ion temperature is not used to evaluate reaction rate coefficients,
        pressure, etc, but is used only to calculate the coefficients of diffusivity.

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
        if p > self.mtorr:
            temp_i = (0.5 * self.e / self.k - self.temp_n) / (
                p / self.mtorr
            ) + self.temp_n
        else:
            temp_i = 0.5 * self.e / self.k
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
        """Calculates the electron temperature from the state vector y, with a lower
        limit set by the gas temperature.

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
        temp_n_ev = float64(self.temp_n * self.k / self.e)
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
        return (self.epsilon_0 * temp_e / self.e / n_e) ** 0.5

    def get_reaction_rate_coefficients(
        self, y: ndarray, temp_e: float64 = None
    ) -> ndarray:
        """Calculate the vector of reaction rate coefficients for all the reactions from
        the state vector y.

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
    ) -> ndarray:
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
            Vector of reaction rate coefficients [SI] for all the reactions in the set.

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
        """Calculates the contributions to the time derivatives of heavy species
        densities due to volumetric reactions.

        Parameters
        ----------
        y : ndarray
        rates : ndarray, optional
            Reaction rates [m-3/s] for all the reactions in the set.

        Returns
        -------
        ndarray
            Vector of contributions to the time derivatives of heavy species densities
            due to volumetric reactions in [m-3/s]. Length as number of heavy species.
        """
        if rates is None:
            rates = self.get_reaction_rates(y)
        # volumetric reactions production rates for all species [m-3/s]
        source_vol = np.sum(self.r_stoichiomatrix_net * rates[:, np.newaxis], axis=0)
        return source_vol

    def get_flow_source_rates(
        self, y: ndarray, n: ndarray = None, p: float64 = None
    ) -> ndarray:
        """Calculates the contributions to the time derivatives of heavy species
        densities due to the gas flows (in and out).

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
            Vector of contributions to the time derivatives of heavy species densities
            due to gas flows, in and out, in [m-3/s]. Length as number of heavy species.
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
        source_flow += self.sp_flows * self.sccm / self.volume
        # loss rate due to the outflow, acting on all neutrals
        source_flow[self.mask_sp_neutral] -= (
            sum(self.sp_flows * self.sccm / self.volume)
            * n[self.mask_sp_neutral]
            / sum(n[self.mask_sp_neutral])
        )
        return source_flow

    def get_mean_speeds(self, y: ndarray, temp_i: float64 = None) -> ndarray:
        """Calculates the mean thermal speeds for all the heavy species.

        Only the ion values are dynamically calculated, the neutral values stay static.

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
            8 * self.k * temp_i / (self.pi * self.sp_masses[~self.mask_sp_neutral])
        ) ** 0.5
        return self.sp_mean_velocities

    def get_sigma_sc(
        self, y: ndarray, v_m: ndarray = None, debye_length: float64 = None
    ) -> ndarray:
        """Calculates the matrix for species-to-species momentum transfer scattering
        cross sections.

        Only the ion-ion pairs are dynamically calculated, the rest stay static.

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
            2-d matrix of cross sections [m2] for momentum transfer between each pair of
            species.
        """
        if v_m is None:
            v_m = self.get_mean_speeds(y)
        if debye_length is None:
            debye_length = self.get_debye_length(y)
        # distance of closest approach matrix for each ion-ion pair:
        b_0_ii = (
            self.e**2
            * abs(
                self.sp_charges[~self.mask_sp_neutral, np.newaxis]
                * self.sp_charges[np.newaxis, ~self.mask_sp_neutral]
            )
            / (2 * self.pi * self.epsilon_0)
            / (
                self.sp_reduced_mass_matrix[
                    np.ix_(~self.mask_sp_neutral, ~self.mask_sp_neutral)
                ]
                * v_m[~self.mask_sp_neutral, np.newaxis] ** 2
            )
        )
        self.sp_sigma_sc[np.ix_(~self.mask_sp_neutral, ~self.mask_sp_neutral)] = (
            self.pi * b_0_ii**2 * np.log(2 * debye_length / b_0_ii)
        )
        self.sp_sigma_sc[np.diag(len(self.sp_sigma_sc) * [True])] = 0.0
        return self.sp_sigma_sc

    def get_mean_free_paths(
        self, y: ndarray, n: ndarray = None, sigma_sc: ndarray = None
    ) -> ndarray:
        """Calculates the vector of mean free paths for all the heavy species.

        Parameters
        ----------
        y : ndarray
        n : ndarray, optional
            Vector of densities [m-3] of all the heavy species.
        sigma_sc : ndarray, optional
            2-d matrix of momentum transfer cross sections [m2] for each species pair.

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
        """Calculate the free diffusion coefficients for all the heavy species.

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
            Vector of coefficients of free diffusion for the heavy species in [SI].
        """
        if mfp is None:
            mfp = self.get_mean_free_paths(y)
        if v_m is None:
            v_m = self.get_mean_speeds(y)
        return self.pi / 8 * mfp * v_m

    def get_ambipolar_diffusivity_pos(
        self,
        y: ndarray,
        n: ndarray = None,
        n_e: float64 = None,
        temp_i: float64 = None,
        temp_e: float64 = None,
        diff_c_free: ndarray = None,
    ) -> float64:
        """Calculate the coefficient of ambipolar diffusion for positive ions.

        Parameters
        ----------
        y : ndarray
        n : ndarray, optional
            Vector of densities [m-3] for all the heavy species.
        n_e : float64, optional
            Electron density [eV].
        temp_i : float64, optional
            Ion temperature [K].
        temp_e : float64, optional
            Electron temperature [eV].
        diff_c_free : ndarray, optional
            Vector of coefficients of free diffusion for all the heavy species [SI].

        Returns
        -------
        float64
            Coefficient of ambipolar diffusion [SI] for positive ions.
        """
        if n is None:
            n = self.get_density_vector(y)
        if n_e is None:
            n_e = self.get_electron_density(y, n=n)
        if temp_i is None:
            temp_i = self.get_ion_temperature(y)
        if temp_e is None:
            temp_e = self.get_electron_temperature(y, n_e=n_e)
        if diff_c_free is None:
            diff_c_free = self.get_free_diffusivities(y)

        gamma = temp_e * self.e / self.k / temp_i
        alpha = n[self.mask_sp_negative].sum() / n_e
        diff_free_pos = diff_c_free[self.mask_sp_positive].mean()
        # NOTE: this only holds for alpha << mu_e/mu_i
        diff_a_pos = diff_free_pos * (1 + gamma * (1 + 2 * alpha)) / (1 + alpha * gamma)
        return diff_a_pos

    def get_diffusivities(
        self,
        y: ndarray,
        diff_c_free: ndarray = None,
        diff_a_pos: float64 = None,
    ) -> ndarray:
        """Calculates the diffusion coefficients.

        The neutrals diffusivities are free diffusivities for neutrals (given by the
        mixture rules using LJ potentials), the +ion diffusivities are equal to
        ambipolar diffusion coefficients for positive ions, and the -ions diffusivities
        are 0.0 in this model.

        Parameters
        ----------
        y : ndarray
        diff_c_free : ndarray, optional
            Vector of coefficients of free diffusion for all the heavy species [SI].
        diff_a_pos : float64, optional
            Coefficient of ambipolar diffusion [SI] for positive ions.

        Returns
        -------
        ndarray
            Vector of diffusion coefficients for all heavy species [SI].
        """
        if diff_c_free is None:
            diff_c_free = self.get_free_diffusivities(y)
        if diff_a_pos is None:
            diff_a_pos = self.get_ambipolar_diffusivity_pos(y, diff_c_free=diff_c_free)
        diff_c = np.empty(self.num_species)
        diff_c[self.mask_sp_neutral] = diff_c_free[self.mask_sp_neutral]
        diff_c[self.mask_sp_positive] = diff_a_pos
        diff_c[self.mask_sp_negative] = 0.0
        return diff_c

    def get_wall_fluxes(
        self, y: ndarray, n: ndarray = None, diff_c: ndarray = None, v_m: ndarray = None
    ) -> ndarray:
        """Calculate the vector of wall-sticking fluxes for all the heavy species.

        The fluxes therefore already take into the account the sticking coefficients -
        if sticking coefficients are null for certain species, fluxes for those will
        be null also.

        This method ONLY takes into account STICKING fluxes. If any species is getting
        stuck to the surface, it will have a negative wall flux returned by this method.
        But the same species might have return coefficient defined as 1.0 with the
        return species of itself, which will mean that the rate of density change due to
        the surface interactions for this species will still be null.

        This is simply how the `wall_fluxes` are defined in this work. The 'in-fluxes'
        of returned species are not at all taken into account by this method! All the
        fluxes are negative by convention (particles "moving out of the system").

        The fluxes depend on the diffusion model (class attribute):
        - 0: wall flux is a pure diffusive flux (Lietz2016)
        - 1: wall flux is a combination of diffusive and thermal flux (Schroter2018)

        Parameters
        ----------
        y : ndarray
        n : ndarray, optional
            Vector of heavy species densities [m-3].
        diff_c : ndarray, optional
            Vector of diffusion coefficients [SI].
        v_m : ndarray, optional
            Vector of thermal speeds of all the heavy species [m/s].

        Returns
        -------
        ndarray
            Vector of the heavy species out-fluxes in [m-2/s]
        """
        if n is None:
            n = self.get_density_vector(y)
        if diff_c is None:
            diff_c = self.get_diffusivities(y)
        if v_m is None:
            v_m = self.get_mean_speeds(y)
        s = self.sp_surface_sticking_coefficients
        if self.diffusion_model == 0:
            return -diff_c * n * s / self.diff_l**2 * self.volume / self.area
        elif self.diffusion_model == 1:
            wall_fluxes = -diff_c * n * s  # just the numerator
            wall_fluxes[s != 0] /= (s * self.diff_l + (4 * diff_c / v_m))[s != 0]
            # `diff_c` is 0.0 for negative ions, which would lead to division by 0 for
            # non-sticking negative ions, therefore dividing only for non-zero sticking
            return wall_fluxes
        else:
            raise ValueError("Unsupported diffusion model!")

    def get_surface_loss_rates(
        self, y: ndarray, wall_fluxes: ndarray = None
    ) -> ndarray:
        """Calculates the vector of contributions to time derivatives of densities due
        to the diffusion sinks to the walls. Only sinks, no sources.

        Parameters
        ----------
        y : ndarray
        wall_fluxes : ndarray, optional
            Vector of the heavy species out-fluxes in [m-2/s].

        Returns
        -------
        ndarray
            The vector of contributions to time derivatives of densities due to the
            diffusion losses for all the heavy species in [m-3/s].
        """
        if wall_fluxes is None:
            wall_fluxes = self.get_wall_fluxes(y)
        surf_loss_rates = wall_fluxes * self.area / self.volume
        return surf_loss_rates

    def get_surface_source_rates(
        self, y: ndarray, surf_loss_rates: ndarray = None
    ) -> ndarray:
        """Calculate the vector of contributions to time derivatives of densities due to
        diffusion sources from the walls, caused by return-species re-injection from the
        walls to the plasma.

        Parameters
        ----------
        y : ndarray
        surf_loss_rates : ndarray, optional
            The vector of contributions to time derivatives of densities due to the
            diffusion losses for all the heavy species in [m-3/s].

        Returns
        -------
        ndarray
            The vector of contributions to time derivatives of densities due to the
            wall-return sources for all the heavy species in [m-3/s].
        """
        if surf_loss_rates is None:
            surf_loss_rates = self.get_surface_loss_rates(y)
        return np.sum(-surf_loss_rates[np.newaxis, :] * self.sp_return_matrix, axis=1)

    def get_diffusion_source_rates(
        self,
        y: ndarray,
        surf_loss_rates: ndarray = None,
        surf_source_rates: ndarray = None,
    ) -> ndarray:
        """Calculate the vector of contributions to time derivatives of densities due to
        diffusion sinks and wall-return sources.

        Parameters
        ----------
        y : ndarray
        surf_loss_rates : ndarray, optional
            The vector of contributions to time derivatives of densities due to the
            diffusion losses for all the heavy species in [m-3/s].
        surf_source_rates : ndarray, optional
            The vector of contributions to time derivatives of densities due to the
            wall-return sources for all the heavy species in [m-3/s].

        Returns
        -------
        ndarray
            The vector of contributions to time derivatives of densities due to the
            diffusion sinks and wall-return sources in [m-3/s].
        """
        if surf_loss_rates is None:
            surf_loss_rates = self.get_surface_loss_rates(y)
        if surf_source_rates is None:
            surf_source_rates = self.get_surface_source_rates(
                y, surf_loss_rates=surf_loss_rates
            )
        return surf_loss_rates + surf_source_rates

    def get_min_n_correction(self, y: ndarray, n: ndarray = None) -> ndarray:
        """This is an artificial (unphysical) correction applied to the RHS of the
        densities ODE system preventing the densities to go under a minimal value (and
        ultimately from reaching unphysical negative densities.) It supplies a nudge for
        all the densities below a lower limit which is proportional to the difference of
        the densities and the limit.

        Parameters
        ----------
        y : ndarray
        n : ndarray, optional
            Vector of densities of all heavy species in [m-3].

        Returns
        -------
        ndarray
            Vector of corrections in [m-3/s] preventing the densities from reaching
            values lower than (hard-coded) `n_min`.
        """
        if n is None:
            n = self.get_density_vector(y)
        n_min = 1.0e0
        t_rec = 1.0e-10  # recovery time - should be approx the solver time step time
        below_min_mask = n < n_min
        min_n_correction = np.zeros(len(n))
        min_n_correction[below_min_mask] = (n_min - n[below_min_mask]) / t_rec
        return min_n_correction

    def get_dn_over_dt(
        self,
        y: ndarray,
        vol_source_rates: ndarray = None,
        flow_source_rates: ndarray = None,
        diff_source_rates: ndarray = None,
        min_n_correction: ndarray = None,
    ) -> ndarray:
        """Calculates the vector of final time derivatives of densities for all the
        heavy species.

        Parameters
        ----------
        y : ndarray,
        vol_source_rates : ndarray, optional
        flow_source_rates : ndarray, optional
        diff_source_rates : ndarray, optional
        min_n_correction : ndarray, optional

        Returns
        -------
        ndarray
            Vector of time derivatives of densities [m-3/s] of all the heavy species.
        """
        if vol_source_rates is None:
            vol_source_rates = self.get_volumetric_source_rates(y)
        if flow_source_rates is None:
            flow_source_rates = self.get_flow_source_rates(y)
        if diff_source_rates is None:
            diff_source_rates = self.get_diffusion_source_rates(y)
        if min_n_correction is None:
            min_n_correction = self.get_min_n_correction(y)

        return (
            vol_source_rates + flow_source_rates + diff_source_rates + min_n_correction
        )

    def get_power_ext(self, t: float64) -> float64:
        """Returns the instant absorbed power *P(t)* [W] at the time *t*. Unlike other
        methods, this one is only dependent on time.

        Parameters
        ----------
        t : float or float64
            Time in [sec].

        Returns
        -------
        float64
            Instantaneous external power *P(t)* in [W] supplied to the plasma.
        """
        if callable(self.power):
            return self.power(t)
        else:
            # time-independent power
            return self.power

    def get_drho_over_dt_ext(self, t: float64, power_ext: float64 = None) -> float64:
        """Calculate the contribution of absorbed power to the time derivative of
        electron energy density. This unlike other methods is only dependent on time.

        Parameters
        ----------
        t : float64
            Time in [sec].
        power_ext : float64, optional
            External (instantaneous) power *P(t)* in [W].

        Returns
        -------
        float64
            The contribution to the time derivative of electron energy density due to
            the absorbed external power in [eV.m-3/s].
        """
        if power_ext is None:
            power_ext = self.get_power_ext(t=t)
        return power_ext / self.volume / self.e

    def get_el_en_losses(self, y: ndarray, temp_e: float64 = None) -> ndarray:
        """Calculates the vector of electron energy losses per reaction for all the
        reactions.

        Only electron collisions (elastic and inelastic, with at least one electron on
        the LHS) will have non-zero values and will be used downstream.

        Parameters
        ----------
        y : ndarray
        temp_e : float64, optional
            Electron temperature in [eV].

        Returns
        -------
        ndarray
            Vector of electron energy losses [eV] per a single collision for each
            reaction. Only electron collisions will have non-zero values.
        """
        if temp_e is None:
            temp_e = self.get_electron_temperature(y)
        mask = self.mask_r_electron & self.mask_r_elastic
        self.r_el_energy_losses[mask] = (
            3
            * self.m_e
            / self.r_col_partner_masses[mask]
            * (temp_e - self.temp_n * self.k / self.e)
        )
        return self.r_el_energy_losses

    def get_drho_over_dt_el_inel(
        self, y: ndarray, el_en_losses: ndarray = None, reaction_rates: ndarray = None
    ) -> float64:
        """Calculates the contribution of elastic and inelastic electron collisions to
        the time derivative of electron energy density.

        Parameters
        ----------
        y : ndarray
        el_en_losses : ndarray, optional
            Electron energy losses [eV] per collision for each reaction in the set.
        reaction_rates : ndarray, optional
            Reaction rates [m-3/s] for each reaction in the set.

        Returns
        -------
        float64
            A contribution to the time derivative of electron energy density [eV.m-3/s]
            due to the inelastic and elastic electron collisions.
        """
        if el_en_losses is None:
            el_en_losses = self.get_el_en_losses(y)
        if reaction_rates is None:
            reaction_rates = self.get_reaction_rates(y)
        return (
            el_en_losses[self.mask_r_electron] * reaction_rates[self.mask_r_electron]
        ).sum()

    def get_drho_over_dt_gain_loss(
        self, y: ndarray, temp_e: float64 = None, reaction_rates: ndarray = None
    ) -> float64:
        """Calculates the contribution of volumetric creation and destruction of
        electrons to the time derivative of electron energy density.

        Parameters
        ----------
        y : ndarray
        temp_e : float64, optional
            Electron temperature in [eV].
        reaction_rates : ndarray, optional
            Reaction rates [m-3/s] for each reaction in the set.

        Returns
        -------
        float64
            A contribution to the time derivative of electron energy density [eV.m-3/s]
            due to the creation and destruction of electrons in volumetric processes.
        """
        if temp_e is None:
            temp_e = self.get_electron_temperature(y)
        if reaction_rates is None:
            reaction_rates = self.get_reaction_rates(y)
        return 3 / 2 * temp_e * (self.r_stoich_electron_net * reaction_rates).sum()

    def get_drho_over_dt_el_walls(
        self, y: ndarray, temp_e: float64 = None, diff_source_rates: ndarray = None
    ) -> float64:
        """Calculates the contribution of the electrons losses to the walls to the time
        derivative of electron energy density.

        Parameters
        ----------
        y : ndarray
        temp_e : float64, optional
            Electron temperature [eV].
        diff_source_rates : ndarray, optional
            The vector of contributions to time derivatives of densities due to the
            diffusion sinks and wall-return sources in [m-3/s].
        Returns
        -------
        float64
            A contribution to the time derivative of electron energy density [eV.m-3/s]
            due to the electrons losses to the walls.
        """
        if temp_e is None:
            temp_e = self.get_electron_temperature(y)
        if diff_source_rates is None:
            diff_source_rates = self.get_diffusion_source_rates(y)
        tot_el_outflux = np.sum(self.sp_charges * diff_source_rates)
        return -2 * temp_e * tot_el_outflux

    def get_sheath_voltage(self, y: ndarray, temp_e: float64 = None) -> float64:
        """Estimates the sheath voltage of the modeled plasma.

        Parameters
        ---------
        y : ndarray
        temp_e : float64, optional
            Electron temperature [eV].

        Returns
        -------
        float64
            Sheath voltage [V] estimated for an ICP plasma based on the T_e.
        """
        if temp_e is None:
            temp_e = self.get_electron_temperature(y)
        return temp_e * self.sheath_voltage_per_ev

    def get_drho_over_dt_ions_walls(
        self,
        y: ndarray,
        temp_e: float64 = None,
        sh_pot: float64 = None,
        diff_source_rates: ndarray = None,
    ) -> float64:
        """Calculates the contribution of the loss of ions to the walls to the time
        derivative of electron energy density.

        Parameters
        ----------
        y : ndarray
        temp_e : float64, optional
            Electron temperature [eV].
        sh_pot : float64, optional
            Sheath voltage [V].
        diff_source_rates : ndarray, optional
            The vector of contributions to time derivatives of densities due to the
            diffusion sinks and wall-return sources in [m-3/s].

        Returns
        -------
        float64
            A contribution to the time derivative of electron energy density [eV.m-3/s]
            due to the loss of ions to the walls.
        """
        if temp_e is None:
            temp_e = self.get_electron_temperature(y)
        if sh_pot is None:
            sh_pot = self.get_sheath_voltage(y)
        if diff_source_rates is None:
            diff_source_rates = self.get_diffusion_source_rates(y)
        mask = self.mask_sp_positive
        return -0.5 * temp_e * np.sum(diff_source_rates[mask]) - sh_pot * np.sum(
            self.sp_charges[mask] * diff_source_rates[mask]
        )

    def get_min_rho_correction(self, y: ndarray, rho: float64 = None) -> float64:
        """This is an artificial (unphysical) correction applied to the RHS of the
        electron energy density ODE.

        It is preventing `rho` getting under a minimal value (and ultimately from
        reaching unphysical negative values.) It supplies a nudge if rho is below a
        lower limit. The nudge is proportional to the difference of the rho and the
        (hard-coded) limit.

        Parameters
        ----------
        y : ndarray
        rho : float64, optional
            Electron energy density [eV.m-3]
        Returns
        -------
        float64
            Correction in [eV.m-3/s] preventing the electron energy density from
            reaching values lower than (hard-coded) `rho_min`.
        """
        if rho is None:
            rho = self.get_electron_energy_density(y)
        rho_min = 1.0e0
        t_rec = 1.0e-10  # recovery time scale - approx solver step time
        min_rho_correction = (rho_min - rho) / t_rec if rho < rho_min else 0.0
        return float64(min_rho_correction)

    def get_drho_over_dt(
        self,
        t: float64,
        y: ndarray,
        ext: float64 = None,
        el_inel: float64 = None,
        gain_loss: float64 = None,
        el_walls: float64 = None,
        ions_walls: float64 = None,
        min_rho_correction: float64 = None,
    ) -> float64:
        """Calculate the total time derivative of the electron energy density.

        Parameters
        ----------
        t : float64
            Time [s].
        y : ndarray
            The state vector y (heavy species densities in [m-3] and electron energy
            density in [eV.m-3/s])
        ext : float64, optional
            The contribution to the time derivative of electron energy density due to
            the absorbed external power in [eV.m-3/s].
        el_inel : float64, optional
            A contribution to the time derivative of electron energy density [eV.m-3/s]
            due to the inelastic and elastic electron collisions.
        gain_loss : float64, optional
            A contribution to the time derivative of electron energy density [eV.m-3/s]
            due to the creation and destruction of electrons in volumetric processes.
        el_walls : float64, optional
            A contribution to the time derivative of electron energy density [eV.m-3/s]
            due to the electrons losses to the walls.
        ions_walls : float64, optional
            A contribution to the time derivative of electron energy density [eV.m-3/s]
            due to the loss of ions to the walls.
        min_rho_correction : float64
            Correction in [eV.m-3/s] preventing the electron energy density from
            reaching values lower than (hard-coded) `rho_min`.

        Returns
        -------
        float64
            The time derivative of the electron energy density [eV.m-3/s] as the sum of
            all the contributions.
        """
        if ext is None:
            ext = self.get_drho_over_dt_ext(t)
        if el_inel is None:
            el_inel = self.get_drho_over_dt_el_inel(y)
        if gain_loss is None:
            gain_loss = self.get_drho_over_dt_gain_loss(y)
        if el_walls is None:
            el_walls = self.get_drho_over_dt_el_walls(y)
        if ions_walls is None:
            ions_walls = self.get_drho_over_dt_ions_walls(y)
        if min_rho_correction is None:
            min_rho_correction = self.get_min_rho_correction(y)

        # noinspection PyTypeChecker
        return ext - el_inel - gain_loss - el_walls - ions_walls + min_rho_correction

    def get_dy_over_dt(
        self,
        t: float64,
        y: ndarray,
        dn_over_dt: ndarray = None,
        drho_over_dt: float64 = None,
    ) -> ndarray:
        """Calculate the vector of right-hand-sides of the system of ODE which solved
        for.

        This is the time derivative of the state vector y.

        Parameters
        ----------
        t : float64
            Time [s].
        y : ndarray
            State vector of heavy species densities and the electron energy density
            [n_1, ..., n_N, rho] in [m-3, ..., m-3, eV.m-3].
        dn_over_dt : ndarray, optional
            Vector of time derivatives of densities [m-3/s] of all the heavy species.
        drho_over_dt : float64, optional
            The time derivative of the electron energy density [eV.m-3/s] as the sum of
            all the contributions.

        Returns
        -------
        ndarray
            Vector of the time derivatives of the state vector y in
            [m-3/s, ..., m-3/s, eV.m-3/s].
        """
        if dn_over_dt is None:
            dn_over_dt = self.get_dn_over_dt(y)
        if drho_over_dt is None:
            drho_over_dt = self.get_drho_over_dt(t, y)
        return np.r_[dn_over_dt, drho_over_dt]

    @property
    def ode_system_rhs(self) -> Callable[[float64, ndarray], ndarray]:
        """A function for the right-hand-side of an ODE system solving for the
        *state vector y*.

        The function returned by this property accepts time [s] and the state vector
        y = [n_1, ..., n_N, rho] in [m-3, ..., m-3, eV.m-3].

        Returns
        -------
        Callable(float64, ndarray) -> ndarray
        """

        def func(t: float64, y: ndarray) -> ndarray:
            """The right-hand-side of an ODE system solving for the *state vector y*.

            Parameters
            ----------
            t : float64
                Time [s].
            y : ndarray
                State vector of heavy species densities and the electron energy density
                [n_1, ..., n_N, rho] in [m-3, ..., m-3, eV.m-3].

            Returns
            -------
            ndarray
                The right-hand-side of an ODE system solving for the *state vector y*.
            """
            n = self.get_density_vector(y)
            rho = self.get_electron_energy_density(y)
            n_tot = self.get_total_density(y, n=n)
            p = self.get_total_pressure(y, n_tot=n_tot)
            temp_i = self.get_ion_temperature(y, p=p)
            n_e = self.get_electron_density(y, n=n)
            temp_e = self.get_electron_temperature(y, n_e=n_e, rho=rho)
            debye_length = self.get_debye_length(y, n_e=n_e, temp_e=temp_e)
            rate_coefs = self.get_reaction_rate_coefficients(y, temp_e=temp_e)
            rates = self.get_reaction_rates(
                y, n=n, n_e=n_e, n_tot=n_tot, k_r=rate_coefs
            )
            source_vol = self.get_volumetric_source_rates(y, rates=rates)
            source_flow = self.get_flow_source_rates(y, n=n, p=p)
            v_m = self.get_mean_speeds(y, temp_i=temp_i)
            sigma_sc = self.get_sigma_sc(y, v_m=v_m, debye_length=debye_length)
            mfp = self.get_mean_free_paths(y, n=n, sigma_sc=sigma_sc)
            diff_c_free = self.get_free_diffusivities(y, mfp=mfp, v_m=v_m)
            diff_a_pos = self.get_ambipolar_diffusivity_pos(
                y, n=n, n_e=n_e, temp_i=temp_i, temp_e=temp_e, diff_c_free=diff_c_free
            )
            diff = self.get_diffusivities(
                y, diff_c_free=diff_c_free, diff_a_pos=diff_a_pos
            )
            wall_fluxes = self.get_wall_fluxes(y, n=n, diff_c=diff, v_m=v_m)
            surf_loss_rates = self.get_surface_loss_rates(y, wall_fluxes=wall_fluxes)
            surf_source_rates = self.get_surface_source_rates(
                y, surf_loss_rates=surf_loss_rates
            )
            source_diff = self.get_diffusion_source_rates(
                y, surf_loss_rates=surf_loss_rates, surf_source_rates=surf_source_rates
            )
            min_n_cor = self.get_min_n_correction(y, n=n)
            dn_over_dt = self.get_dn_over_dt(
                y,
                vol_source_rates=source_vol,
                flow_source_rates=source_flow,
                diff_source_rates=source_diff,
                min_n_correction=min_n_cor,
            )
            power_ext = self.get_power_ext(t)
            drho_over_dt_ext = self.get_drho_over_dt_ext(t, power_ext=power_ext)
            el_en_losses = self.get_el_en_losses(y, temp_e=temp_e)
            drho_over_dt_el_inel = self.get_drho_over_dt_el_inel(
                y, el_en_losses=el_en_losses, reaction_rates=rates
            )
            drho_over_dt_gain_loss = self.get_drho_over_dt_gain_loss(
                y, temp_e=temp_e, reaction_rates=rates
            )
            drho_over_dt_el_walls = self.get_drho_over_dt_el_walls(
                y, temp_e=temp_e, diff_source_rates=source_diff
            )
            sh_pot = self.get_sheath_voltage(y, temp_e=temp_e)
            drho_over_dt_ions_walls = self.get_drho_over_dt_ions_walls(
                y, temp_e=temp_e, sh_pot=sh_pot, diff_source_rates=source_diff
            )
            min_rho_correction = self.get_min_rho_correction(y, rho=rho)
            drho_over_dt = self.get_drho_over_dt(
                t,
                y,
                ext=drho_over_dt_ext,
                el_inel=drho_over_dt_el_inel,
                gain_loss=drho_over_dt_gain_loss,
                el_walls=drho_over_dt_el_walls,
                ions_walls=drho_over_dt_ions_walls,
                min_rho_correction=min_rho_correction,
            )
            dy_over_dt = self.get_dy_over_dt(
                t, y, dn_over_dt=dn_over_dt, drho_over_dt=drho_over_dt
            )
            return dy_over_dt

        return func

    @property
    def final_solution_labels(self) -> List[str]:
        """The string labels for the final solution built downstream by the global
        model.

        Returns
        -------
        list of str
            Labels of the final solution, e.g. `["Ar", "Ar+", "e", "T_e"]`
        """
        return list(self.chemistry.species_ids) + ["e", "T_e", "T_n", "p", "P"]

    def get_final_solution_values(self, t: float64, y: ndarray) -> ndarray:
        """Turns the raw state vector y into the final values consistent with
        the `final_solution_labels` above.

        Parameters
        ----------
        t: float64
            Time sample in [s].
        y : ndarray
            State vector *y*.

        Returns
        -------
        ndarray
            Values of the final solution for the given state vector.
        """
        n = self.get_density_vector(y)
        n_e = self.get_electron_density(y, n=n)
        temp_e = self.get_electron_temperature(y, n_e=n_e)
        temp_n = self.plasma_params.temp_n
        p = self.get_total_pressure(y)
        power = self.get_power_ext(t)
        return np.r_[n, n_e, temp_e, temp_n, p, power]

    def get_y0_default(
        self,
        initial_densities: Dict[str, float] = None,
        ionization_degree: float = 1.0e-15,
        negative_ion_fraction: float = 1.0e-15,
        non_feed_species_fraction: float = 1.0e-15,
    ) -> ndarray:
        """Method building an initial guess for the state vector `y` self-consistent
        and consistent with the chemistry and plasma parameters.

        Parameters
        ----------
        initial_densities : dict[str, float], optional
            Mapping between heavy species ids (must exist in the chemistry) and their
            initial densities fractions (will be re-normalized to the pressure).
            Species which are not present in `initial_densities` will be initialized
            with n = 0.
            If not given, the initial densities will be defaulted using the next three
            arguments.
        ionization_degree : float, optional
            This determines the electron density as a fraction of the total density.
            Only if `initial_densities` not given.
        negative_ion_fraction : float, optional
            This determines the densities of negative ions as fraction of total density.
            Only if `initial_densities` not given.
        non_feed_species_fraction : float, optional
            This determines the densities of non-feed neutrals as fractions of total
            density. Only if `initial_densities` not given.
        Returns
        -------
        ndarray
            Self-consistent initial guess for the state vector `y`.
        """
        # initial total density based on the pressure and temperature:
        n_tot = self.pressure / (self.k * self.temp_n)

        if initial_densities is not None:
            n0 = np.array(
                [
                    initial_densities.get(sp_id, 0.0)
                    for sp_id in self.chemistry.species_ids
                ]
            )
            n0 = n0 / sum(n0) * n_tot
            n_e = sum(n0 * self.sp_charges)
        else:
            mask_neg = self.mask_sp_negative
            mask_pos = self.mask_sp_positive
            mask_neu = self.mask_sp_neutral

            mask_flows = self.sp_flows > 0
            # initial electron density:
            n_e = n_tot * ionization_degree
            # initial density of negative ions:
            n_neg_ions_tot = (
                n_tot * -sum(self.sp_charges[mask_neg]) * negative_ion_fraction
            )
            # initial density of all negative species:
            n_neg_tot = n_neg_ions_tot + n_e
            # initial density of positive species
            n_pos_tot = n_neg_tot

            # initial density vector
            n0 = np.zeros(self.num_species)
            n0[mask_pos] = n_pos_tot / sum(self.sp_charges[mask_pos])
            if mask_neg.any():
                n0[mask_neg] = -n_neg_ions_tot / sum(self.sp_charges[mask_neg])
            # non-feed neutrals:
            n0[mask_neu & ~mask_flows] = n_tot * non_feed_species_fraction
            # feed species divide the remaining density in proportion to feed flows
            if self.sp_flows.sum():
                n0[mask_flows] = (
                    self.sp_flows[mask_flows] / self.sp_flows.sum() * (n_tot - n0.sum())
                )
            else:
                # if no flows defined, distribute to the total flow among neutrals
                n_residual = n_tot - n0[~mask_neu].sum()
                n0[mask_neu] = n_residual / len(n0[mask_neu])

        return np.r_[n0, 3 / 2 * n_e * self.plasma_params.temp_e]
