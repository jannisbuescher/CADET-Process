"""Store and plot solution of simulation.

NCOL/N_Z: Number of axial cells
z: Axial position

NRAD/N_rho: Number of
rho: Annulus cell (cylinder shell)

NCOMP/N_C: Number of components
i: component index

NPARTYPE/N_P: Number of particle types
j: Particle type index

NPAR/N_R: Number of particle cells
r: Radial position (particle)

IO: NCOL * NRAD
Bulk/Interstitial: NCOL * NRAD * NCOMP
Particle_liquid: NCOL * NRAD * sum_{j}^{NPARTYPE}{NCOMP * NPAR,j}
Particle_solid: NCOL * NRAD * sum_{j}^{NPARTYPE}{NBOUND,j * NPAR,j}
Flux: NCOL * NRAD * NPARTYPE * NCOMP
"""

import numpy as np

from CADETProcess import CADETProcessError
from CADETProcess.dataStructure import StructMeta
from CADETProcess.dataStructure import UnsignedInteger, Vector, DependentlySizedNdArray
from CADETProcess.processModel import ComponentSystem
from CADETProcess import plotting

class BaseSolution(metaclass=StructMeta):
    time = Vector()
    solution = DependentlySizedNdArray(dep='solution_shape')
    
    _coordinates = []
    
    @property
    def component_system(self):
        return self._component_system

    @component_system.setter
    def component_system(self, component_system):
        if not isinstance(component_system, ComponentSystem):
            raise TypeError('Expected ComponentSystem')
        self._component_system = component_system
    
    @property
    def n_comp(self):
        return self.component_system.n_comp

    @property
    def nt(self):
        return len(self.time)
    
    @property
    def coordinates(self):
        coordinates = {}
        
        for c in self._coordinates:
            v = getattr(self, c)
            if v is None:
                continue
            coordinates[c] = v
        return coordinates

    @property
    def solution_shape(self):
        """tuple: (Expected) shape of the solution
        """
        coordinates = tuple(len(c) for c in self.coordinates.values())
        return (self.nt,) + coordinates + (self.n_comp,)
    
    
    def total_concentration(self):
        total_concentration = np.zeros(
            self.solution_shape[0:-1] + (self.component_system.n_components,)
        )
        
        counter = 0
        for index, comp in enumerate(self.component_system.components):
            comp_indices = slice(counter, comp.n_species)
            total_concentration[:,index] = np.sum(self.solution[...,comp_indices], axis=1)
        
        return total_concentration
    
    def transform_solution(self, fun, comp=None):
        if comp is None:
            self.solution = fun(self.solution)
        else:
            self.solution[...,comp] = fun(self.solution[...,comp])
        
            
class SolutionIO(BaseSolution):
    """Solution at unit inlet or outlet.
    
    IO: NCOL * NRAD
    """
    def __init__(self, component_system, time, solution):
        self.component_system = component_system
        self.time = time
        self.solution = solution
        
    @plotting.save_fig
    def plot(self, start=0, end=None, overlay=None):
        """Plots the whole time_signal for each component.

        Parameters
        ----------
        start : float
            start time for plotting
        end : float
            end time for plotting

        See also
        --------
        plotlib
        plot_purity
        """
        x = self.time / 60
        y = self.solution
        
        ymax = y.max()
        
        fig, ax = plotting.setup_figure()
        
        ax.plot(x,y)

        if overlay is not None:
            ymax = np.max(overlay)
            plotting.add_overlay(ax, overlay)
            
        layout = plotting.Layout()
        layout.x_label = '$time~/~min$'
        layout.y_label = '$c~/~mM$'
        layout.xlim = (start, end)
        layout.ylim = (0, ymax)
            
        plotting.set_layout(fig, ax, layout)        
        
        return ax
    
        
class SolutionBulk(BaseSolution):
    """Interstitial solution.
    
    Bulk/Interstitial: NCOL * NRAD * NCOMP
    """
    _coordinates = ['axial_coordinates', 'radial_coordinates']

    def __init__(
            self,
            component_system,
            time, solution, 
            axial_coordinates=None, radial_coordinates=None
            ):
        self.component_system = component_system
        self.time = time
        self.axial_coordinates = axial_coordinates
        # Account for dimension reduction in case of only one cell (e.g. LRMP)
        if radial_coordinates is not None and len(radial_coordinates) == 1:
            radial_coordinates = None
        self.radial_coordinates = radial_coordinates
        self.solution = solution

    @property
    def ncol(self):
        if self.axial_coordinates is None:
            return
        return len(self.axial_coordinates)

    @property
    def nrad(self):
        if self.radial_coordinates is None:
            return
        return len(self.radial_coordinates)
    
    @plotting.save_fig
    def plot_at_time(self, t, overlay=None, ymax=None):
        """Plot bulk solution over space at given time.

        Parameters
        ----------
        t : float
            time for plotting

        See also
        --------
        plot_at_location
        CADETProcess.plotting
        """
        x = self.axial_coordinates
        
        if not self.time[0] <= t <= self.time[-1]:
            raise ValueError("Time exceeds bounds.")
        t_i = np.where(t<=self.time)[0][0]    
        
        y = self.solution[t_i,:]
        if ymax is None:
            ymax = 1.1*np.max(y)

        fig, ax = plotting.setup_figure()
        ax.plot(x,y)
        
        plotting.add_text(ax, f'time = {t:.2f} s')
        
        if overlay is not None:
            ymax = np.max(overlay)
            plotting.add_overlay(ax, overlay)
            
        layout = plotting.Layout()
        layout.x_label = '$z~/~m$'
        layout.y_label = '$c~/~mM$'
        layout.ylim = (0, ymax)
        plotting.set_layout(fig, ax, layout)        
        
        return ax
    
    @plotting.save_fig        
    def plot_at_location(self, z, overlay=None, ymax=None):
        """Plot bulk solution over time at given location.

        Parameters
        ----------
        z : float
            space for plotting

        See also
        --------
        plot_at_time
        CADETProcess.plotting
        """
        x = self.time
        
        if not self.axial_coordinates[0] <= z <= self.axial_coordinates[-1]:
            raise ValueError("Axial coordinate exceets boundaries.")
        z_i = np.where(z<=self.axial_coordinates)[0][0]
        
        y = self.solution[:,z_i]
        if ymax is None:
            ymax = 1.1*np.max(y)

        fig, ax = plotting.setup_figure()
        ax.plot(x,y)
        
        plotting.add_text(ax, f'z = {z:.2f} m')

        if overlay is not None:
            ymax = np.max(overlay)
            plotting.add_overlay(ax, overlay)
            
        layout = plotting.Layout()
        layout.x_label = '$time~/~min$'
        layout.y_label = '$c~/~mM$'
        layout.ylim = (0, ymax)
        plotting.set_layout(fig, ax, layout)        
        
        return ax
    
    
class SolutionParticle(BaseSolution):
    """Mobile phase solution inside the particles.
    
    Particle_liquid: NCOL * NRAD * sum_{j}^{NPARTYPE}{NCOMP * NPAR,j}
    """
    
    _coordinates = ['axial_coordinates', 'radial_coordinates', 'particle_coordinates']

    def __init__(
            self,
            component_system,
            time, solution, 
            axial_coordinates=None,
            radial_coordinates=None,
            particle_coordinates=None
            ):
        self.component_system = component_system
        self.time = time
        self.axial_coordinates = axial_coordinates
        # Account for dimension reduction in case of only one cell (e.g. LRMP)
        if radial_coordinates is not None and len(radial_coordinates) == 1:
            radial_coordinates = None
        self.radial_coordinates = radial_coordinates
        # Account for dimension reduction in case of only one cell (e.g. CSTR)
        if particle_coordinates is not None and len(particle_coordinates) == 1:
            particle_coordinates = None
        self.particle_coordinates = particle_coordinates
        self.solution = solution

    @property
    def ncol(self):
        if self.axial_coordinates is None:
            return
        return len(self.axial_coordinates)
        
    @property
    def nrad(self):
        if self.radial_coordinates is None:
            return
        return len(self.radial_coordinates)

    @property
    def npar(self):
        if self.particle_coordinates is None:
            return 
        return len(self.particle_coordinates)
    
    def _plot_1D(self, t, ymax):
        x = self.axial_coordinates
        
        if not self.time[0] <= t <= self.time[-1]:
            raise ValueError("Time exceeds bounds.")
        t_i = np.where(t<=self.time)[0][0]    
        y = self.solution[t_i,:]

        if ymax is None:
            ymax = 1.1*np.max(y)

        fig, ax = plotting.setup_figure()
        ax.plot(x,y)
        
        plotting.add_text(ax, f'time = {t:.2f} s')
        
        layout = plotting.Layout()
        layout.x_label = '$z~/~m$'
        layout.y_label = '$c~/~mM$'
        layout.ylim = (0, ymax)
        plotting.set_layout(fig, ax, layout) 
        
        return fig, ax
        
    def _plot_2D(self, t, comp, vmax):
        x = self.axial_coordinates
        y = self.particle_coordinates
        
        if not self.time[0] <= t <= self.time[-1]:
            raise ValueError("Time exceeds bounds.")
        i = np.where(t<=self.time)[0][0]
        v = self.solution[i,:,:,comp].transpose()
        
        if vmax is None:
            vmax = v.max()
            
        fig, ax = plotting.setup_figure()
        mesh = ax.pcolormesh(x, y, v, shading='gouraud', vmin=0, vmax=vmax)
        
        plotting.add_text(ax, f'time = {t:.2f} s')
        
        layout = plotting.Layout()
        layout.x_label = '$z~/~m$'
        layout.y_label = '$r~/~m$'
        layout.title = f'Solid phase concentration, comp={comp}, state={state}'
        plotting.set_layout(fig, ax, layout)        
        fig.colorbar(mesh)
        
        return fig, ax, mesh
    
    @plotting.save_fig
    def plot_at_time(self, t, comp=0, vmax=None):
        """Plot particle liquid solution over space at given time.

        Parameters
        ----------
        t : float
            time for plotting
        comp : int
            component index

        See also
        --------
        CADETProcess.plotting
        """
        if self.npar is None:
            fig, ax = self._plot_1D(t, vmax)
        else:
            fig, ax = self._plot_2D(t, comp, vmax)
        return ax
    
class SolutionSolid(BaseSolution):
    """Solid phase solution inside the particles.
    
    Particle_solid: NCOL * NRAD * sum_{j}^{NPARTYPE}{NBOUND,j * NPAR,j}
    """
    
    n_bound = UnsignedInteger()

    _coordinates = ['axial_coordinates', 'radial_coordinates', 'particle_coordinates']

    def __init__(
            self,
            component_system, n_bound,
            time, solution, 
            axial_coordinates=None,
            radial_coordinates=None,
            particle_coordinates=None
            ):
        self.component_system = component_system
        self.n_bound = n_bound
        self.time = time
        self.axial_coordinates = axial_coordinates
        # Account for dimension reduction in case of only one cell (e.g. LRMP)
        if radial_coordinates is not None and len(radial_coordinates) == 1:
            radial_coordinates = None
        self.radial_coordinates = radial_coordinates
        # Account for dimension reduction in case of only one cell (e.g. CSTR)
        if particle_coordinates is not None and len(particle_coordinates) == 1:
            particle_coordinates = None
        self.particle_coordinates = particle_coordinates
        self.solution = solution
        
    @property
    def n_comp(self):
        return self.component_system.n_comp * self.n_bound
    
    @property
    def ncol(self):
        if self.axial_coordinates is None:
            return None
        else:
            return len(self.axial_coordinates)
        
    @property
    def nrad(self):
        if self.radial_coordinates is None:
            return
        return len(self.radial_coordinates)

    @property
    def npar(self):
        if self.particle_coordinates is None:
            return 
        return len(self.particle_coordinates)
    
    
    def _plot_1D(self, t, ymax):
        x = self.axial_coordinates

        if not self.time[0] <= t <= self.time[-1]:
            raise ValueError("Time exceeds bounds.")
        t_i = np.where(t<=self.time)[0][0]
        y = self.solution[t_i,:]

        if ymax is None:
            ymax = 1.1*np.max(y)

        fig, ax = plotting.setup_figure()
        ax.plot(x,y)
        
        plotting.add_text(ax, f'time = {t:.2f} s')
        
        layout = plotting.Layout()
        layout.x_label = '$z~/~m$'
        layout.y_label = '$c~/~mM$'
        layout.labels = self.component_system.labels
        layout.ylim = (0, ymax)
        plotting.set_layout(fig, ax, layout) 
        
        return fig, ax
        
    def _plot_2D(self, t, comp, state, vmax):
        x = self.axial_coordinates
        y = self.particle_coordinates
        
        if not self.time[0] <= t <= self.time[-1]:
            raise ValueError("Time exceeds bounds.")
        t_i = np.where(t<=self.time)[0][0]
        c_i = comp*self.n_bound + state
        v = self.solution[t_i,:,:,c_i].transpose()
        
        if vmax is None:
            vmax = v.max()
            
        fig, ax = plotting.setup_figure()
        mesh = ax.pcolormesh(x, y, v, shading='gouraud', vmin=0, vmax=vmax)
        
        plotting.add_text(ax, f'time = {t:.2f} s')
        
        layout = plotting.Layout()
        layout.title = f'Solid phase concentration, comp={comp}, state={state}'
        layout.x_label = '$z~/~m$'
        layout.y_label = '$r~/~m$'
        layout.labels = self.component_system.labels[c_i]
        plotting.set_layout(fig, ax, layout)        
        fig.colorbar(mesh)
        
        return fig, ax, mesh
    
    @plotting.save_fig
    def plot_at_time(self, t, comp=0, state=0, vmax=None):
        """Plot bulk solution over spce at given time.

        Parameters
        ----------
        t : float
            time for plotting
        comp : int, optional
            component index
        state : int, optional
            bound state
        vmax : float, optional
            Maximum values for plotting.

        See also
        --------
        CADETProcess.plotting
        """
        if self.npar is None:
            fig, ax = self._plot_1D(t, vmax)
        else:
            fig, ax, mesh = self._plot_2D(t, comp, state, vmax)
        return ax


class SolutionVolume(BaseSolution):
    """Volume solution (of e.g. CSTR).
    """
    def __init__(self, component_system, time, solution):
        self.time = time
        self.solution = solution
        
    @property
    def solution_shape(self):
        """tuple: (Expected) shape of the solution
        """
        return (self.nt, 1)

    @plotting.save_fig
    def plot(self, start=0, end=None, overlay=None):
        """Plots the whole time_signal for each component.

        Parameters
        ----------
        start : float
            start time for plotting
        end : float
            end time for plotting

        See also
        --------
        CADETProcess.plot
        """
        x = self.time / 60
        y = self.solution / 1000
        ymax = np.max(y)

        fig, ax = plotting.setup_figure()
        
        
        ax.plot(x,y)

        if overlay is not None:
            ymax = 1.1*np.max(overlay)
            plotting.add_overlay(ax, overlay)
            
        layout = plotting.Layout()
        layout.x_label = '$time~/~min$'
        layout.y_label = '$V~/~L$'
        layout.xlim = (start, end)
        layout.ylim = (0, ymax)
        plotting.set_layout(fig, ax, layout)        
        
        return ax
    