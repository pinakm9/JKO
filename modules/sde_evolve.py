import numpy as np
import tables
import os
import matplotlib.pyplot as plt
import cv2
import shutil, copy

class SDE:
    """
    Description:
        Implementation of a an SDE of the form dX_t = mu(t, X_t)dt + sigma(t, X_t)dB_t
    Args:
        space_dim: space dimension of the SDE
        mu: a function of time and space mu(t, X_t)
        sigma: a function of time and space sigma(t, X_t)
        record_path: file path ending with .h5 describing where to record the ensemble evolution 
    """
    def __init__(self, space_dim, mu, sigma, record_path, dtype=np.float64):
        self.space_dim = space_dim
        self.mu = mu
        self.sigma = sigma
        self.record_path = record_path

    def evolve(self, initial_ensembles, initial_probs, initial_first_partials, final_time, time_step):
        """
        Description:
            evolves an initial ensemble according to the SDE dynamics
        Args:
            initial_ensemble: the ensemble that starts the evolution
            initial_probs: probabilities for the members of the initial ensemble
            initial_first_partials: first partial derivatives for the initial ensemble
            final_time: final time in the evolution assuming we're starting at time=0
            time_step: time_step in Euler-Maruyma method
        """
        self.num_particles = len(initial_ensembles[0]) 
        num_steps = int(final_time / time_step)
        noise_std = np.sqrt(time_step)
        hdf5 = tables.open_file(self.record_path, 'w')
        for j, initial_ensemble in enumerate(initial_ensembles):
            run_folder = hdf5.create_group(hdf5.root, 'run_{}'.format(j))
            ens_folder = hdf5.create_group(run_folder, 'ensemble')
            new_ensemble = initial_ensemble
            # record the initial ensemble
            hdf5.create_array(ens_folder, 'time_0', initial_ensemble)
            # evolve ensemble with Euler-Maruyama
            for step in range(num_steps):
            # evolve ensemble with Euler-Maruyama
                for i in range(self.num_particles):
                    noise = np.random.normal(loc=0.0, scale=noise_std, size=self.space_dim)
                    new_ensemble[i] += self.mu(step*time_step, new_ensemble[i])*time_step + self.sigma(step*time_step, new_ensemble[i]) * noise
                # record the new ensemble
                hdf5.create_array(ens_folder, 'time_' + str(step + 1), new_ensemble)
        # add some extra useful information to the evolution file
        class Config(tables.IsDescription):
            num_steps = tables.Int32Col(pos=0)
            time_step = tables.Float32Col(pos=1)
            ensemble_size = tables.Int32Col(pos=2)
            dimension = tables.Int32Col(pos=3)
            num_runs = tables.Int32Col(pos=4)
        tbl = hdf5.create_table(hdf5.root, 'config', Config)
        config = tbl.row
        config['num_steps'] = num_steps
        config['time_step'] = time_step
        config['ensemble_size'] = len(new_ensemble)
        config['dimension'] = new_ensemble.shape[-1]
        config['num_runs'] = len(initial_ensembles)
        config.append()
        tbl.flush()
        hdf5.close()

    def extend(self, final_time):
        hdf5 = tables.open_file(self.record_path, 'r+')
        self.final_available_time_id, self.time_step, self.num_particles, _ = hdf5.root.config.read()[0]
        #self.time_step = float(self.time_step)
        num_steps = int((final_time - int(self.final_available_time_id) * self.time_step)/ self.time_step) + 1
        noise_std = np.sqrt(self.time_step)
        new_ensemble = getattr(hdf5.root.ensemble, 'time_' + str(self.final_available_time_id)).read()
        for step in range(num_steps):
            print('working on step #{} ...'.format(step))
            # evolve ensemble with Euler-Maruyama
            noise = np.random.normal(loc=0.0, scale=noise_std, size=self.space_dim)
            for i in range(self.num_particles):
                new_ensemble[i] += self.mu(step*self.time_step, new_ensemble[i])*self.time_step + self.sigma(step*self.time_step, new_ensemble[i]) * noise
            # record the new ensemble
            hdf5.create_array(hdf5.root.ensemble, 'time_' + str(self.final_available_time_id + step + 1), new_ensemble)
        hdf5.close()



class SDEPlotter:
    """
    Description:
        plots the ensemble evolution of an SDE
    Args:
        ens_file: path to .h5 file containing the ensemble evolution
        fig_size: size of plots in inch x inch as a tuple (width, height), default=(10, 10) 
    """
    def __init__(self, ens_file, fig_size=(10, 10), ax_lims=[None, None, None]):
        self.ens_file = ens_file
        self.fig_size = fig_size
        hdf5 = tables.open_file(ens_file, 'r')
        self.num_frames, self.time_step, _, self.dim, _ = hdf5.root.config.read()[0]
        self.num_frames += 1
        if self.dim == 2 or self.dim == 3:
            # create folder to store images
            try:
                self.frames_folder = os.path.dirname(ens_file) + '/{}'.format(os.path.basename(ens_file).split('.')[0]) + '_frames'
                os.mkdir(self.frames_folder)
            except:
                pass
            # animate
            self.animate(hdf5, ax_lims)
        else:
            print('Particles in the ensemble must be of either dimension 2 or 3!!!')
        

    def animate(self, hdf5, ax_lims):
        """
        Description:
            creates an animation of the ensemble evolution
        Args:
            hdf5: handle to the .h5 file containing the ensemble evolution
        """
        fig = plt.figure(figsize=self.fig_size)
        ax = fig.add_subplot(111) if self.dim == 2 else fig.add_subplot(111, projection='3d')

        def update_plot(frame):
            ax.clear()
            # read data to plot
            ensemble = getattr(hdf5.root.run_0.ensemble, 'time_' + str(frame)).read()
            if self.dim == 2:
                ax.scatter(ensemble[:, 0], ensemble[:, 1])
            else:
                ax.scatter(ensemble[:, 0], ensemble[:, 1], ensemble[:, 2], c='b', s=10.0)
            ax.set_title('time = {:.2f}'.format(frame * self.time_step))
            ax.set_xlabel('x')
            if ax_lims[0] is not None:
                ax.set_xlim(ax_lims[0])
            ax.set_ylabel('y')
            if ax_lims[1] is not None:
                ax.set_xlim(ax_lims[1])
            if self.dim == 3:
                ax.set_zlabel('z')
                if ax_lims[2] is not None:
                    ax.set_xlim(ax_lims[2])
            plt.savefig(self.frames_folder + '/frame_{}.png'.format(frame))
            print('Frame {} has been drawn.'.format(frame))

        for frame in range(self.num_frames):
            update_plot(frame)

        height, width, _ = cv2.imread(self.frames_folder + '/frame_0.png').shape
        video_path = os.path.dirname(self.ens_file) + '/{}'.format(os.path.basename(self.ens_file).split('.')[0]) + '.mp4'
        video = cv2.VideoWriter(video_path, fourcc = cv2.VideoWriter_fourcc(*'mp4v'), frameSize=(width,height), fps=24)
        for frame in range(self.num_frames):
            video.write(cv2.imread(self.frames_folder + '/frame_{}.png'.format(frame)))
        cv2.destroyAllWindows()
        video.release()
        shutil.rmtree(self.frames_folder)
        hdf5.close()


