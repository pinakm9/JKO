import numpy as np
import tables
import os
import matplotlib.pyplot as plt
import cv2
import shutil

class SDE:
    """
    Description:
        Implementation of a an SDE of the form dX_t = mu(t, X_t)dt + sigma(t, X_t)dB_t
    Args:
        space_dim: space dimension of the SDE
        mu: a function of time and space mu(t, X_t)
        sigma: a function of time and space sigma(t, X_t)
        record_path: file path ending with .h5 describing where to record the ensemble evolution 
        dtype: numpy float dtype 32-bit or 64-bit, default=np.float64 
    """
    def __init__(self, space_dim, mu, sigma, record_path, dtype=np.float64):
        self.space_dim = space_dim
        self.mu = mu
        self.sigma = sigma
        self.record_path = record_path
        hdf5 = tables.open_file(record_path, 'w')
        hdf5.close()
        self.dtype = dtype
        col = tables.Float64Col if self.dtype == np.float64 else tables.Float32Col
        self.point_description = {}
        for j in range(space_dim):
            self.point_description['x' + str(j)] = col(pos = j)

    def evolve(self, initial_ensemble, final_time, time_step):
        """
        Description:
            evolves an initial ensemble according to the SDE dynamics
        Args:
            initial_ensemble: the ensemble that starts the evolution
            final_time: final time in the evolution assuming we're starting at time=0
            time_step: time_step in Euler-Maruyma method
        """
        self.num_particles = len(initial_ensemble) 
        num_steps = int(final_time / time_step)
        noise_std = np.sqrt(time_step)
        hdf5 = tables.open_file(self.record_path, 'a')
        new_ensemble = np.zeros((self.num_particles, self.space_dim))
        # record the initial ensemble
        tbl = hdf5.create_table(hdf5.root, 'time_0', self.point_description)
        tbl.append(initial_ensemble)
        tbl.flush()
        for step in range(num_steps):
            # evolve ensemble with Euler-Maruyama
            noise = np.random.normal(loc=0.0, scale=noise_std, size=self.space_dim)
            for i, particle in enumerate(initial_ensemble):
                #print('particle:', particle, type(particle))
                new_ensemble[i] = particle + self.mu(step*time_step, particle)*time_step + np.dot(self.sigma(step*time_step, particle), noise)
            # record the new ensemble
            tbl = hdf5.create_table(hdf5.root, 'time_' + str(step + 1), self.point_description)
            tbl.append(new_ensemble)
            tbl.flush()
            # prepare for the next step
            initial_ensemble = new_ensemble
        hdf5.close()

class SDEPlotter:
    """
    Description:
        plots the ensemble evolution of an SDE
    Args:
        ens_file: path to .h5 file containing the ensemble evolution
        fig_size: size of plots in inch x inch as a tuple (width, height), default=(10, 10) 
    """
    def __init__(self, ens_file, time_step, fig_size=(10, 10), ax_lims=[None, None, None]):
        self.ens_file = ens_file
        self.fig_size = fig_size
        hdf5 = tables.open_file(ens_file, 'r')
        ensemble = np.array(hdf5.root.time_0.read().tolist())
        dim = ensemble.shape[1]
        if dim == 2 or dim == 3:
            # create folder to store images
            try:
                self.frames_folder = os.path.dirname(ens_file) + '/{}'.format(os.path.basename(ens_file).split('.')[0]) + '_frames'
                os.mkdir(self.frames_folder)
            except:
                pass
            # animate
            self.animate(dim, hdf5, time_step, ax_lims)
        else:
            print('Particles in the ensemble must be of either dimension 2 or 3!!!')
        hdf5.close()

    def animate(self, dim, hdf5, time_step, ax_lims):
        """
        Description:
            creates an animation of the ensemble evolution
        Args:
            hdf5: handle to the .h5 file containing the ensemble evolution
            time_step: time between consecutive frames for labelling
        """
        fig = plt.figure(figsize=self.fig_size)
        ax = fig.add_subplot(111) if dim == 2 else fig.add_subplot(111, projection='3d')

        def update_plot(frame):
            ax.clear()
            # read data to plot
            ensemble = np.array(getattr(hdf5.root, 'time_' + str(frame)).read().tolist())
            if dim == 2:
                ax.scatter(ensemble[:, 0], ensemble[:, 1])
            else:
                ax.scatter(ensemble[:, 0], ensemble[:, 1], ensemble[:, 2], c='b', s=10.0)
            ax.set_title('time = {:.2f}'.format(frame*time_step))
            ax.set_xlabel('x')
            if ax_lims[0] is not None:
                ax.set_xlim(ax_lims[0])
            ax.set_ylabel('y')
            if ax_lims[1] is not None:
                ax.set_xlim(ax_lims[1])
            if dim == 3:
                ax.set_zlabel('z')
                if ax_lims[2] is not None:
                    ax.set_xlim(ax_lims[2])
            plt.savefig(self.frames_folder + '/frame_{}.png'.format(frame))
            print('Frame {} has been drawn.'.format(frame))

        for frame, _ in enumerate(hdf5.walk_nodes("/", "Table")):
            update_plot(frame)

        height, width, _ = cv2.imread(self.frames_folder + '/frame_0.png').shape
        video_path = os.path.dirname(self.ens_file) + '/{}'.format(os.path.basename(self.ens_file).split('.')[0]) + '.mp4'
        video = cv2.VideoWriter(video_path, fourcc = cv2.VideoWriter_fourcc(*'mp4v'), frameSize=(width,height), fps=24)
        for frame, _ in enumerate(hdf5.walk_nodes("/", "Table")):
            video.write(cv2.imread(self.frames_folder + '/frame_{}.png'.format(frame)))
        cv2.destroyAllWindows()
        video.release()
        shutil.rmtree(self.frames_folder)
        hdf5.close()


