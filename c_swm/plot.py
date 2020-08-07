import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

class TransitionPlot:

    N_COL = 8
    OBJ_NAMES = ['BG', 'SQ', 'SC', 'BQ', 'BC', 'P1', 'P2', 'P3', 'P4']

    def __init__(self, num_obj_slots):
        assert num_obj_slots in (4,8,9)
        self.COLORS = [cm.rainbow(x) for x in np.linspace(0, 1, num_obj_slots)]
        self.OBJ_ROW = np.ceil(num_obj_slots / 4).astype(np.int32)
        self.N_ROW = 4 + self.OBJ_ROW + 8
        self.FIGURE_SIZE = (self.N_COL*2, self.N_ROW*2)
        print(num_obj_slots)
        plt.subplots(figsize=self.FIGURE_SIZE)
        self.obs_axs = []
        self.act_axs = []
        self.latent_axs = []
        self.obj_axs = []
        self.next_obj_axs = []

        for i in [0, 3]:
            self.obs_axs.append(
                plt.subplot2grid(
                    shape=(self.N_ROW, self.N_COL), loc=(0, i*2), colspan=2, rowspan=2
                )
            )

        for i in [1, 2]:
            self.act_axs.append(
                plt.subplot2grid(
                    shape=(self.N_ROW, self.N_COL), loc=(0, i*2), colspan=2, rowspan=2
                )
            )

        for i in range(self.OBJ_ROW):
            for j in range(4):
                self.obj_axs.append(
                    plt.subplot2grid(shape=(self.N_ROW, self.N_COL), loc=(2 + i, 0 + j), colspan=1, rowspan=1)
                )
                self.next_obj_axs.append(
                    plt.subplot2grid(shape=(self.N_ROW, self.N_COL), loc=(2 + i, 4 + j), colspan=1, rowspan=1)
                )


        self.latent_axs.append(
            plt.subplot2grid(
                shape=(self.N_ROW, self.N_COL), loc=(2 + self.OBJ_ROW, 0), colspan=4, rowspan=4
            )
        )

        self.latent_axs.append(
            plt.subplot2grid(
                shape=(self.N_ROW, self.N_COL), loc=(2 + self.OBJ_ROW, 4), colspan=4, rowspan=4
            )
        )

        self.latent_axs.append(
            plt.subplot2grid(
                shape=(self.N_ROW, self.N_COL), loc=(2 + self.OBJ_ROW + 4, 4), colspan=4, rowspan=4
            )
        )

        plt.tight_layout()


    def reset(self):
        for ax in self.obj_axs + self.next_obj_axs + self.act_axs + self.obs_axs:
            ax.axis('off')
        for i in range(3):
            self.latent_axs[i].cla()
            self.latent_axs[i].set_xlim(-5, 5)
            self.latent_axs[i].set_ylim(-5, 5)
        self.latent_axs[0].set_title("Pre State Latent", fontsize=6)
        self.latent_axs[1].set_title("Next State Latent", fontsize=6)
        self.latent_axs[2].set_title("Pre State Latent +\n Transition", fontsize=6)
        self.obs_axs[0].set_title("Pre State Latent", fontsize=6)
        self.obs_axs[1].set_title("Next State Latent", fontsize=6)
        self.act_axs[0].set_title("Action Moving Object", fontsize=6)
        self.act_axs[1].set_title("Action Target Object", fontsize=6)


    def plt_observations(self, obs, next_obs):
        np_obs = np.transpose(obs[0].cpu().numpy(), (1,2,0))
        np_next_obs = np.transpose(next_obs[0].cpu().numpy(), (1,2,0))
        # print(np_obs.shape, np_next_obs.shape)
        self.obs_axs[0].imshow(np_obs)
        self.obs_axs[1].imshow(np_next_obs)

    def plt_action(self, action):
        np_mov_obj = np.transpose(action[0][0].cpu().numpy(), (1,2,0))
        np_tar_obj = np.transpose(action[0][1].cpu().numpy(), (1,2,0))
        # print(np_obs.shape, np_next_obs.shape)
        self.act_axs[0].imshow(np_mov_obj)
        self.act_axs[1].imshow(np_tar_obj)

    def plt_objects(self, objs, next_objs):
        for i in range(objs.size()[1]):
            np_obj = np.transpose(objs[0][i].cpu().numpy(), (1,2,0))
            np_next_obj = np.transpose(next_objs[0][i].cpu().numpy(), (1,2,0))
            self.obj_axs[i].imshow(np_obj)
            self.next_obj_axs[i].imshow(np_next_obj)


    def plt_latent(self, state, next_state, pred_state):
        legend = [[], [], []]
        for i in range(state.size()[1]):
            np_state = state[0][i].cpu().numpy()
            np_next_state = next_state[0][i].cpu().numpy()
            np_pred_state = pred_state[0][i].cpu().numpy()
            # if i == 0:
            #     print(np_state, np_next_state, np_pred_state)
            #     print("-*40")

            self.latent_axs[0].scatter(np_state[0], np_state[1], color=self.COLORS[i], marker='x', s=10)
            self.latent_axs[1].scatter(np_next_state[0], np_next_state[1], color=self.COLORS[i], marker='x', s=10)
            self.latent_axs[2].scatter(np_pred_state[0], np_pred_state[1], color=self.COLORS[i], marker='x', s=10)
        #
            for j, st in enumerate([np_state, np_next_state, np_pred_state]):
                legend[j].append("{}-({:.2f},{:.2f})".format(self.OBJ_NAMES[i], st[0], st[1]))

        for ax, lgd in zip(self.latent_axs, legend):
            ax.legend(lgd, prop={'size': 6}, loc=2, ncol=2)

    def show(self, interval=0.5):
        plt.pause(interval)

    def close(self):
        plt.close()

