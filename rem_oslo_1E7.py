# Axel Voss-Schrader
# abv15, 00951180
# Project started 17/1 2018

import numpy as np
import sys
import datetime
import random
# np.random.seed(512)


class Lattice:
    def __init__(self, L, p=0.5):
        """
        Initialise system of L sites in an empty configuration.
        Give each site a threshold slope of 1 or 2 with probabilities p or 1-p respectively.
        """
        self.L = L
        self.p = p
        self.reset_lattice()

    def reset_lattice(self):
        self.sites = []
        self.crossover = False
        self.crossover_time = 0

        self.ave_sizes, self.heights = [], []

        for _ in range(self.L):
            zth = np.random.choice([1, 2], p=[self.p, 1-self.p])
            self.sites.append({"height": 0, "threshold": zth})

    def add_until_crossover(self, print_stats=True):
        """
        Keeps adding grains at the boundary until the system reaches the cross over point (steady state)
        """
        assert self.crossover_time == 0, "System needs to be in initial state to use add_until_crossover()."

        self.ave_sizes, self.heights = [], []

        t0 = t1 = datetime.datetime.now()
        sys.stdout.write("Reaching cross over for system size {0}\nProcess started: {1}\n"
                         .format(self.L, t0.strftime('%Y-%m-%d %H:%M:%S')))

        i = 0
        while not self.crossover:
            if (datetime.datetime.now() - t1).total_seconds() > 1:
                t1 = datetime.datetime.now()
                sys.stdout.write("\rDropped: {0} grains // {1} elapsed".format(i, str(t1 - t0)[:-7]))
                sys.stdout.flush()

            self.sites[0]["height"] += 1
            self.crossover_time += 1
            size = self._relax_system()

            # Append values for analysis
            self.heights.append(self.sites[0]["height"])
            self.ave_sizes.append(size)

            i += 1

        sys.stdout.write("\rDropped: {0} grains // {1} elapsed".format(i, str(t1 - t0)[:-7]))
        sys.stdout.flush()

        stats = self._get_stats_dict(t0, t1, i)

        if print_stats:
            for key, val in stats.items():
                sys.stdout.write("\n - {0}: {1}".format(key, val))
        sys.stdout.write("\n\n")

        return stats

    def add_and_relax(self, num_grains, print_stats=True):
        """
        Adds grains at the boundary and relaxes the system
        """
        if not self.crossover:
            print("WARNING: Calling add_and_relax() but system is not yet in steady state.")

        self.ave_sizes, self.heights = [], []

        num_grains = int(num_grains)

        last_perc = 0
        t0 = t1 = datetime.datetime.now()
        sys.stdout.write("Dropping {0} grains in system size {1}\nSimulation started: {2}\n"
                         .format(num_grains, self.L, t0.strftime('%Y-%m-%d %H:%M:%S')))

        for i in range(num_grains):
            if (datetime.datetime.now() - t1).total_seconds() > 1:
                curr_perc = 100 * i / num_grains
                if last_perc + 1 < curr_perc:
                    last_perc = int(curr_perc)

                t1 = datetime.datetime.now()
                sys.stdout.write("\rProgress: {0}% // {1} elapsed".format(last_perc, str(t1-t0)[:-7]))
                sys.stdout.flush()

            self.sites[0]["height"] += 1
            if not self.crossover: self.crossover_time += 1
            size = self._relax_system()

            # Append values for analysis
            self.heights.append(self.sites[0]["height"])
            self.ave_sizes.append(size)

        t1 = datetime.datetime.now()
        sys.stdout.write("\rProgress: 100% // {0} elapsed".format(str(t1 - t0)[:-7]))
        sys.stdout.flush()

        stats = self._get_stats_dict(t0, t1, num_grains)

        if print_stats:
            for key, val in stats.items():
                sys.stdout.write("\n - {0}: {1}".format(key, val))
        sys.stdout.write("\n\n")

        return stats

    def _relax_system(self):
        relaxed = False
        avalanche_size = 0
        lo, hi = 0, 0

        while not relaxed:
            relaxed = True  # Start every iteration by assuming the system is relaxed until shown otherwise

            # Iterate through the lattice sites
            for i in range(lo, self.L):

                # Check if the slope is greater than its threshold
                if self.get_slope(i) > self.sites[i]["threshold"]:
                    if relaxed: lo_next = i - 1  # Keep track of where avalanche started
                    hi_next = i  # Keep track of where avalanche ended
                    avalanche_size += 1  # Increment avalanche size count
                    relaxed = False  # System is not relaxed yet

                    self.sites[i]["height"] -= 1  # Decrement the current height
                    try:  # Try incrementing the subsequent height
                        self.sites[i + 1]["height"] += 1
                    except IndexError:  # Special case if at the last site, L
                        # TODO check behaves as expected
                        # Defined as the time BEFORE a grain leaves for the first time
                        if not self.crossover: self.crossover_time -= 1
                        self.crossover = True
                    finally:  # Assign new threshold slope and increment avalanche size count
                        self.sites[i]["threshold"] = np.random.choice([1, 2], p=[self.p, 1-self.p])

                elif i >= hi:
                    break

            if not relaxed:
                lo = lo_next
                hi = hi_next

        return avalanche_size

    def _get_stats_dict(self, ti, tf, tot_grains):
        """
        Creates a dictionary with the statistics from the simulation.
        Used by simulation methods and should not be called externally.
        """
        return {"Simulation started": ti.strftime('%Y-%m-%d %H:%M:%S'),
                "Simulation finished": tf.strftime('%Y-%m-%d %H:%M:%S'),
                "Time taken": str(tf - ti)[:-7],
                "System size": self.L,
                "p": self.p,
                "Grains dropped": tot_grains,
                "Reached cross over after": False if not self.crossover else self.crossover_time,
                "Largest avalanche": max(self.ave_sizes),
                "Average pile height": sum(self.heights) / len(self.heights)
                }

    def get_slope(self, i):
        try:
            return self.sites[i]["height"] - self.sites[i + 1]["height"]
        except IndexError:  # If at the last site, L
            return self.sites[i]["height"]

    def get_current_pile(self):
        """
        Returns the current state of the pile (index, height, threshold slope)
        """
        i, h, zth = [], [], []
        for j in range(self.L):
            i.append(j)
            h.append(self.sites[j]["height"])
            zth.append(self.sites[j]["threshold"])
        return i, h, zth

    def get_avalanches(self):
        """
        Returns the time and  avalanche sizes from the last simulation.
        """
        return self.ave_sizes

    def get_heights(self):
        """
        Returns the time and heights from the last simulation.
        """
        return self.heights


def test_pile_height():
    pile16 = Lattice(16, p=0.5)
    pile16.add_until_crossover()
    pile16.add_and_relax(1E4)
    del pile16

    pile32 = Lattice(32, p=0.5)
    pile32.add_until_crossover()
    pile32.add_and_relax(1E4)
    del pile32


def sim_log_once_numgrains(filename, size, num_grains):
    pile = Lattice(size, p=0.5)
    stats = pile.add_and_relax(num_grains, print_stats=True)
    heights = pile.get_heights()
    asizes = pile.get_avalanches()

    with open(filename, "a") as outfile:
        outfile.write("stats = {0}\nh = {1}\na = {2}\nfinal_state = {3}\n\n"
                      .format(stats, heights, asizes, pile.sites))

    del pile


def sim_log_once_crossover(filename, sizes):
    for L in sizes:
        pile = Lattice(L, p=0.5)
        stats = pile.add_until_crossover(print_stats=True)
        heights = pile.get_heights()
        asizes = pile.get_avalanches()

        with open(filename, "a") as outfile:
            outfile.write("stats = {0}\nh = {1}\na = {2}\nfinal_state = {3}\n\n"
                          .format(stats, heights, asizes, pile.sites))

        del pile


s = [4, 8, 16, 32, 64, 128, 256, 512]

# Stage 1
print("\n### STAGE 1 ###\n")
for L in s:
    sim_log_once_numgrains("s{0}_n1E6_2a3a.txt".format(L), L, 10000000)

# Stage 2
# print("\n### STAGE 2 ###\n")
# i = 1
# rand = random.randint(0, 100000)
# while True:
#     print("### ITERATION: {0} ###\n".format(i))
#     sim_log_once_crossover("i{0}_4to512_crossover_2a3a_{1}.txt".format(i, rand), s)
#     i += 1


# plot_height()

# plot_moving_averages("height_log.txt", start=1, rsep=3, num_sets=7, W=25)

# pile512 = Lattice(512, p=0.5)
# pile512.add_until_crossover(print_stats=True)

# pile64 = Lattice(64, p=0.5)
# pile64.add_and_relax(1000, print_stats=True)

# pile64.add_and_relax(4000, print_stats=True)

# pile1024 = Lattice(1024, p=0.5)
# pile1024.add_until_crossover(print_stats=True)
# pile1024.add_and_relax(100, print_stats=True)

# i, h, zth = pile.get_pile()
# plt.errorbar(i, z, [zth, [0 for _ in range(len(zth))]], linestyle=None, ecolor="red")
# t, s = pile.get_avalanches()
# plt.vlines(t, 0, s)
# plt.xlabel("t")
# plt.ylabel("S / S_max")
# plt.show()


# """
# How many iterations for average?
# do not include zeros in logbin plot
# Points for larger systems?
#
# no for loops
# not append to np array
# use numpy
#
#
# # For the task on fitting: How can you plot, to get 2 of the three variables from the fit?
#
# # Covariance matrix in scipy
# # IMPORTANT: Do fitting for each parameter in turn, not for all three at once!
# # Estimate 1 param
# # Based on that param, read off the others
#
#
# Do 10^6 avalanches in task 3!!!!!
#
#
# Test to use: set p=1 and let the system just reach steady state.
# You know the next avalanche should be of exactly size L.
#
# """
