import os
import sys
import matplotlib.pyplot as plt

from GraphSLAM import GraphSLAM
import DataCenter

method = int(sys.argv[2])


dirname = "../data/sample3"
fname = os.path.join(dirname, "input.txt")
savefile = os.path.join(dirname, "result.png")

z, u, c, n = DataCenter.read(fname)

graphslam = GraphSLAM(z, u, c, n)

graphslam.fig = plt.figure(figsize=(5,5))
graphslam.ax1 = graphslam.fig.add_subplot(111)

graphslam.main(method=method)
plt.savefig(savefile)
plt.show()
