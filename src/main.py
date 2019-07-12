import os
import sys
import matplotlib.pyplot as plt

from GraphSLAM import GraphSLAM
import DataCenter


dirname = sys.argv[1]
method = int(sys.argv[2])

fname = os.path.join(dirname, "input.txt")
savefile = os.path.join(dirname, "result.png")

z, u, c, n = DataCenter.read(fname)

graphslam = GraphSLAM(z, u, c, n)

graphslam.main(method=method)
plt.savefig(savefile)
plt.show()
