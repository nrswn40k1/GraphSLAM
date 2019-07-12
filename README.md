# GraphSLAM

### Graph SLAM algorithm

Graph SLAM is graph-based Simultaneous Localization and Mapping. 

#### Requirements (library dependency)
You need Python 3.6 or later to run GraphSLAM.
- numpy
- scipy
- matplotlib
- pandas

#### Datasets
You can make sample datasets locally by ```MakeSmaple.py```. 
Like this: 
```bash
mkdir data
cd src
python MakeSample.py
```
You will see ```sample0``` directory in ```data```.

You can also download datasets online [here](https://sourceforge.net/projects/slam-plus-plus/files/data/).

#### Quick start
First, install the libraries and change the current directory to src.

```bash
cd src
python main.py ../data/sample0 0
```
The last argument is index of methods for inverse matrix calculation.
You can choose the number between 0 to 2.

- "0" : numpy package as dense
- "1" : sum of block diagonal as dense
- "2" : scipy package as sparse

### References
[1] Sebastian Thrum, Wolfram Burgard, and Dieter Fox, 上田隆一(訳)(2017)「確率ロボティクス」
