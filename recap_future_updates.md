# Future developments

## Problems
1. The procedure of Global Motion Estimation right now is performed only with the affine motion model, which is powerful but has also some limitations
2. There are some parameters that depend on the size of the video in input, on the quantity of motion and similar factors
    - blocksize depends on the size and also on the motion
    - search window
    - percentage of outliers
3. The procedure of calling the scripts is bo-ring and ugly

## Solutions
1. Implement other motion models to detect global motion
2. Implement some heuristics that automatically detect the best values for those parameters
3. Implement a CLI that shows the user the various possibilities for the parameters and the usages of the various scripts