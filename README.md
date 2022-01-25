# Global motion estimation
Global motion estimation w/ a hierarchical and robust approach.

## Pseudocode
```
const LEVELS = n
video = capture()
for i=1 to video.length:
    current = video[i]
    precedent = video[i-1]

    global_motion = GME(current, precedent)
    show(global_motion)


//// Functions

function GME(curr, prec){
    curr_pyr = pyr(curr)
    prec_pyr = pyr(prec)

    // execution of level 1: dense motion estimation
    coarser_motion_estimation = motion_estimation(curr_pyr[0], prec_pyr[0])
    current_parameters = estimate_parameters(coarser_motion_estimation)

    // execution of lower levels:
    for i=1 to LEVELS:
        current_parameters = project_parameters()
        current_parameters = optimize_parameters(current_parameters, curr[i], prec[i])
}

function motion_estimation(current_frame, precedent_frame){
    // BBME here
}

function estimate_parameters(motion_map){
    // parameter estimation from dense motion field
}

function project_parameters(current){
    // transform parameters from level i to level i+1
    // paper pag. 2
}

function optimize_parameters(current_parameters, current_frame, previous_frame){
    // gradient descent to fix the parameters
}
```

## Assumptions
- Since the paper does not specify how to get the parameters from the first motion estimation I assume it sets all of them to 0 but a0 and a1 which are clearly initialized
