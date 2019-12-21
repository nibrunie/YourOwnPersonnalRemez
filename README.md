
# YourOwnPersonnalRemez
Simple tool for playing around polynomial approximation for numerical functions.

# Dependencies

- numpy
- matplotlib
- bigfloat
- fpylll [https://github.com/fplll/fpylll](https://github.com/fplll/fpylll)

All dependencies can be installed with pip using the provided ` requirements.txt ` file (assuming your setup already has their dependencies, such as a recent version of fplll for example [https://github.com/fplll/fplll](https://github.com/fplll/fplll))

 ` pip3 install -r requirements.txt `

# Usage
##  CVP based approximation
Using Euclidean Lattices and the Closest Vector Problem, this script tries to find a good approximation for a given function
 ` python3 cvp_approx.py `

 # References
 -- N. Brisebarre, S. Chevillard "Efficient polynomial L-inf approximation" ([https://hal.archives-ouvertes.fr/inria-00119513v1](https://hal.archives-ouvertes.fr/inria-00119513v1))
