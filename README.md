
# YourOwnPersonnalRemez
Simple tool for playing around polynomial approximation for numerical functions.

# Dependencies

- numpy
- matplotlib
- bigfloat
- json
- fpylll [https://github.com/fplll/fpylll](https://github.com/fplll/fpylll)

All dependencies can be installed with pip using the provided ` requirements.txt ` file (assuming your setup already has their dependencies, such as a recent version of fplll for example [https://github.com/fplll/fplll](https://github.com/fplll/fplll))

 ` pip3 install -r requirements.txt `

# Usage

## Command line

yopr can be called from the command line

 ` python3 yopr.py --method remez --function "bigfloat.exp" --interval 0,0.125 --epsilon 1e-7 --degree 8 --plot `

A list of possible options can be displayed:
 ` python3 yopr.py --help `

### Dumping axf-compatible output

By using `--dump-axf-approx <filename>` `yopr` can output the approximation in the AXF format (see https://github.com/metalibm/axf for more information on AXF, the Approximation eXchange Format).

```python
python3 yopr.py --method remez --function "bigfloat.exp" --interval 0,0.125 --epsilon 1e-7 --degree 8 --dump-axf-approx exp-fp32.json.axf 
```

## Examples

Various example scripts are available and can be adapted to your needs:

# Internal

##  CVP based approximation
Using Euclidean Lattices and the Closest Vector Problem, this method tries to find a good approximation for a given function.

## Remez approximation

## Polynomial conditionner


# References
 -- N. Brisebarre, S. Chevillard "Efficient polynomial L-inf approximation" ([https://hal.archives-ouvertes.fr/inria-00119513v1](https://hal.archives-ouvertes.fr/inria-00119513v1))
