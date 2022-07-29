# PHYS449

## Dependencies

- json
- numpy
-pytorch

## Running `main.py`

To run `main.py`, use

```sh
python main.py -json "data/arguments.json" -input "data/in.txt" -output "results" -verbosity 4
```
All of the parameters have default values that can be seen using main.py --help.

The json file contains the information about the learning rate, the number of epochs and the number of iterations for the Metropolis-Hasting sampling.


