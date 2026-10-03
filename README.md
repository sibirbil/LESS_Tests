# LESS_Tests

Benchmarks of LESS (Learning with Subset Stacking) against other regression methods.

## Data

The datasets are not stored in the repository. Download `datasetsR.zip` from the
link below and extract it so that the CSV files sit in a `datasetsR/` folder at
the repository root:

https://1drv.ms/u/c/0b26aa4e19835550/IQCEN0mqxK_3TI87u5uZ_jvoAQj8fQPWWK9t2MeiRfxtcak?e=rkVz94

```
LESS_Tests/
├── datasetsR/
│   ├── abalone.csv
│   ├── ...
├── main.py
└── ...
```

## Running

Install the dependencies with `pip install -r requirements.txt` and run the
scripts from the repository root with `PYTHONPATH=.`, for example:

```
PYTHONPATH=. python main.py
PYTHONPATH=. python plot_funcs/plot.py
```
