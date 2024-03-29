# BOCA: Bayesian-Optimization-based Combinatorial Assignment

Published at [AAAI 2023](https://aaai.org/Conferences/AAAI-23/)

This is a piece of software used for computing the efficiency of the BOCA mechanism in the spectrum auction test suite (SATS) that are shown in Table 1 of the paper
[Bayesian-Optimization-based Combinatorial Assignment](https://arxiv.org/abs/2208.14698). The BOCA mechanism is described in detail in the following paper:

**BOCA: Bayesian-Optimization-based Combinatorial Assignment**<br/>
Jakob Weissteiner, Jakob Heiss, Julien Siems, and Sven Seuken.<br/>
*In Proceedings of the Thirty-first International joint Conference on Artificial Intelligence IJCAI'22, Vienna, AUT, July 2022.*<br/>
Full paper version including appendix: [[pdf](https://arxiv.org/abs/2208.14698)]


## Requirements

* Python 3.8
* Java 8 (or later)
  * Java environment variables set as described [here](https://pyjnius.readthedocs.io/en/stable/installation.html#installation)
* JAR-files ready (they should already be)
  * CPLEX (=20.01.0): The file cplex.jar (for 20.01.0) is provided in the folder lib.
  * [SATS](http://spectrumauctions.org/) (=0.8.0): The file sats-0.8.0.jar is provided in the folder lib.
* CPLEX Python API installed as described [here](https://www.ibm.com/docs/en/icos/20.1.0?topic=cplex-setting-up-python-api)
* Make sure that your version of CPLEX is compatible with the cplex.jar file in the folder lib.

## Dependencies

Prepare your python environment (whether you do that with `conda`, `virtualenv`, etc.) and enter this environment. Then install the required packages as provided in the requirements.txt

Using pip:
```bash
$ pip install -r requirements.txt

```
## SATS
In requirements.txt you ran pip install pysats. Finally, you have to set the PYJNIUS_CLASSPATH environment variable to the absolute path of the lib folder.

To automatically do this when activating the your conda evironment do the following:

* Locate the directory for the conda environment in your Anaconda Prompt by running in the command shell %CONDA_PREFIX%.

* Enter that directory and create these subdirectories and files:

```bash
cd %CONDA_PREFIX%
mkdir .\etc\conda\activate.d
mkdir .\etc\conda\deactivate.d
type NUL > .\etc\conda\activate.d\env_vars.bat
type NUL > .\etc\conda\deactivate.d\env_vars.bat
```

* Edit .\etc\conda\activate.d\env_vars.bat as follows:

set PYJNIUS_CLASSPATH=C:\path\to\lib\

* Edit .\etc\conda\deactivate.d\env_vars.bat as follows:

set PYJNIUS_CLASSPATH=

When you run conda activate <name_of_yur_environment> the environment variable PYJNIUS_CLASSPATH is set to the value you wrote in the env_vars.bat file. When you run conda deactivate, this variable is erased.


## How to run

### 1. BOCA: using our uUBs $\mathcal{M}_i^{\text{uUB}}$ in the acquisition function $\mathcal{A}:=\sum_i \mathcal{M}_i^{\text{uUB}}$.

To start BOCA for a specific quantile parameter $q$, a SATS domain (LSVM, SRVM, and MRVM), and a seed run the following command:

```bash
python sim_mlca.py --domain=LSVM --q=0.9 --seed=10001 --acquisition=uUB_model
```

This will create a results folder where you then find in results\LSVM\0.9\uUB_model the following files

1. a configuration file: config.json
2. a log file: log.txt
3. a result file: results.json.

Specifically, results.json contains the efficiency of the final allocation of the BOCA mechanism in the field "MLCA Efficiency". Note that in sim_mlca.py, we set the parameters (Qinit, Qmax, Qround) = (40, 100, 4) to their default values that were used to create the results in Table 1.

### 2. OUR-MVNN-MLCA: using our mean MVNNs $\mathcal{M}_i^{\text{mean}}$ in the acquisition function $\mathcal{A}:=\sum_i \mathcal{M}_i^{\text{mean}}$. 

To start OUR-MVNN-MLCA for a specific quantile parameter $q$, a SATS domain (LSVM, SRVM, and MRVM), and a seed run the following command:

```bash
python sim_mlca.py --domain=LSVM --q=0.9 --seed=10001 --acquisition=mean_model
```

This will create a results folder where you then find in results\LSVM\0.9\mean_model the following files

1. a configuration file: config.json
2. a log file: log.txt
3. a result file: results.json.

Specifically, results.json contains the efficiency of the final allocation of the OUR-MVNN-MLCA mechanism in the field "MLCA Efficiency".



