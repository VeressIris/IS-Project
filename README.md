# Intelligent Systems Project 2026

## Getting started

To use the platform, your python version must be at least 3.10.
The code has been tested with python 3.13, let us know if you encounter issues.

We strongly suggest installing conda (or pip) and using an environment.

### Create environment

An easy way to get that is using virtual environments. We suggest you install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) to manage them.
Then, you can use conda to copy the environment used to develop this project by running this command:

```sh
conda env create -f environment.yml
```

With this environment created, you can start it by running:

```sh
conda activate isp
```

Inside this environment you can install the dependencies as instructed below. Some IDEs also support the use of environments, and can help you install dependencies.

### getting the platform

After you have created the environment, make sure it is enabled.
Then, clone the repository, go into the folder and install the schnapsen package and its dependencies in editable mode by running:

```sh
pip install -e .
```

To run the tests, run:

```sh
pip install -e '.[test]'  # on Linux / MacOS
pip install -e ".[test]"  # on Windows

pytest ./tests
```

If the above fails, try deactivating your environment and activating it again.
Then retry installing the dependencies.

## Running the CLI

After intalling, you can try the provided command line interface examples.
This is where all of the relevant experiment code resides.

To run the CLI, run:

```sh
python cd executables
```

```sh
python cli.py
```

This will list the available commands.

## Re-creating the experiment

### Generating necessary data

Make sure your working directory is `IS-Project/executables`
Run

```sh
python cli.py get-tournament-data
```

If you want to change the number of games played you can add instead run `python cli.py get-tournament-data --games=<NUM_GAMES>`.
For example, to run 5000 games, execute

```sh
python cli.py get-tournament-data --games=2500
```

### Running the actual experiment

This is all simply done with one command:

```sh
python cli.py final-experiment
```

This command runs a binomial test, corrects the p-values and generates a bot ranking based on the significance of the performance and the win rate.

### Vizualizing data (in graph form)

#### Win rates

To view a bar graph comparing the **win rate** of one of the custom bots against the baselines and the other custom bot, run:

```sh
python cli.py show-win-rate-graph --bot=<BOT_NAME>
```

`BOT_NAME` can only be either `"blitz"` or `"siege"`.

#### Average scores

To view a bar graph comparing the **average score** of one of the custom bots against the baselines and the other custom bot, run:

```sh
python cli.py show-average-score-graph --bot=<BOT_NAME>
```

`BOT_NAME` can only be either `"blitz"` or `"siege"`.

#### Final ranking

The following command will display a directed graph where each node represents a bot and the weights of each edge contain the win rates of one bot vs the other.

```sh
python cli.py show-ranking
```
