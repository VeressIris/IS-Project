import random
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import networkx as nx
from statistics import mean

from typing import Optional

import click
from schnapsen.alternative_engines.ace_one_engine import AceOneGamePlayEngine

from schnapsen.bots import MLDataBot, train_ML_model, MLPlayingBot, RandBot, BullyBot

from schnapsen.bots.example_bot import ExampleBot
from schnapsen.bots.blitz_siege import Blitz, Siege

from schnapsen.game import (Bot, GamePlayEngine, Move, PlayerPerspective,
                            SchnapsenGamePlayEngine, TrumpExchange)
from schnapsen.alternative_engines.twenty_four_card_schnapsen import TwentyFourSchnapsenGamePlayEngine

from schnapsen.bots.rdeep import RdeepBot


@click.group()
def main() -> None:
    """Various Schnapsen Game Examples"""


def play_games_and_return_stats(engine: GamePlayEngine, bot1: Bot, bot2: Bot, pairs_of_games: int) -> int:
    """
    Play 2 * pairs_of_games games between bot1 and bot2, using the SchnapsenGamePlayEngine, and return how often bot1 won.
    Prints progress. Each pair of games is the same original dealing of cards, but the roles of the bots are swapped.
    """
    bot1_wins: int = 0
    lead, follower = bot1, bot2
    for game_pair in range(pairs_of_games):
        for lead, follower in [(bot1, bot2), (bot2, bot1)]:
            winner, _, _ = engine.play_game(lead, follower, random.Random(game_pair))
            if winner == bot1:
                bot1_wins += 1
        if game_pair > 0 and (game_pair + 1) % 500 == 0:
            print(f"Progress: {game_pair + 1}/{pairs_of_games} game pairs played")
    return bot1_wins


@main.command()
def random_game() -> None:
    engine = SchnapsenGamePlayEngine()
    bot1 = RandBot(random.Random(12112121))
    bot2 = RandBot(random.Random(464566))
    for i in range(1000):
        winner_id, game_points, score = engine.play_game(bot1, bot2, random.Random(i))
        print(f"Game ended. Winner is {winner_id} with {game_points} points, score {score}")


class NotificationExampleBot(Bot):

    def get_move(self, perspective: PlayerPerspective, leader_move: Optional[Move]) -> Move:
        moves = perspective.valid_moves()
        return moves[0]

    def notify_game_end(self, won: bool, perspective: PlayerPerspective) -> None:
        print(f'result {"win" if won else "lost"}')
        print(f'I still have {len(perspective.get_hand())} cards left')

    def notify_trump_exchange(self, move: TrumpExchange) -> None:
        print(f"That trump exchanged! {move.jack}")


@main.command()
def notification_game() -> None:
    engine = TwentyFourSchnapsenGamePlayEngine()
    bot1 = NotificationExampleBot()
    bot2 = RandBot(random.Random(464566))
    engine.play_game(bot1, bot2, random.Random(94))


class HistoryBot(Bot):
    def get_move(self, perspective: PlayerPerspective, leader_move: Optional[Move]) -> Move:
        history = perspective.get_game_history()
        print(f'the initial state of this game was {history[0][0]}')
        moves = perspective.valid_moves()
        return moves[0]


@main.command()
def try_example_bot_game() -> None:
    engine = SchnapsenGamePlayEngine()
    bot1 = ExampleBot()
    bot2 = RandBot(random.Random(464566))
    winner, points, score = engine.play_game(bot1, bot2, random.Random(1))
    print(f"Winner is: {winner}, with {points} points, score {score}!")


@main.command()
def rdeep_game() -> None:
    bot1: Bot
    bot2: Bot
    engine = SchnapsenGamePlayEngine()
    rdeep = bot1 = RdeepBot(num_samples=16, depth=4, rand=random.Random(4564654644))
    bot2 = RandBot(random.Random(464566))
    wins = 0
    amount = 100
    for game_number in range(1, amount + 1):
        if game_number % 2 == 0:
            bot1, bot2 = bot2, bot1
        winner_id, _, _ = engine.play_game(bot1, bot2, random.Random(game_number))
        if winner_id == rdeep:
            wins += 1
        if game_number % 10 == 0:
            print(f"won {wins} out of {game_number}")


@main.group()
def ml() -> None:
    """Commands for the ML bot"""


@ml.command()
def create_replay_memory_dataset() -> None:
    # define replay memory database creation parameters
    num_of_games: int = 10000
    replay_memory_dir: str = 'ML_replay_memories'
    replay_memory_filename: str = 'random_random_10k_games.txt'
    replay_memory_location = pathlib.Path(replay_memory_dir) / replay_memory_filename

    bot_1_behaviour: Bot = RandBot(random.Random(5234243))
    # bot_1_behaviour: Bot = RdeepBot(num_samples=4, depth=4, rand=random.Random(4564654644))
    bot_2_behaviour: Bot = RandBot(random.Random(54354))
    # bot_2_behaviour: Bot = RdeepBot(num_samples=4, depth=4, rand=random.Random(68438))
    delete_existing_older_dataset = False

    # check if needed to delete any older versions of the dataset
    if delete_existing_older_dataset and replay_memory_location.exists():
        print(f"An existing dataset was found at location '{replay_memory_location}', which will be deleted as selected.")
        replay_memory_location.unlink()

    # in any case make sure the directory exists
    replay_memory_location.parent.mkdir(parents=True, exist_ok=True)

    # create new replay memory dataset, according to the behaviour of the provided bots and the provided random seed
    engine = SchnapsenGamePlayEngine()
    replay_memory_recording_bot_1 = MLDataBot(bot_1_behaviour, replay_memory_location=replay_memory_location)
    replay_memory_recording_bot_2 = MLDataBot(bot_2_behaviour, replay_memory_location=replay_memory_location)
    for i in range(1, num_of_games + 1):
        if i % 500 == 0:
            print(f"Progress: {i}/{num_of_games}")
        engine.play_game(replay_memory_recording_bot_1, replay_memory_recording_bot_2, random.Random(i))
    print(f"Replay memory dataset recorder for {num_of_games} games.\nDataset is stored at: {replay_memory_location}")


@ml.command()
def train_model() -> None:
    # directory where the replay memory is saved
    replay_memory_filename: str = 'random_random_10k_games.txt'
    # filename of replay memory within that directory
    replay_memories_directory: str = 'ML_replay_memories'
    # Whether to train a complicated Neural Network model or a simple one.
    # Tips: a neural network usually requires bigger datasets to be trained on, and to play with the parameters of the model.
    # Feel free to play with the hyperparameters of the model in file 'ml_bot.py', function 'train_ML_model',
    # under the code of body of the if statement 'if use_neural_network:'
    replay_memory_location = pathlib.Path(replay_memories_directory) / replay_memory_filename
    model_name: str = 'simple_model'
    model_dir: str = "ML_models"
    model_location = pathlib.Path(model_dir) / model_name
    overwrite: bool = False

    if overwrite and model_location.exists():
        print(f"Model at {model_location} exists already and will be overwritten as selected.")
        model_location.unlink()

    train_ML_model(replay_memory_location=replay_memory_location, model_location=model_location,
                   model_class='LR')


@ml.command()
def try_bot_game() -> None:
    engine = SchnapsenGamePlayEngine()
    model_dir: str = 'ML_models'
    model_name: str = 'simple_model'
    model_location = pathlib.Path(model_dir) / model_name
    bot1: Bot = MLPlayingBot(model_location=model_location)
    bot2: Bot = RandBot(random.Random(464566))
    number_of_games: int = 10000
    pairs_of_games = number_of_games // 2

    # play games with altering leader position on first rounds
    ml_bot_wins_against_random = play_games_and_return_stats(engine=engine, bot1=bot1, bot2=bot2, pairs_of_games=pairs_of_games)
    print(f"The ML bot with name {model_name}, won {ml_bot_wins_against_random} times out of {number_of_games} games played.")


@main.command()
def game_24() -> None:
    engine = TwentyFourSchnapsenGamePlayEngine()
    bot1 = RandBot(random.Random(12112121))
    bot2 = RandBot(random.Random(464566))
    for i in range(1000):
        winner_id, game_points, score = engine.play_game(bot1, bot2, random.Random(i))
        print(f"Game ended. Winner is {winner_id} with {game_points} points, score {score}")


@main.command()
def game_ace_one() -> None:
    engine = AceOneGamePlayEngine()
    bot1 = RandBot(random.Random(12112121))
    bot2 = RdeepBot(num_samples=16, depth=4, rand=random.Random(464566))
    for i in range(1000):
        winner_id, game_points, score = engine.play_game(bot1, bot2, random.Random(i))
        print(f"Game ended. Winner is {winner_id} with {game_points} points, score {score}")

BASELINES: list[Bot] = [RandBot(random.Random(81), "RandBot"), BullyBot(random.Random(24), "BullyBot"), RdeepBot(4, 10, random.Random(1221), "RdeepBot")]
def run_tournament(games: int, bot: Bot) -> dict[int, dict[str, any]]:
    """
    Plays 'games' amount of games with 'bot' against RandBot, BullyBot and BullyBot and our other bot and returns the results of each game.

    Params:
        games (int): number of games to play
        bot (Bot): the bot that plays against the baseline bots

    Returns:
        dict[int, dict[str, any]]: The name of the winner, score of the winner, name of loser and score of loser in each match
    """
    engine = SchnapsenGamePlayEngine()
    
    scores:dict[int, dict[str, any]] = {}

    game = 1
    for baseline in BASELINES:
        for i in range(games):
            if baseline == bot:
                continue
            state = engine.play_game(bot, baseline, random.Random(i))
            scores[game] = state
            game += 1
    
    other_bot = Blitz("blitz") if isinstance(bot, Siege) else Siege("siege")
    for i in range(games):
        state = engine.play_game(bot, other_bot, random.Random(i))
        scores[game] = state
        game += 1

    return scores

@main.command()
@click.option("--games", default=2500, show_default=True, type=int)
def get_tournament_data(games: int) -> None:
    """
    Stores scores of each game the bots played against the baseline and each other in separate csv files.
    
    Params:
        games (int): number of different games the bots should play against the baseline bots.

    Returns:
        None
    """
    BLITZ_TOURNAMENTS: str = "../experiments/blitz_tournament.csv"
    SIEGE_TOURNAMENTS: str = "../experiments/siege_tournament.csv"
    
    blitz_scores = run_tournament(games, Blitz("blitz"))
    siege_scores = run_tournament(games, Siege("siege"))
    
    # store game data for blitz
    df = pd.DataFrame.from_dict(blitz_scores, orient="index")
    df.to_csv(BLITZ_TOURNAMENTS, index=False)

    # store game data for siege
    df = pd.DataFrame.from_dict(siege_scores, orient="index")
    df.to_csv(SIEGE_TOURNAMENTS, index=False)

    for baseline in BASELINES:
        path = f"../experiments/{baseline.get_name()}_tournament.csv"
        baseline_scores = run_tournament(games, baseline)
        df = pd.DataFrame.from_dict(baseline_scores, orient="index")
        df.to_csv(path, index=False)

def get_wins_count(winner_name: str, opponent_name: str) -> int:
    """
    Calculates the number of wins a bot achieved against another specified bot.
    
    Params:
        winner_name (str): name of winning bot
        opponent_name (str): name of opponent bot
    
    Returns:
        int: the number of wins a bot got against another bot
    """
    tournament_path = f"../experiments/{winner_name}_tournament.csv"
    df = pd.read_csv(tournament_path)
    winning_games = df[(df['winner'] == winner_name) & (df['loser'] == opponent_name)] # get only games played against specified opponent
    return winning_games.shape[0]

def get_win_rate(winner_name: str, opponent_name: str) -> float:
    """
    Calculates the win rate of a bot against another specified bot.
    
    Params:
        winner_name (str): name of winning bot
        opponent_name (str): name of opponent bot
    
    Returns:
        float: Win rate of specified bot
    """
    tournament_path = f"../experiments/{winner_name}_tournament.csv"
    df = pd.read_csv(tournament_path)
    games = df.shape[0] // 3 # divided by 3 because we have 3 baseline bots
    winning_games = df[(df['winner'] == winner_name) & (df['loser'] == opponent_name)] # get only games played against specified opponent
    wins = winning_games.shape[0]

    return wins / games

@main.command()
@click.option("--bot", type=str)
def show_win_rate_graph(bot: str) -> None:
    """
    Plot bar graph comparing the win rate of a specified bot against the baselines and the other bot.
    
    Params:
        bot (str) : name of bot

    Returns:
        None
    """

    baselines_win_rates: list[float] = []
    bot_win_rates: list[float] = []
    for baseline in BASELINE_NAMES:
        bot_win_rate = get_win_rate(bot, baseline)
        bot_win_rates.append(bot_win_rate)
        baselines_win_rates.append(1 - bot_win_rate)

    other_bot = "blitz" if bot == "siege" else "siege"

    index = BASELINE_NAMES
    index.append(other_bot)

    bot_win_rate = get_win_rate(bot, other_bot)
    bot_win_rates.append(bot_win_rate)
    baselines_win_rates.append(1 - bot_win_rate)

    df = pd.DataFrame({
        f"{bot} win rate": bot_win_rates,
        "opponent win rate": baselines_win_rates
    },index=index)
    df.plot.bar(rot=0, color=["lightseagreen", "tomato"])
    plt.show()

@main.command()
@click.option("--bot", type=str)
def show_average_points_graph(bot: str) -> None:
    """
    Plot bar graph comparing the win rate average score of a specified bot against the baselines.
    
    Params:
        bot (str) : name of bot

    Returns:
        None
    """

    path = f"../experiments/{bot}_tournament.csv"

    avg_baselines_scores: list[float] = []
    avg_bot_scores: list[float] = []
    for baseline in BASELINE_NAMES:
        avg_bot_score = mean(get_bot_scores(bot, baseline, path))
        avg_bot_scores.append(avg_bot_score)
        avg_baselines_scores.append(mean(get_bot_scores(baseline, bot, path)))

    other_bot = "blitz" if bot == "siege" else "siege"

    index = BASELINE_NAMES
    index.append(other_bot)

    avg_bot_score = mean(get_bot_scores(bot, other_bot, path))
    avg_bot_scores.append(avg_bot_score)
    avg_baselines_scores.append(mean(get_bot_scores(other_bot, bot, path)))

    df = pd.DataFrame({
        f"average score - {bot}": avg_bot_scores,
        "average score - opponent": avg_baselines_scores
    },index=index)

    df.plot.bar(rot=0, color=["gold", "crimson"])
    plt.show()

def get_bot_scores(bot: str, opponent: str, tournament_path: str) -> list[int]:
    """
    Returns a list of the scores the bot earned against its opponent in each game.
    
    Params:
        bot (str): name of bot whose scores we want
        opponent (str): name of opponent bot 
        tournament_path (str): path to file that holds the tournament data where the bot and opponent played
    
    Returns:
        list[int]: list of scores the bot achieved in each game
    """
    
    df = pd.read_csv(tournament_path)

    winning_games = df[(df["winner"] == bot) & (df["loser"] == opponent)]
    losing_games  = df[(df["winner"] == opponent) & (df["loser"] == bot)]

    scores = pd.concat([
        winning_games["winner_score"],
        losing_games["loser_score"]
    ]).sort_index()

    return scores.tolist()

BASELINE_NAMES = ["BullyBot", "RandBot", "RdeepBot"]

def run_bionmial_test_baselines() -> dict[list[float]]:
    results: dict[tuple, list[float]] = {}

    for baseline in BASELINE_NAMES:
        for opp_baseline in BASELINE_NAMES:
            if baseline == opp_baseline:
                continue
            
            wins = get_wins_count(baseline, opp_baseline)
            results[(baseline, opp_baseline)] = stats.binomtest(wins, 3000, alternative="greater").pvalue

    return results

def run_bionmial_test(bot: str) -> dict[tuple, list[float]]:
    # Are the bot's win rates against the baselines + other bot meaningfully above 50%
    results: dict[tuple, list[float]] = {}

    for baseline in BASELINE_NAMES:
        wins = get_wins_count(bot, baseline)
        results[(bot, baseline)] = stats.binomtest(wins, 3000, alternative="greater").pvalue

    other_bot = "blitz" if bot == "siege" else "siege"
    wins = get_wins_count(bot, other_bot)
    results[(bot, other_bot)] = stats.binomtest(wins, 3000, alternative="greater").pvalue
    
    return results

def correct_p_values() -> dict[tuple, list[float]]:
    results = run_bionmial_test_baselines() | run_bionmial_test("siege") | run_bionmial_test("blitz")
    p_values = list(results.values())

    corrected_p_values = stats.false_discovery_control(p_values)
   
    i = 0
    for pair in results.keys():
        results[pair] = corrected_p_values[i]
        i += 1
    
    return results

def get_bot_win_rates() -> dict[tuple, list[float]]:
    """
    Get the win rate of each pair of bots.

    Returns:
        dict[tuple, list[float]]: dictionary of bot pairs and the win rate of the first bot in the pair
    """
    win_rates: dict[tuple, list[float]] = {}

    for baseline in BASELINE_NAMES:
        for opp_baseline in BASELINE_NAMES:
            if baseline == opp_baseline:
                continue
            
            win_rate = get_win_rate(baseline, opp_baseline)
            win_rates[(baseline, opp_baseline)] = win_rate

    for baseline in BASELINE_NAMES:
        win_rate = get_win_rate("siege", baseline)
        win_rates[("siege", baseline)] = win_rate

        win_rate = get_win_rate("blitz", baseline)
        win_rates[("blitz", baseline)] = win_rate
    
    return win_rates

def get_average_win_rates(win_rates: dict[tuple, list[float]]) -> dict[str, float]:
    """
    Computes the average win rate of each of the bots
    
    Params:

    Returns:

    """
    avg_win_rates = {}
    prev_lead = None
    prev_sum = 0
    for pair, win_rate in win_rates.items():
        if pair[0] != prev_lead and prev_lead:
            avg_win_rates[prev_lead] = prev_sum / 3 
            prev_sum = win_rate
        else:
            prev_sum += win_rate
        
        prev_lead = pair[0]
    return avg_win_rates

def rank_bots() -> pd.DataFrame:
    """
    Ranks the bots with significant win rate differences based on their win rate
    
    Returns:
        pd.DataFrame: a sorted pandas data frame with columns 'winner', 'loser' and 'win_rate' 
    """      
    better_than: dict[tuple, float] = {}
    corrected_p_values = correct_p_values()

    for pair, p_value in corrected_p_values.items():
        win_rate_bot_1 = get_win_rate(pair[0], pair[1])
        win_rate_bot_2 = get_win_rate(pair[1], pair[0])

        if p_value <= 0.05 and win_rate_bot_1 > win_rate_bot_2:
            better_than[pair] = get_win_rate(pair[0], pair[1])

    better_than_df = pd.DataFrame([(winner, loser, win_rate) for (winner, loser), win_rate in better_than.items()],
        columns=["winner", "loser", "win_rate"])
    
    better_than_df = better_than_df.sort_values("win_rate", ascending=False)
    
    return better_than_df

def infer_performance_rankings():
    G = nx.DiGraph()

    ranking = rank_bots()

    for _, row in ranking.iterrows():
        G.add_edge(row["winner"], row["loser"], weight=row["win_rate"])

    results = {
        bot: set(nx.descendants(G, bot)) 
        for bot in G.nodes
    }

    return results

@main.command()
def final_experiment():
    performance_rankings = infer_performance_rankings()
    print(performance_rankings)

if __name__ == "__main__":
    main()
