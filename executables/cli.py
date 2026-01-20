import random
import pathlib
import pandas as pd
import matplotlib.pyplot as plt

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

def run_tournament(games: int, bot: Bot) -> dict[int, dict[str, any]]:
    """
    Plays 'games' amount of games with 'bot' against RandBot, BullyBot and BullyBot and returns the results of each game.

    Params:
        games (int): number of games to play
        bot (Bot): the bot that plays against the baseline bots

    Returns:
        dict[int, dict[str, any]]: The name of the winner, score of the winner, name of loser and score of loser in each match
    """
    engine = SchnapsenGamePlayEngine()
    
    baselines: list[Bot] = [RandBot(random.Random(81), "RandBot"), BullyBot(random.Random(24), "BullyBot"), RdeepBot(4, 10, random.Random(1221), "RdeepBot")]
    
    scores:dict[int, dict[str, any]] = {}

    game = 1
    for baseline in baselines:
        for i in range(games):
            state = engine.play_game(bot, baseline, random.Random(i))
            scores[game] = state
            game += 1
    
    return scores

@main.command()
@click.option("--games", default=2500, show_default=True, type=int)
def get_tournament_data(games: int) -> None:
    """
    Stores scores of each game the bots played against the baseline in separate csv files.
    
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

def get_win_rate(winner_name: str, opponent_name: str, tournament_path: str) -> float:
    """
    Calculates the win rate of a bot against another specified bot.
    
    Params:
        winner_name (str): name of winning bot
        opponent_name (str): name of opponent bot
        tournament_path (str): path to file that contains the tournament information of the winner bot
    
    Returns:
        float: Win rate of specified bot
    """
    df = pd.read_csv(tournament_path)
    games = df.shape[0] // 3 # divided by 3 because we have 3 baseline bots
    winning_games = df[(df['winner'] == winner_name) & (df['loser'] == opponent_name)] # get only games played against specified opponent
    wins = winning_games.shape[0]

    return wins / games

@main.command()
@click.option("--bot", type=str)
@click.option("--tournament-path", type=str)
def show_win_rate_graph(bot: str, tournament_path: str) -> None:
    """
    Plot bar graph comparing the win rate of a specified bot against the baselines.
    
    Params:
        bot (str) : name of bot
        tournament_path (str): path to file that contains the tournament information of the bot

    Returns:
        None
    """
    index: list[str] = ["RandBot", "BullyBot", "RdeepBot"]
    
    baselines_win_rates: list[float] = []
    bot_win_rates: list[float] = []
    for baseline in index:
        bot_win_rate = get_win_rate(bot, baseline, tournament_path)
        bot_win_rates.append(bot_win_rate)
        baselines_win_rates.append(1 - bot_win_rate)

    df = pd.DataFrame({
        f"{bot} win rate": bot_win_rates,
        "opponent win rate": baselines_win_rates
    },index=index )
    df.plot.bar(rot=0, color=["green", "orange"])
    plt.show()

if __name__ == "__main__":
    main()
