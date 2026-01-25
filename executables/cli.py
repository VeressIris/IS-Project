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

BASELINES: list[Bot] = [RandBot(random.Random(81), "RandBot"), BullyBot(random.Random(24), "BullyBot"), RdeepBot(4, 10, random.Random(1221), "RdeepBot")]
def run_tournament(games: int, bot: Bot, other_bot: Bot = None) -> dict[int, dict[str, any]]:
    """
    Plays 'games' amount of games with 'bot' against RandBot, BullyBot and RdeepBot and our other bot.
    Each matchup plays games with alternating leader/follower roles, using consistent seeds for the same bot pairs.

    Params:
        games (int): number of games to play (total games will be 2*games due to role swapping)
        bot (Bot): the bot that plays against the baseline bots
        other_bot (Bot): the other custom bot (blitz or siege) to play against

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
            # Play with bot as leader
            state = engine.play_game(bot, baseline, random.Random(i))
            scores[game] = state
            game += 1
            # Play with bot as follower (same seed ensures same card dealing)
            state = engine.play_game(baseline, bot, random.Random(i))
            scores[game] = state
            game += 1
    
    if other_bot is not None:
        for i in range(games):
            # Play with bot as leader
            state = engine.play_game(bot, other_bot, random.Random(i))
            scores[game] = state
            game += 1
            # Play with bot as follower (same seed ensures same card dealing)
            state = engine.play_game(other_bot, bot, random.Random(i))
            scores[game] = state
            game += 1

    return scores

@main.command()
@click.option("--games", default=1500, show_default=True, type=int)
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
    
    # Create ALL bot instances ONCE - these will be shared across all games
    blitz = Blitz("blitz")
    siege = Siege("siege")
    baselines_dict = {
        "RandBot": RandBot(random.Random(81), "RandBot"),
        "BullyBot": BullyBot(random.Random(24), "BullyBot"),
        "RdeepBot": RdeepBot(4, 10, random.Random(1221), "RdeepBot")
    }
    
    all_bots = {"blitz": blitz, "siege": siege, **baselines_dict}
    engine = SchnapsenGamePlayEngine()
    
    # Store all games for each bot
    all_games = {name: {} for name in all_bots.keys()}
    
    # Play all unique pairings ONCE
    bot_names = list(all_bots.keys())
    for i, bot1_name in enumerate(bot_names):
        for bot2_name in bot_names[i+1:]:
            bot1 = all_bots[bot1_name]
            bot2 = all_bots[bot2_name]
            
            # Play games with role alternation
            for game_idx in range(games):
                # bot1 as leader
                state = engine.play_game(bot1, bot2, random.Random(game_idx))
                all_games[bot1_name][len(all_games[bot1_name]) + 1] = state
                all_games[bot2_name][len(all_games[bot2_name]) + 1] = state
                
                # bot1 as follower
                state = engine.play_game(bot2, bot1, random.Random(game_idx))
                all_games[bot1_name][len(all_games[bot1_name]) + 1] = state
                all_games[bot2_name][len(all_games[bot2_name]) + 1] = state
    
    # Save all tournament files
    df = pd.DataFrame.from_dict(all_games["blitz"], orient="index")
    df.to_csv(BLITZ_TOURNAMENTS, index=False)
    
    df = pd.DataFrame.from_dict(all_games["siege"], orient="index")
    df.to_csv(SIEGE_TOURNAMENTS, index=False)
    
    for baseline_name in baselines_dict.keys():
        path = f"../experiments/{baseline_name}_tournament.csv"
        df = pd.DataFrame.from_dict(all_games[baseline_name], orient="index")
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
    
    # get all games played against the specified opponent (both wins and losses)
    games_against_opponent = df[((df['winner'] == winner_name) & (df['loser'] == opponent_name)) | 
                                  ((df['winner'] == opponent_name) & (df['loser'] == winner_name))]
    total_games = games_against_opponent.shape[0]
    
    # count only the wins
    winning_games = df[(df['winner'] == winner_name) & (df['loser'] == opponent_name)]
    wins = winning_games.shape[0]

    return wins / total_games

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
def show_average_score_graph(bot: str) -> None:
    """
    Plot bar graph comparing the average score of a specified bot against the baselines and the other custom bot.
    
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
            # Calculate total games from the CSV data
            tournament_path = f"../experiments/{baseline}_tournament.csv"
            df = pd.read_csv(tournament_path)
            games_against_opponent = df[((df['winner'] == baseline) & (df['loser'] == opp_baseline)) | 
                                          ((df['winner'] == opp_baseline) & (df['loser'] == baseline))]
            total_games = games_against_opponent.shape[0]
            
            if total_games > 0:
                results[(baseline, opp_baseline)] = stats.binomtest(wins, total_games, alternative="greater").pvalue

    return results

def run_bionmial_test(bot: str) -> dict[tuple, list[float]]:
    # Are the bot's win rates against the baselines + other bot meaningfully above 50%
    results: dict[tuple, list[float]] = {}
    tournament_path = f"../experiments/{bot}_tournament.csv"
    df = pd.read_csv(tournament_path)

    for baseline in BASELINE_NAMES:
        wins = get_wins_count(bot, baseline)
        # Calculate total games from the CSV data
        games_against_opponent = df[((df['winner'] == bot) & (df['loser'] == baseline)) | 
                                      ((df['winner'] == baseline) & (df['loser'] == bot))]
        total_games = games_against_opponent.shape[0]
        
        if total_games > 0:
            results[(bot, baseline)] = stats.binomtest(wins, total_games, alternative="greater").pvalue

    other_bot = "blitz" if bot == "siege" else "siege"
    wins = get_wins_count(bot, other_bot)
    # Calculate total games from the CSV data
    games_against_opponent = df[((df['winner'] == bot) & (df['loser'] == other_bot)) | 
                                  ((df['winner'] == other_bot) & (df['loser'] == bot))]
    total_games = games_against_opponent.shape[0]
    
    if total_games > 0:
        results[(bot, other_bot)] = stats.binomtest(wins, total_games, alternative="greater").pvalue
    
    return results

def correct_p_values() -> dict[tuple, list[float]]:
    results = run_bionmial_test_baselines() | run_bionmial_test("siege") | run_bionmial_test("blitz")
    
    # Also test baselines against siege and blitz
    for baseline in BASELINE_NAMES:
        for bot in ["siege", "blitz"]:
            wins = get_wins_count(baseline, bot)
            # Calculate total games from the CSV data
            tournament_path = f"../experiments/{bot}_tournament.csv"
            df = pd.read_csv(tournament_path)
            games_against_opponent = df[((df['winner'] == baseline) & (df['loser'] == bot)) | 
                                          ((df['winner'] == bot) & (df['loser'] == baseline))]
            total_games = games_against_opponent.shape[0]
            
            results[(baseline, bot)] = stats.binomtest(wins, total_games, alternative="greater").pvalue
    
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

def infer_performance_rankings() -> tuple[dict, nx.DiGraph]:
    G = nx.DiGraph()

    ranking = rank_bots()

    for _, row in ranking.iterrows():
        G.add_edge(row["winner"], row["loser"], weight=row["win_rate"])

    results = {
        bot: set(nx.descendants(G, bot)) 
        for bot in G.nodes
    }

    return (results, G)

@main.command()
def final_experiment():
    print("Binomial test (H_0: win_prob(bot1, bot2) = 0.5; H_1: win_prob(bot1, bot2) > 0.5)")
    print("blitz & siege p-values:")
    print(run_bionmial_test("blitz"))
    print(run_bionmial_test("siege"), end='\n\n')
    
    print("baseline bots p-values:")
    print(run_bionmial_test_baselines(), end='\n\n')
    
    print("Corrected p-values:")
    print(correct_p_values(), end='\n\n')
    
    print("Final bot ranking:")
    performance_rankings = infer_performance_rankings()
    print(performance_rankings[0])
    print(f"\nIf you wish to visualize the ranking in the form of a graph, run `show-ranking` command.")

@main.command()
def show_ranking():
    _, G = infer_performance_rankings()

    # compute layer based on number of ancestors
    layers = {node: len(nx.ancestors(G, node)) for node in G.nodes}
    nx.set_node_attributes(G, layers, "layer")

    pos = nx.multipartite_layout(G, subset_key="layer")

    plt.figure(figsize=(8, 6))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=3750,
        arrows=True,
        node_color="lightseagreen",
        font_size=13
    )

    # show weights
    edge_labels = nx.get_edge_attributes(G, "weight")
    formatted_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=formatted_labels, font_size=12)

    plt.show()

if __name__ == "__main__":
    main()
