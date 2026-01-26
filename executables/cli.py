import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import networkx as nx
from statistics import mean

import click

from schnapsen.bots import RandBot, BullyBot

from schnapsen.bots.blitz_siege import Blitz, Siege

from schnapsen.game import (Bot, SchnapsenGamePlayEngine)

from schnapsen.bots.rdeep import RdeepBot

@click.group()
def main() -> None:
    """Various Schnapsen Game Examples"""

BASELINES: list[Bot] = [RandBot(random.Random(81), "RandBot"), BullyBot(random.Random(24), "BullyBot"), RdeepBot(4, 10, random.Random(1221), "RdeepBot")]
def play_pair(
    engine: SchnapsenGamePlayEngine,
    bot1: Bot,
    bot2: Bot,
    games: int,
    all_games: dict,
) -> None:
    """
    Play games between two bots with role alternation and store results.

    Params:
        engine (SchnapsenGamePlayEngine): the schnapsen game engine required to play the games between the bots
        bot1 (Bot): first bot in tournament
        bot2 (Bot): second bot in tournament
        games (int): number of total games between bot1 and bot2
        all_games (dict): dictionary where the tournament data is stored
    
    Returns:
        None
    """
    for game in range(games // 2):
        for leader, follower in [(bot1, bot2), (bot2, bot1)]:
            leader_name = leader.get_name()
            follower_name = follower.get_name()
            state = engine.play_game(leader, follower, random.Random(game))
            all_games[leader_name][len(all_games[leader_name]) + 1] = state
            all_games[follower_name][len(all_games[follower_name]) + 1] = state

def save_tournament(bot_name: str, games: dict, path: str) -> None:
    """
    Save a bot's tournament data to .csv file
    
    Params:
        bot_name (str): name of bot whose data should be stored
        games (dict): games data
        path (str): path to file where the tournmanet data should be saved

    Returns:
        None
    """
    df = pd.DataFrame.from_dict(games[bot_name], orient="index")
    df.to_csv(path, index=False)


@main.command()
@click.option("--games", default=3000, show_default=True, type=int)
def get_tournament_data(games: int) -> None:
    """
    Stores scores of each game the bots played against the baseline and each other in separate csv files.
    
    Params:
        games (int): number of total games

    Returns:
        None
    """
    BASE_PATH = "../experiments"

    blitz = Blitz("blitz")
    siege = Siege("siege")

    all_bots = BASELINES + [blitz, siege]
    engine = SchnapsenGamePlayEngine()

    # init empty dictionary with bot names as keys
    all_games = {bot.get_name(): {} for bot in all_bots}

    for i, bot1 in enumerate(all_bots):
        for bot2 in all_bots[i+1:]:
            play_pair(
                engine,
                bot1,
                bot2,
                games,
                all_games,
            )

    save_tournament("blitz", all_games, f"{BASE_PATH}/blitz_tournament.csv")
    save_tournament("siege", all_games, f"{BASE_PATH}/siege_tournament.csv")

    for baseline in BASELINES:
        baseline_name = baseline.get_name()
        save_tournament(
            baseline_name,
            all_games,
            f"{BASE_PATH}/{baseline_name}_tournament.csv",
        )

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

def run_bionmial_test_baselines() -> dict[tuple, float]:
    """
    Run a binomial test comparing the win rates of baseline bots with the alternative hypothesis p > 0.5
    
    Returns:
        dict[tuple, float]: dictionary of the pairs of bots and their resulting p-values
    """
    results: dict[tuple, float] = {}

    for baseline in BASELINE_NAMES:
        for opp_baseline in BASELINE_NAMES:
            if baseline == opp_baseline:
                continue
            
            wins = get_wins_count(baseline, opp_baseline)
            # calculate total games from the CSV data
            tournament_path = f"../experiments/{baseline}_tournament.csv"
            df = pd.read_csv(tournament_path)
            games_against_opponent = df[((df['winner'] == baseline) & (df['loser'] == opp_baseline)) | 
                                          ((df['winner'] == opp_baseline) & (df['loser'] == baseline))]
            total_games = games_against_opponent.shape[0]
            
            results[(baseline, opp_baseline)] = stats.binomtest(wins, total_games, alternative="greater").pvalue

    return results

def run_bionmial_test(bot: str) -> dict[tuple, float]:
    """
    Run a binomial test comparing the win rates of blitz and siege against the baselines and each other with the alternative hypothesis p > 0.5
    
    Returns:
        dict[tuple, float]: dictionary of the pairs of bots and their resulting p-values
    """
    results: dict[tuple, float] = {}
    
    tournament_path = f"../experiments/{bot}_tournament.csv"
    df = pd.read_csv(tournament_path)

    for baseline in BASELINE_NAMES:
        wins = get_wins_count(bot, baseline)
        # calculate total games from the CSV data
        games_against_opponent = df[((df['winner'] == bot) & (df['loser'] == baseline)) | 
                                      ((df['winner'] == baseline) & (df['loser'] == bot))]
        total_games = games_against_opponent.shape[0]
        
        results[(bot, baseline)] = stats.binomtest(wins, total_games, alternative="greater").pvalue

    # run binomial test between siege and blitz
    other_bot = "blitz" if bot == "siege" else "siege"
    wins = get_wins_count(bot, other_bot)
    # calculate total games from the CSV data
    games_against_opponent = df[((df['winner'] == bot) & (df['loser'] == other_bot)) | 
                                  ((df['winner'] == other_bot) & (df['loser'] == bot))]
    total_games = games_against_opponent.shape[0]
    
    results[(bot, other_bot)] = stats.binomtest(wins, total_games, alternative="greater").pvalue
    
    return results

def correct_p_values() -> dict[tuple, float]:
    """
    Correct previous p-values
    
    Returns:
        dict[tuple, float]: dictionary of the pairs of bots and their corrected p-values
    """
    results = run_bionmial_test_baselines() | run_bionmial_test("siege") | run_bionmial_test("blitz")
    
    # also test baselines against siege and blitz, not just siege and blits against baselines
    for baseline in BASELINE_NAMES:
        for bot in ["siege", "blitz"]:
            wins = get_wins_count(baseline, bot)
            # calculate total games from the CSV data
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

        # only count a bot as better than another if 1. it has statistical significance and 2. its win rate is higher than the other one's win rate
        if p_value <= 0.05 and win_rate_bot_1 > win_rate_bot_2:
            better_than[pair] = get_win_rate(pair[0], pair[1])

    better_than_df = pd.DataFrame([(winner, loser, win_rate) for (winner, loser), win_rate in better_than.items()],
        columns=["winner", "loser", "win_rate"])
    
    better_than_df = better_than_df.sort_values("win_rate", ascending=False)
    
    return better_than_df

def infer_performance_rankings() -> tuple[dict, nx.DiGraph]:
    """
    Create a graph of the ranking
    
    Returns:
        tuple[dict, DiGraph]: the bot ranking in dictionary form and a directed graph of the ranking
    """
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
def final_experiment() -> None:
    """
    Runs the final experiment: binomial test, correct p-values and create final bot ranking.

    Returns:
        None
    """
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
def show_ranking() -> None:
    """
    Plot directed graph of bot ranking

    Returns:
        None
    """
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
