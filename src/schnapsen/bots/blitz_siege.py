from schnapsen.game import (
    Bot,
    Move,
    PlayerPerspective,
    GamePhase,
    SchnapsenTrickScorer,
    RegularMove
)
from typing import Optional
from schnapsen import bot_import
import random

class HighCardsBot(Bot):
    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__()
        self.trick_scorer = SchnapsenTrickScorer
    def get_move(self, perspective: PlayerPerspective, leader_move: Move | None) -> Move:
        valid_moves = perspective.valid_moves()
        valid_regular_moves: list[RegularMove] = [move.as_regular_move() for move in perspective.valid_moves() if move.is_regular_move()]

        #if has marriage, plays marriage
        marriage: list[Move] = [move for move in valid_moves if move.is_marriage]
        if marriage:
            return marriage[0]

        # If the bot has cards of the trump suit, it plays the highest one
        trump_suit = perspective.get_trump_suit()
        trumps: list[RegularMove] = [move for move in valid_regular_moves if move.card.suit == trump_suit]
        if trumps:
            #choose highest one
            sorted_trumps = sorted(trumps, key=lambda m: self.trick_scorer.rank_to_points(m.cards[0].rank))
            return sorted_trumps[0]

        #if no trumps, choose the highest card overall
        sorted_moves = sorted(valid_regular_moves, key=lambda m: self.trick_scorer.rank_to_points(m.cards[0].rank))
        return sorted_moves[0]

class Blitz(Bot):
    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__()
        self.delegate_phase1 = HighCardsBot
        self.delegate_phase2 = bot_import()
    
    def get_move(self, perspective: PlayerPerspective, leader_move: Move | None) -> Move:
        """Get the move for the Bot.
        """
        if perspective.get_phase() == GamePhase.ONE:
            # delegates to HighCardsBot
            return self.delegate_phase1.get_move(perspective, leader_move)
        if perspective.get_phase() == GamePhase.TWO:
            #delegates to rdeep
            return self.delegate_phase2.get_move(perspective, leader_move)


class Siege(Bot):
    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__()
        self.delegate_phase1 = bot_import()
        self.delegate_phase2 = HighCardsBot
    
    def get_move(self, perspective: PlayerPerspective, leader_move: Move | None) -> Move:
        """Get the move for the Bot.
        """
        if perspective.get_phase() == GamePhase.ONE:
            # delegates to rdeep
            return self.delegate_phase1.get_move(perspective, leader_move)
        if perspective.get_phase() == GamePhase.TWO:
            #delegates to HighCardsBot
            return self.delegate_phase1.get_move(perspective, leader_move)
        