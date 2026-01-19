import random
def import_rdeep():
    from src.schnapsen.bots import RdeepBot
    rdeep = RdeepBot(4, 10, random.Random)
    return rdeep