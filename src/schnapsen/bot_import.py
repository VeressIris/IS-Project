import random
def import_rdeep():
    from schnapsen.bots import RdeepBot
    rdeep = RdeepBot(4, 10, random.Random(1221))
    return rdeep