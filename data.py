MAX_X1 = 18.0
MAX_X2 = 400.0


def normalize(x1, x2):
    return x1 / MAX_X1, x2 / MAX_X2


TRAIN_RAW = [
    (5.2,  90,  1),
    (4.8,  75,  1),
    (6.1, 110,  1),
    (5.5,  95,  1),
    (4.5,  68,  1),
    (6.5, 130,  1),
    (13.5, 320, 2),
    (15.2, 380, 2),
    (12.8, 290, 2),
    (14.0, 350, 2),
    (11.5, 260, 2),
    (16.5, 400, 2),
]

UNKNOWN_RAW = [
    ("Городской хэтчбек",   5.5, 100),
    ("Спорткар",           14.5, 340),
    ("Кроссовер",           8.0, 180),
    ("Гибрид",              4.2,  85),
    ("Седан бизнес-класса", 10.5, 240),
]


def get_train_set():
    return [(*normalize(x1, x2), c) for x1, x2, c in TRAIN_RAW]


def get_unknown_set():
    return [(name, *normalize(x1, x2)) for name, x1, x2 in UNKNOWN_RAW]