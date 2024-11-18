INPUT_SIZE = (704, 704, 1)

CHANNELS = 3

CLASSES = 15
NEW_CLASSES = 10

STATE = 42

CLASS_DICT = {
        0: "background",
        1: "lumen",
        2: "guidewire",
        3: "intima",
        4: "lipid",
        5: "calcium",
        6: "media",
        7: "catheter",
        8: "sidebranch",
        9: "rthrombus",
        10: "wthrombus",
        11: "dissection",
        12: "rupture",
        13: "healed plaque",
        14: "neovascularisation",
    }

NEW_CLASS_DICT = {
        0: "background",
        1: "lumen",
        2: "guidewire",
        3: "intima",
        4: "lipid",
        5: "calcium",
        6: "media",
        7: "catheter",
        8: "sidebranch",
        9: "healed plaque",
    }


CLASS_COLORS = {
    0: (0,0,0),
    1: (255,0,0),
    2: (63,63,63),
    3: (0,0,255),
    4: (255,255,0),
    5: (225,225,225),
    6: (225,0,225),
    7: (146,0,0),
    8: (255,123,0),
    9: (230,141,230),
    10: (0,225,225),
    11: (65,135,100),
    12: (208,190,161),
    13: (0,255,255),
    14: (162,162,162),
}

NEW_CLASS_COLORS = {
    0: (0,0,0),
    1: (255,0,0),
    2: (63,63,63),
    3: (0,0,255),
    4: (255,255,0),
    5: (225,225,225),
    6: (225,0,225),
    7: (146,0,0),
    8: (255,123,0),
    9: (0,255,255),
}
