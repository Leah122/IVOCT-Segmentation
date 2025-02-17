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
        10: "total", # just for plotting purposes
    }

NEW_CLASS_DICT_CAPITAL = {
        0: "Background",
        1: "Lumen",
        2: "Guidewire",
        3: "Intima",
        4: "Lipid",
        5: "Calcium",
        6: "Media",
        7: "Catheter",
        8: "Sidebranch",
        9: "Healed plaque",
        10: "Total", # just for plotting purposes
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

PLOT_COLORS = ["cornflowerblue", "mediumpurple", "deeppink", "darkorange", "gold"]


UNCERTAIN = ['RUSMNRC0009_1_frame0_190', 'NLDRADB0086_1_frame400_055', 'NLDISALA0085_1_frame80_042', 'NLDISALA0044_1_frame390_173', 'NLDRADB0021_1_frame360_081', 'NLDZUYD0004_1_frame520_226', 'NLDRADB0088_2_frame400_056', 'NLDRADB0088_2_frame320_056', 'NLDZUYD0004_1_frame242_226', 'RUSMNRC0009_1_frame196_190', 'NLDRADB0071_1_frame280_225', 'NLDAMPH0028_1_frame236_126', 'NLDTERG0008_1_frame280_188', 'NLDISALA0012_1_frame232_253', 'NLDISALA0080_1_frame257_179', 'NLDTERG0008_1_frame275_188', 'NLDAMPH0075_1_frame0_164', 'NLDRADB0024_1_frame160_092', 'NLDISALA0097_1_frame40_052', 'NLDISALA0080_1_frame280_179', 'NLDISALA0080_1_frame80_179', 'NLDRADB0071_1_frame410_225', 'NLDISALA0090_1_frame280_047', 'NLDISALA0055_1_frame40_273', 'NLDISALA0080_1_frame250_179', 'NLDRADB0088_2_frame440_056', 'NLDZUYD0004_1_frame240_226', 'NLDISALA0078_1_frame200_222', 'NLDISALA0012_1_frame326_253', 'NLDRADB0054_1_frame200_258', 'RUSMNRC0009_1_frame200_190', 'NLDZUYD0004_1_frame280_226', 'NLDAMPH0028_1_frame177_126', 'NLDISALA0012_1_frame165_253', 'NLDISALA0012_1_frame100_253', 'NLDRADB0086_1_frame280_055', 'NLDRADB0080_1_frame120_123', 'NLDTERG0008_1_frame200_188', 'NLDRADB0058_2_frame360_142', 'NLDAMPH0028_1_frame235_126', 'NLDISALA0012_1_frame70_253', 'NLDISALA0080_1_frame240_179', 'NLDUMCG0002_1_frame340_197', 'NLDISALA0065_1_frame200_034', 'RUSMNRC0009_1_frame8_190', 'NLDRADB0003_1_frame80_068', 'NLDAMPH0039_1_frame120_109', 'NLDRADB0071_1_frame400_225', 'NLDISALA0012_1_frame140_253', 'NLDISALA0012_1_frame371_253', 'NLDRADB0058_2_frame400_142', 'NLDAMPH0075_1_frame360_164', 'NLDAMPH0039_1_frame80_109', 'NLDISALA0078_1_frame170_222', 'NLDISALA0085_1_frame160_042', 'RUSMNRC0009_1_frame197_190', 'NLDAMPH0039_1_frame240_109', 'NLDUMCG0002_1_frame400_197', 'NLDAMPH0039_1_frame160_109', 'NLDISALA0065_2_frame80_034', 'NLDISALA0012_1_frame48_253', 'NLDISALA0012_1_frame64_253', 'NLDISALA0023_1_frame360_269', 'NLDRADB0058_2_frame80_142', 'NLDAMPH0028_1_frame179_126']

METHODS_METRICS = {
    "entropy_tta": "Entropy (TTA)",
    "MI_tta": "MI (TTA)",
    "Brier_tta": "Brier (TTA)",
    "entropy_mc": "Entropy (MC)",
    "MI_mc": "MI (MC)",
    "Brier_mc": "Brier (MC)",
    "entropy_ens": "Entropy (Ens)",
    "MI_ens": "MI (Ens)",
    "Brier_ens": "Brier (Ens)",
}


