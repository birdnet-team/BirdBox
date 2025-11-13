#################################################################################################################
### This file contains the configuration for the inference as well as for the evaluation of the detections    ###
#################################################################################################################

### Has to be adapted if the underlying dataset is changed ###
# DATASET_NAME = 'Hawaii'
DATASET_NAME = 'Hawaii_subset'
# DATASET_NAME = 'Northeastern-US'
# DATASET_NAME = 'Northeastern-US_subset'
# DATASET_NAME = 'Southern-Sierra-Nevada'
# DATASET_NAME = 'Western-US'


# height and width of the spectrograms
HEIGHT_AND_WIDTH_IN_PIXELS:int = 256

CLIP_LENGTH:int = 3

PCEN_SEGMENT_LENGTH:int = 60


# this has to match the definition in the conf.yaml of which the model is trained with
if DATASET_NAME == 'Hawaii' or DATASET_NAME == 'Hawaii_subset':
    DATASOURCE_ABBREVIATION = 'UHH'

    ###################### for Hawaii-Dataset ######################
    ID_TO_EBIRD_CODES = {
        0 : "hawama",
        1 : "ercfra",
        2 : "jabwar",
        3 : "apapan",
        4 : "skylar",
        5 : "houfin",
        6 : "blkfra",
        7 : "yefcan",
        8 : "reblei",
        9 : "warwhe1",
        10 : "kalphe",
        11 : "elepai",
        12 : "hawhaw",
        13 : "melthr",
        14 : "iiwi",
        15 : "calqua",
        16 : "norcar",
        17 : "wiltur",
        18 : "chukar",
        19 : "palila",
        20 : "hawcre",
        21 : "comwax",
        22 : "hawpet1",
        23 : "barpet",
        24 : "akepa1",
        25 : "hawgoo",
        26 : "omao",
    }

    # this refers directly to ID_TO_EBIRD_CODES
    BIRD_COLORS = {
        0: (255, 69, 0),      # Bright red-orange
        1: (0, 255, 0),       # Bright green
        2: (30, 144, 255),    # Dodger blue
        3: (255, 215, 0),     # Gold
        4: (255, 105, 180),   # Hot pink
        5: (138, 43, 226),    # Blue violet
        6: (0, 255, 255),     # Cyan
        7: (255, 140, 0),     # Dark orange
        8: (50, 205, 50),     # Lime green
        9: (70, 130, 180),    # Steel blue
        10: (220, 20, 60),    # Crimson
        11: (0, 191, 255),    # Deep sky blue
        12: (255, 20, 147),   # Deep pink
        13: (154, 205, 50),   # Yellow green
        14: (255, 99, 71),    # Tomato
        15: (218, 112, 214),  # Orchid
        16: (255, 250, 205),  # Lemon chiffon
        17: (0, 206, 209),    # Dark turquoise
        18: (233, 150, 122),  # Dark salmon
        19: (186, 85, 211),   # Medium orchid
        20: (255, 182, 193),  # Light pink
        21: (144, 238, 144),  # Light green
        22: (173, 216, 230),  # Light blue
        23: (240, 230, 140),  # Khaki
        24: (102, 205, 170),  # Medium aquamarine
        25: (199, 21, 133),   # Medium violet-red
        26: (127, 255, 212),  # Aquamarine
    }

elif DATASET_NAME == 'Northeastern-US' or DATASET_NAME == 'Northeastern-US_subset':
    DATASOURCE_ABBREVIATION = 'SSW'

    ###################### for Northeastern-US-Dataset ######################
    ID_TO_EBIRD_CODES = {
        0 : "????",
        1 : "aldfly",
        2 : "amecro",
        3 : "amegfi",
        4 : "amered",
        5 : "houfin",
        6 : "amerob",
        7 : "amewoo",
        8 : "balori",
        9 : "bcnher",
        10 : "belkin1",
        11 : "bkbwar",
        12 : "bkcchi",
        13 : "blujay",
        14 : "bnhcow",
        15 : "boboli",
        16 : "norcar",
        17 : "brdowl",
        18 : "brncre",
        19 : "btnwar",
        20 : "buhvir",
        21 : "buwwar",
        22 : "cangoo",
        23 : "cedwax",
        24 : "chswar",
        25 : "comgra",
        26 : "comrav",
        27 : "comyel",
        28 : "coohaw",
        29 : "daejun",
        30 : "dowwoo",
        31 : "easblu",
        32 : "easkin",
        33 : "easpho",
        34 : "eastow",
        35 : "eawpew",
        36 : "eursta",
        37 : "gockin",
        38 : "grbher3",
        39 : "grcfly",
        40 : "grycat",
        41 : "haiwoo",
        42 : "herthr",
        43 : "hoowar",
        44 : "houwre",
        45 : "killde",
        46 : "mallar3",
        47 : "moudov",
        48 : "naswar",
        49 : "norfli",
        50 : "norwat",
        51 : "ovenbi1",
        52 : "pilwoo",
        53 : "pinsis",
        54 : "purfin",
        55 : "rebnut",
        56 : "rebwoo",
        57 : "redcro",
        58 : "reevir1",
        59 : "rewbla",
        60 : "ribgul",
        61 : "robgro",
        62 : "ruckin",
        63 : "rusbla",
        64 : "scatan",
        65 : "snogoo",
        66 : "solsan",
        67 : "sonspa",
        68 : "swaspa",
        69 : "treswa",
        70 : "tuftit",
        71 : "tunswa",
        72 : "veery",
        73 : "warvir",
        74 : "whbnut",
        75 : "whtspa",
        76 : "wooduc",
        77 : "woothr",
        78 : "yebsap",
        79 : "yelwar",
        80 : "yerwar",
        81 : "yetvir",
    }

    # this refers directly to ID_TO_EBIRD_CODES
    BIRD_COLORS = {
        0:  (255, 69, 0),      # OrangeRed
        1:  (0, 255, 0),       # Lime
        2:  (30, 144, 255),    # DodgerBlue
        3:  (255, 215, 0),     # Gold
        4:  (255, 105, 180),   # HotPink
        5:  (138, 43, 226),    # BlueViolet
        6:  (0, 255, 255),     # Cyan
        7:  (255, 140, 0),     # DarkOrange
        8:  (50, 205, 50),     # LimeGreen
        9:  (70, 130, 180),    # SteelBlue
        10:  (220, 20, 60),     # Crimson
        11:  (0, 191, 255),     # DeepSkyBlue
        12:  (255, 20, 147),    # DeepPink
        13:  (154, 205, 50),    # YellowGreen
        14:  (255, 99, 71),     # Tomato
        15:  (218, 112, 214),   # Orchid
        16:  (255, 250, 205),   # LemonChiffon
        17:  (0, 206, 209),     # DarkTurquoise
        18:  (233, 150, 122),   # DarkSalmon
        19:  (186, 85, 211),    # MediumOrchid
        20:  (255, 182, 193),   # LightPink
        21:  (144, 238, 144),   # LightGreen
        22:  (173, 216, 230),   # LightBlue
        23:  (240, 230, 140),   # Khaki
        24:  (102, 205, 170),   # MediumAquamarine
        25:  (199, 21, 133),    # MediumVioletRed
        26:  (127, 255, 212),   # Aquamarine
        27:  (255, 0, 0),       # Red
        28:  (0, 128, 0),       # Green
        29:  (0, 0, 255),       # Blue
        30:  (255, 255, 0),     # Yellow
        31:  (255, 0, 255),     # Magenta
        32:  (0, 255, 127),     # SpringGreen
        33:  (255, 165, 0),     # Orange
        34:  (138, 221, 45),    # Chartreuse
        35:  (70, 240, 240),    # Turquoise
        36:  (0, 128, 255),     # Azure
        37:  (204, 0, 204),     # Purple
        38:  (255, 204, 153),   # Peach
        39:  (255, 51, 153),    # Pink
        40:  (153, 204, 50),    # YellowGreen2
        41:  (0, 255, 200),     # Aqua
        42:  (255, 102, 0),     # BrightOrange
        43:  (255, 0, 127),     # Rose
        44:  (64, 224, 208),    # Turquoise2
        45:  (210, 105, 30),    # Chocolate
        46:  (176, 224, 230),   # PowderBlue
        47:  (255, 228, 181),   # Moccasin
        48:  (240, 128, 128),   # LightCoral
        49:  (255, 99, 255),    # VioletPink
        50:  (124, 252, 0),     # LawnGreen
        51:  (255, 239, 0),     # BrightYellow
        52:  (139, 0, 139),     # DarkMagenta
        53:  (0, 139, 139),     # DarkCyan
        54:  (65, 105, 225),    # RoyalBlue
        55:  (205, 92, 92),     # IndianRed
        56:  (255, 248, 220),   # Cornsilk
        57:  (34, 139, 34),     # ForestGreen
        58:  (255, 160, 122),   # LightSalmon
        59:  (221, 160, 221),   # Plum
        60:  (46, 139, 87),     # SeaGreen
        61:  (210, 180, 140),   # Tan
        62:  (123, 104, 238),   # MediumSlateBlue
        63:  (255, 218, 185),   # PeachPuff
        64:  (32, 178, 170),    # LightSeaGreen
        65:  (240, 230, 250),   # Lavender
        66:  (255, 228, 225),   # MistyRose
        67:  (189, 183, 107),   # DarkKhaki
        68:  (0, 250, 154),     # MediumSpringGreen
        69:  (218, 165, 32),    # Goldenrod
        70:  (0, 100, 0),       # DarkGreen
        71:  (255, 69, 200),    # NeonPink
        72:  (233, 150, 255),   # LightViolet
        73:  (135, 206, 250),   # LightSkyBlue
        74:  (222, 184, 135),   # Burlywood
        75:  (100, 149, 237),   # CornflowerBlue
        76:  (152, 251, 152),   # PaleGreen
        77:  (250, 128, 114),   # Salmon
        78:  (244, 164, 96),    # SandyBrown
        79:  (0, 191, 100),     # TealGreen
        80:  (147, 112, 219),   # MediumPurple
        81:  (255, 255, 224),   # LightYellow
    }

elif DATASET_NAME == 'Southern-Sierra-Nevada':
    DATASOURCE_ABBREVIATION = 'HSN'

    ###################### for Southern-Sierra-Nevada-Dataset ######################
    ID_TO_EBIRD_CODES = {
        0 : "????",
        1 : "amepip",
        2 : "amerob",
        3 : "brebla",
        4 : "casfin",
        5 : "clanut",
        6 : "daejun",
        7 : "dusfly",
        8 : "foxspa",
        9 : "gcrfin",
        10 : "herthr",
        11 : "mallar3",
        12 : "moublu",
        13 : "mouchi",
        14 : "norfli",
        15 : "orcwar",
        16 : "rocwre",
        17 : "sposan",
        18 : "warvir",
        19 : "whcspa",
        20 : "yelwar",
        21 : "yerwar",
    }

    # this refers directly to ID_TO_EBIRD_CODES
    BIRD_COLORS = {
        0: (255, 69, 0),      # Bright red-orange
        1: (0, 255, 0),       # Bright green
        2: (30, 144, 255),    # Dodger blue
        3: (255, 215, 0),     # Gold
        4: (255, 105, 180),   # Hot pink
        5: (138, 43, 226),    # Blue violet
        6: (0, 255, 255),     # Cyan
        7: (255, 140, 0),     # Dark orange
        8: (50, 205, 50),     # Lime green
        9: (70, 130, 180),    # Steel blue
        10: (220, 20, 60),    # Crimson
        11: (0, 191, 255),    # Deep sky blue
        12: (255, 20, 147),   # Deep pink
        13: (154, 205, 50),   # Yellow green
        14: (255, 99, 71),    # Tomato
        15: (218, 112, 214),  # Orchid
        16: (255, 250, 205),  # Lemon chiffon
        17: (0, 206, 209),    # Dark turquoise
        18: (233, 150, 122),  # Dark salmon
        19: (186, 85, 211),   # Medium orchid
        20: (255, 182, 193),  # Light pink
        21: (144, 238, 144),  # Light green
    }

elif DATASET_NAME == 'Western-US':
    DATASOURCE_ABBREVIATION = 'SNE'

    ###################### for Western-US-Dataset ######################
    ID_TO_EBIRD_CODES = {
        0 : "acowoo",
        1 : "amegfi",
        2 : "amerob",
        3 : "annhum",
        4 : "batpig1",
        5 : "bewwre",
        6 : "bkhgro",
        7 : "bnhcow",
        8 : "brncre",
        9 : "btywar",
        10 : "cangoo",
        11 : "casfin",
        12 : "casvir",
        13 : "chbchi",
        14 : "comrav",
        15 : "daejun",
        16 : "dusfly",
        17 : "evegro",
        18 : "foxspa",
        19 : "gnttow",
        20 : "gockin",
        21 : "hamfly",
        22 : "herthr",
        23 : "herwar",
        24 : "houwre",
        25 : "hutvir",
        26 : "lazbun",
        27 : "linspa",
        28 : "macwar",
        29 : "mouchi",
        30 : "moudov",
        31 : "mouqua",
        32 : "naswar",
        33 : "norfli",
        34 : "olsfly",
        35 : "orcwar",
        36 : "pasfly",
        37 : "pinsis",
        38 : "purfin",
        39 : "rebnut",
        40 : "redcro",
        41 : "ruckin",
        42 : "spotow",
        43 : "stejay",
        44 : "swathr",
        45 : "towsol",
        46 : "towwar",
        47 : "vesspa",
        48 : "warvir",
        49 : "westan",
        50 : "wewpew",
        51 : "whcspa",
        52 : "whhwoo",
        53 : "wilsap",
        54 : "wlswar",
        55 : "yerwar",
    }

    # this refers directly to ID_TO_EBIRD_CODES
    BIRD_COLORS = {
        0:  (255, 69, 0),      # OrangeRed
        1:  (0, 255, 0),       # Lime
        2:  (30, 144, 255),    # DodgerBlue
        3:  (255, 215, 0),     # Gold
        4:  (255, 105, 180),   # HotPink
        5:  (138, 43, 226),    # BlueViolet
        6:  (0, 255, 255),     # Cyan
        7:  (255, 140, 0),     # DarkOrange
        8:  (50, 205, 50),     # LimeGreen
        9:  (70, 130, 180),    # SteelBlue
        10:  (220, 20, 60),     # Crimson
        11:  (0, 191, 255),     # DeepSkyBlue
        12:  (255, 20, 147),    # DeepPink
        13:  (154, 205, 50),    # YellowGreen
        14:  (255, 99, 71),     # Tomato
        15:  (218, 112, 214),   # Orchid
        16:  (255, 250, 205),   # LemonChiffon
        17:  (0, 206, 209),     # DarkTurquoise
        18:  (233, 150, 122),   # DarkSalmon
        19:  (186, 85, 211),    # MediumOrchid
        20:  (255, 182, 193),   # LightPink
        21:  (144, 238, 144),   # LightGreen
        22:  (173, 216, 230),   # LightBlue
        23:  (240, 230, 140),   # Khaki
        24:  (102, 205, 170),   # MediumAquamarine
        25:  (199, 21, 133),    # MediumVioletRed
        26:  (127, 255, 212),   # Aquamarine
        27:  (255, 0, 0),       # Red
        28:  (0, 128, 0),       # Green
        29:  (0, 0, 255),       # Blue
        30:  (255, 255, 0),     # Yellow
        31:  (255, 0, 255),     # Magenta
        32:  (0, 255, 127),     # SpringGreen
        33:  (255, 165, 0),     # Orange
        34:  (138, 221, 45),    # Chartreuse
        35:  (70, 240, 240),    # Turquoise
        36:  (0, 128, 255),     # Azure
        37:  (204, 0, 204),     # Purple
        38:  (255, 204, 153),   # Peach
        39:  (255, 51, 153),    # Pink
        40:  (153, 204, 50),    # YellowGreen2
        41:  (0, 255, 200),     # Aqua
        42:  (255, 102, 0),     # BrightOrange
        43:  (255, 0, 127),     # Rose
        44:  (64, 224, 208),    # Turquoise2
        45:  (210, 105, 30),    # Chocolate
        46:  (176, 224, 230),   # PowderBlue
        47:  (255, 228, 181),   # Moccasin
        48:  (240, 128, 128),   # LightCoral
        49:  (255, 99, 255),    # VioletPink
        50:  (124, 252, 0),     # LawnGreen
        51:  (255, 239, 0),     # BrightYellow
        52:  (139, 0, 139),     # DarkMagenta
        53:  (0, 139, 139),     # DarkCyan
        54:  (65, 105, 225),    # RoyalBlue
        55:  (205, 92, 92),     # IndianRed
    }

else:
    raise ValueError(f"Invalid dataset name: {DATASET_NAME}")
