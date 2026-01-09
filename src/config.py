#################################################################################################################
### This file contains the configuration for the inference as well as for the evaluation of the detections.   ###
### All species mappings are defined as static data structures - no runtime mutations.                                 ###
#################################################################################################################

from typing import Dict
from pathlib import Path

# Global constants (same for all datasets)
HEIGHT_AND_WIDTH_IN_PIXELS: int = 256
CLIP_LENGTH: int = 3
PCEN_SEGMENT_LENGTH: int = 60

# Constants for Streamlit WebAPP
MAX_DURATION_SECONDS: int = 600  # 10 minutes
MAX_CONCURRENT_DETECTIONS: int = 6  # Maximum number of simultaneous sessions that run detection

# Species mapping configurations - all defined as static dictionaries
SPECIES_MAPPING = {
    'Hawaii': {
        'abbreviation': 'UHH',
        'id_to_ebird': {
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
        },
        'ebird_to_name': {
            "akepa1": "Loxops coccineus_Hawaii Akepa",
            "apapan": "Himatione sanguinea_Apapane",
            "barpet": "Hydrobates castro_Band-rumped Storm-Petrel",
            "blkfra": "Francolinus francolinus_Black Francolin",
            "calqua": "Callipepla californica_California Quail",
            "chukar": "Alectoris chukar_Chukar",
            "comwax": "Estrilda astrild_Common Waxbill",
            "elepai": "Chasiempis sandwichensis_Hawaii Elepaio",
            "ercfra": "Pternistis erckelii_Erckel's Francolin",
            "hawama": "Chlorodrepanis virens_Hawaii Amakihi",
            "hawcre": "Loxops mana_Hawaii Creeper",
            "hawgoo": "Branta sandvicensis_Hawaiian Goose",
            "hawhaw": "Buteo solitarius_Hawaiian Hawk",
            "hawpet1": "Pterodroma sandwichensis_Hawaiian Petrel",
            "houfin": "Haemorhous mexicanus_House Finch",
            "iiwi": "Drepanis coccinea_Iiwi",
            "jabwar": "Horornis diphone_Japanese Bush Warbler",
            "kalphe": "Lophura leucomelanos_Kalij Pheasant",
            "melthr": "Garrulax canorus_Chinese Hwamei",
            "norcar": "Cardinalis cardinalis_Northern Cardinal",
            "omao": "Myadestes obscurus_Omao",
            "palila": "Loxioides bailleui_Palila",
            "reblei": "Leiothrix lutea_Red-billed Leiothrix",
            "skylar": "Alauda arvensis_Eurasian Skylark",
            "warwhe1": "Zosterops japonicus_Warbling White-eye",
            "wiltur": "Meleagris gallopavo_Wild Turkey",
            "yefcan": "Crithagra mozambica_Yellow-fronted Canary",
        },
        'bird_colors': {
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
    },
    
    'Hawaii_subset': {
        'abbreviation': 'UHH',
        'id_to_ebird': {
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
        },
        'ebird_to_name': {
            "akepa1": "Loxops coccineus_Hawaii Akepa",
            "apapan": "Himatione sanguinea_Apapane",
            "barpet": "Hydrobates castro_Band-rumped Storm-Petrel",
            "blkfra": "Francolinus francolinus_Black Francolin",
            "calqua": "Callipepla californica_California Quail",
            "chukar": "Alectoris chukar_Chukar",
            "comwax": "Estrilda astrild_Common Waxbill",
            "elepai": "Chasiempis sandwichensis_Hawaii Elepaio",
            "ercfra": "Pternistis erckelii_Erckel's Francolin",
            "hawama": "Chlorodrepanis virens_Hawaii Amakihi",
            "hawcre": "Loxops mana_Hawaii Creeper",
            "hawgoo": "Branta sandvicensis_Hawaiian Goose",
            "hawhaw": "Buteo solitarius_Hawaiian Hawk",
            "hawpet1": "Pterodroma sandwichensis_Hawaiian Petrel",
            "houfin": "Haemorhous mexicanus_House Finch",
            "iiwi": "Drepanis coccinea_Iiwi",
            "jabwar": "Horornis diphone_Japanese Bush Warbler",
            "kalphe": "Lophura leucomelanos_Kalij Pheasant",
            "melthr": "Garrulax canorus_Chinese Hwamei",
            "norcar": "Cardinalis cardinalis_Northern Cardinal",
            "omao": "Myadestes obscurus_Omao",
            "palila": "Loxioides bailleui_Palila",
            "reblei": "Leiothrix lutea_Red-billed Leiothrix",
            "skylar": "Alauda arvensis_Eurasian Skylark",
            "warwhe1": "Zosterops japonicus_Warbling White-eye",
            "wiltur": "Meleagris gallopavo_Wild Turkey",
            "yefcan": "Crithagra mozambica_Yellow-fronted Canary",
        },
        'bird_colors': {
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
    },
    
    'Northeastern-US': {
        'abbreviation': 'SSW',
        'id_to_ebird': {
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
        },
        'ebird_to_name': {
            "????": "Unknown",
            "aldfly": "Empidonax alnorum_Alder Flycatcher",
            "amecro": "Corvus brachyrhynchos_American Crow",
            "amegfi": "Spinus tristis_American Goldfinch",
            "amered": "Setophaga ruticilla_American Redstart",
            "amerob": "Turdus migratorius_American Robin",
            "amewoo": "Scolopax minor_American Woodcock",
            "balori": "Icterus galbula_Baltimore Oriole",
            "bcnher": "Nycticorax nycticorax_Black-crowned Night-Heron",
            "belkin1": "Megaceryle alcyon_Belted Kingfisher",
            "bkbwar": "Setophaga fusca_Blackburnian Warbler",
            "bkcchi": "Poecile atricapillus_Black-capped Chickadee",
            "blujay": "Cyanocitta cristata_Blue Jay",
            "bnhcow": "Molothrus ater_Brown-headed Cowbird",
            "boboli": "Dolichonyx oryzivorus_Bobolink",
            "brdowl": "Strix varia_Barred Owl",
            "brncre": "Certhia americana_Brown Creeper",
            "btnwar": "Setophaga virens_Black-throated Green Warbler",
            "buhvir": "Vireo solitarius_Blue-headed Vireo",
            "buwwar": "Vermivora cyanoptera_Blue-winged Warbler",
            "cangoo": "Branta canadensis_Canada Goose",
            "cedwax": "Bombycilla cedrorum_Cedar Waxwing",
            "chswar": "Setophaga pensylvanica_Chestnut-sided Warbler",
            "comgra": "Quiscalus quiscula_Common Grackle",
            "comrav": "Corvus corax_Common Raven",
            "comyel": "Geothlypis trichas_Common Yellowthroat",
            "coohaw": "Accipiter cooperii_Cooper's Hawk",
            "daejun": "Junco hyemalis_Dark-eyed Junco",
            "dowwoo": "Dryobates pubescens_Downy Woodpecker",
            "easblu": "Sialia sialis_Eastern Bluebird",
            "easkin": "Tyrannus tyrannus_Eastern Kingbird",
            "easpho": "Sayornis phoebe_Eastern Phoebe",
            "eastow": "Pipilo erythrophthalmus_Eastern Towhee",
            "eawpew": "Contopus virens_Eastern Wood-Pewee",
            "eursta": "Sturnus vulgaris_European Starling",
            "gockin": "Regulus satrapa_Golden-crowned Kinglet",
            "grbher3": "Ardea herodias_Great Blue Heron",
            "grcfly": "Myiarchus crinitus_Great Crested Flycatcher",
            "grycat": "Dumetella carolinensis_Gray Catbird",
            "haiwoo": "Dryobates villosus_Hairy Woodpecker",
            "herthr": "Catharus guttatus_Hermit Thrush",
            "hoowar": "Setophaga citrina_Hooded Warbler",
            "houfin": "Haemorhous mexicanus_House Finch",
            "houwre": "Troglodytes aedon_House Wren",
            "killde": "Charadrius vociferus_Killdeer",
            "mallar3": "Anas platyrhynchos_Mallard",
            "moudov": "Zenaida macroura_Mourning Dove",
            "naswar": "Leiothlypis ruficapilla_Nashville Warbler",
            "norcar": "Cardinalis cardinalis_Northern Cardinal",
            "norfli": "Colaptes auratus_Northern Flicker",
            "norwat": "Parkesia noveboracensis_Northern Waterthrush",
            "ovenbi1": "Seiurus aurocapilla_Ovenbird",
            "pilwoo": "Dryocopus pileatus_Pileated Woodpecker",
            "pinsis": "Spinus pinus_Pine Siskin",
            "purfin": "Haemorhous purpureus_Purple Finch",
            "rebnut": "Sitta canadensis_Red-breasted Nuthatch",
            "rebwoo": "Melanerpes carolinus_Red-bellied Woodpecker",
            "redcro": "Loxia curvirostra_Red Crossbill",
            "reevir1": "Vireo olivaceus_Red-eyed Vireo",
            "rewbla": "Agelaius phoeniceus_Red-winged Blackbird",
            "ribgul": "Larus delawarensis_Ring-billed Gull",
            "robgro": "Pheucticus ludovicianus_Rose-breasted Grosbeak",
            "ruckin": "Corthylio calendula_Ruby-crowned Kinglet",
            "rusbla": "Euphagus carolinus_Rusty Blackbird",
            "scatan": "Piranga olivacea_Scarlet Tanager",
            "snogoo": "Anser caerulescens_Snow Goose",
            "solsan": "Tringa solitaria_Solitary Sandpiper",
            "sonspa": "Melospiza melodia_Song Sparrow",
            "swaspa": "Melospiza georgiana_Swamp Sparrow",
            "treswa": "Tachycineta bicolor_Tree Swallow",
            "tuftit": "Baeolophus bicolor_Tufted Titmouse",
            "tunswa": "Cygnus columbianus_Tundra Swan",
            "veery": "Catharus fuscescens_Veery",
            "warvir": "Vireo gilvus_Warbling Vireo",
            "whbnut": "Sitta carolinensis_White-breasted Nuthatch",
            "whtspa": "Zonotrichia albicollis_White-throated Sparrow",
            "wooduc": "Aix sponsa_Wood Duck",
            "woothr": "Hylocichla mustelina_Wood Thrush",
            "yebsap": "Sphyrapicus varius_Yellow-bellied Sapsucker",
            "yelwar": "Setophaga petechia_Yellow Warbler",
            "yerwar": "Setophaga coronata_Yellow-rumped Warbler",
            "yetvir": "Vireo flavifrons_Yellow-throated Vireo",
        },
        'bird_colors': {
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
    },
    
    'Northeastern-US_subset': {
        'abbreviation': 'SSW',
        'id_to_ebird': {
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
        },
        'ebird_to_name': {
            "????": "Unknown",
            "aldfly": "Empidonax alnorum_Alder Flycatcher",
            "amecro": "Corvus brachyrhynchos_American Crow",
            "amegfi": "Spinus tristis_American Goldfinch",
            "amered": "Setophaga ruticilla_American Redstart",
            "amerob": "Turdus migratorius_American Robin",
            "amewoo": "Scolopax minor_American Woodcock",
            "balori": "Icterus galbula_Baltimore Oriole",
            "bcnher": "Nycticorax nycticorax_Black-crowned Night-Heron",
            "belkin1": "Megaceryle alcyon_Belted Kingfisher",
            "bkbwar": "Setophaga fusca_Blackburnian Warbler",
            "bkcchi": "Poecile atricapillus_Black-capped Chickadee",
            "blujay": "Cyanocitta cristata_Blue Jay",
            "bnhcow": "Molothrus ater_Brown-headed Cowbird",
            "boboli": "Dolichonyx oryzivorus_Bobolink",
            "brdowl": "Strix varia_Barred Owl",
            "brncre": "Certhia americana_Brown Creeper",
            "btnwar": "Setophaga virens_Black-throated Green Warbler",
            "buhvir": "Vireo solitarius_Blue-headed Vireo",
            "buwwar": "Vermivora cyanoptera_Blue-winged Warbler",
            "cangoo": "Branta canadensis_Canada Goose",
            "cedwax": "Bombycilla cedrorum_Cedar Waxwing",
            "chswar": "Setophaga pensylvanica_Chestnut-sided Warbler",
            "comgra": "Quiscalus quiscula_Common Grackle",
            "comrav": "Corvus corax_Common Raven",
            "comyel": "Geothlypis trichas_Common Yellowthroat",
            "coohaw": "Accipiter cooperii_Cooper's Hawk",
            "daejun": "Junco hyemalis_Dark-eyed Junco",
            "dowwoo": "Dryobates pubescens_Downy Woodpecker",
            "easblu": "Sialia sialis_Eastern Bluebird",
            "easkin": "Tyrannus tyrannus_Eastern Kingbird",
            "easpho": "Sayornis phoebe_Eastern Phoebe",
            "eastow": "Pipilo erythrophthalmus_Eastern Towhee",
            "eawpew": "Contopus virens_Eastern Wood-Pewee",
            "eursta": "Sturnus vulgaris_European Starling",
            "gockin": "Regulus satrapa_Golden-crowned Kinglet",
            "grbher3": "Ardea herodias_Great Blue Heron",
            "grcfly": "Myiarchus crinitus_Great Crested Flycatcher",
            "grycat": "Dumetella carolinensis_Gray Catbird",
            "haiwoo": "Dryobates villosus_Hairy Woodpecker",
            "herthr": "Catharus guttatus_Hermit Thrush",
            "hoowar": "Setophaga citrina_Hooded Warbler",
            "houfin": "Haemorhous mexicanus_House Finch",
            "houwre": "Troglodytes aedon_House Wren",
            "killde": "Charadrius vociferus_Killdeer",
            "mallar3": "Anas platyrhynchos_Mallard",
            "moudov": "Zenaida macroura_Mourning Dove",
            "naswar": "Leiothlypis ruficapilla_Nashville Warbler",
            "norcar": "Cardinalis cardinalis_Northern Cardinal",
            "norfli": "Colaptes auratus_Northern Flicker",
            "norwat": "Parkesia noveboracensis_Northern Waterthrush",
            "ovenbi1": "Seiurus aurocapilla_Ovenbird",
            "pilwoo": "Dryocopus pileatus_Pileated Woodpecker",
            "pinsis": "Spinus pinus_Pine Siskin",
            "purfin": "Haemorhous purpureus_Purple Finch",
            "rebnut": "Sitta canadensis_Red-breasted Nuthatch",
            "rebwoo": "Melanerpes carolinus_Red-bellied Woodpecker",
            "redcro": "Loxia curvirostra_Red Crossbill",
            "reevir1": "Vireo olivaceus_Red-eyed Vireo",
            "rewbla": "Agelaius phoeniceus_Red-winged Blackbird",
            "ribgul": "Larus delawarensis_Ring-billed Gull",
            "robgro": "Pheucticus ludovicianus_Rose-breasted Grosbeak",
            "ruckin": "Corthylio calendula_Ruby-crowned Kinglet",
            "rusbla": "Euphagus carolinus_Rusty Blackbird",
            "scatan": "Piranga olivacea_Scarlet Tanager",
            "snogoo": "Anser caerulescens_Snow Goose",
            "solsan": "Tringa solitaria_Solitary Sandpiper",
            "sonspa": "Melospiza melodia_Song Sparrow",
            "swaspa": "Melospiza georgiana_Swamp Sparrow",
            "treswa": "Tachycineta bicolor_Tree Swallow",
            "tuftit": "Baeolophus bicolor_Tufted Titmouse",
            "tunswa": "Cygnus columbianus_Tundra Swan",
            "veery": "Catharus fuscescens_Veery",
            "warvir": "Vireo gilvus_Warbling Vireo",
            "whbnut": "Sitta carolinensis_White-breasted Nuthatch",
            "whtspa": "Zonotrichia albicollis_White-throated Sparrow",
            "wooduc": "Aix sponsa_Wood Duck",
            "woothr": "Hylocichla mustelina_Wood Thrush",
            "yebsap": "Sphyrapicus varius_Yellow-bellied Sapsucker",
            "yelwar": "Setophaga petechia_Yellow Warbler",
            "yerwar": "Setophaga coronata_Yellow-rumped Warbler",
            "yetvir": "Vireo flavifrons_Yellow-throated Vireo",
        },
        'bird_colors': {
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
    },
    
    'Southern-Sierra-Nevada': {
        'abbreviation': 'HSN',
        'id_to_ebird': {
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
        },
        'ebird_to_name': {
            "????": "Unknown",
            "amepip": "Anthus rubescens_American Pipit",
            "amerob": "Turdus migratorius_American Robin",
            "brebla": "Euphagus cyanocephalus_Brewer's Blackbird",
            "casfin": "Haemorhous cassinii_Cassin's Finch",
            "clanut": "Nucifraga columbiana_Clark's Nutcracker",
            "daejun": "Junco hyemalis_Dark-eyed Junco",
            "dusfly": "Empidonax oberholseri_Dusky Flycatcher",
            "foxspa": "Passerella iliaca_Fox Sparrow",
            "gcrfin": "Leucosticte tephrocotis_Gray-crowned Rosy-Finch",
            "herthr": "Catharus guttatus_Hermit Thrush",
            "mallar3": "Anas platyrhynchos_Mallard",
            "moublu": "Sialia currucoides_Mountain Bluebird",
            "mouchi": "Poecile gambeli_Mountain Chickadee",
            "norfli": "Colaptes auratus_Northern Flicker",
            "orcwar": "Leiothlypis celata_Orange-crowned Warbler",
            "rocwre": "Salpinctes obsoletus_Rock Wren",
            "sposan": "Actitis macularius_Spotted Sandpiper",
            "warvir": "Vireo gilvus_Warbling Vireo",
            "whcspa": "Zonotrichia leucophrys_White-crowned Sparrow",
            "yelwar": "Setophaga petechia_Yellow Warbler",
            "yerwar": "Setophaga coronata_Yellow-rumped Warbler",
        },
        'bird_colors': {
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
    },
    
    'Western-US': {
        'abbreviation': 'SNE',
        'id_to_ebird': {
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
        },
        'ebird_to_name': {
            "acowoo": "Melanerpes formicivorus_Acorn Woodpecker",
            "amegfi": "Spinus tristis_American Goldfinch",
            "amerob": "Turdus migratorius_American Robin",
            "annhum": "Calypte anna_Anna's Hummingbird",
            "batpig1": "Patagioenas fasciata_Band-tailed Pigeon",
            "bewwre": "Thryomanes bewickii_Bewick's Wren",
            "bkhgro": "Pheucticus melanocephalus_Black-headed Grosbeak",
            "bnhcow": "Molothrus ater_Brown-headed Cowbird",
            "brncre": "Certhia americana_Brown Creeper",
            "btywar": "Setophaga nigrescens_Black-throated Gray Warbler",
            "cangoo": "Branta canadensis_Canada Goose",
            "casfin": "Haemorhous cassinii_Cassin's Finch",
            "casvir": "Vireo cassinii_Cassin's Vireo",
            "chbchi": "Poecile rufescens_Chestnut-backed Chickadee",
            "comrav": "Corvus corax_Common Raven",
            "daejun": "Junco hyemalis_Dark-eyed Junco",
            "dusfly": "Empidonax oberholseri_Dusky Flycatcher",
            "evegro": "Coccothraustes vespertinus_Evening Grosbeak",
            "foxspa": "Passerella iliaca_Fox Sparrow",
            "gnttow": "Pipilo chlorurus_Green-tailed Towhee",
            "gockin": "Regulus satrapa_Golden-crowned Kinglet",
            "hamfly": "Empidonax hammondii_Hammond's Flycatcher",
            "herthr": "Catharus guttatus_Hermit Thrush",
            "herwar": "Setophaga occidentalis_Hermit Warbler",
            "houwre": "Troglodytes aedon_House Wren",
            "hutvir": "Vireo huttoni_Hutton's Vireo",
            "lazbun": "Passerina amoena_Lazuli Bunting",
            "linspa": "Melospiza lincolnii_Lincoln's Sparrow",
            "macwar": "Geothlypis tolmiei_MacGillivray's Warbler",
            "mouchi": "Poecile gambeli_Mountain Chickadee",
            "moudov": "Zenaida macroura_Mourning Dove",
            "mouqua": "Oreortyx pictus_Mountain Quail",
            "naswar": "Leiothlypis ruficapilla_Nashville Warbler",
            "norfli": "Colaptes auratus_Northern Flicker",
            "olsfly": "Contopus cooperi_Olive-sided Flycatcher",
            "orcwar": "Leiothlypis celata_Orange-crowned Warbler",
            "pasfly": "Empidonax difficilis_Pacific-slope Flycatcher",
            "pinsis": "Spinus pinus_Pine Siskin",
            "purfin": "Haemorhous purpureus_Purple Finch",
            "rebnut": "Sitta canadensis_Red-breasted Nuthatch",
            "redcro": "Loxia curvirostra_Red Crossbill",
            "ruckin": "Corthylio calendula_Ruby-crowned Kinglet",
            "spotow": "Pipilo maculatus_Spotted Towhee",
            "stejay": "Cyanocitta stelleri_Steller's Jay",
            "swathr": "Catharus ustulatus_Swainson's Thrush",
            "towsol": "Myadestes townsendi_Townsend's Solitaire",
            "towwar": "Setophaga townsendi_Townsend's Warbler",
            "vesspa": "Pooecetes gramineus_Vesper Sparrow",
            "warvir": "Vireo gilvus_Warbling Vireo",
            "westan": "Piranga ludoviciana_Western Tanager",
            "wewpew": "Contopus sordidulus_Western Wood-Pewee",
            "whcspa": "Zonotrichia leucophrys_White-crowned Sparrow",
            "whhwoo": "Dryobates albolarvatus_White-headed Woodpecker",
            "wilsap": "Sphyrapicus thyroideus_Williamson's Sapsucker",
            "wlswar": "Cardellina pusilla_Wilson's Warbler",
            "yerwar": "Setophaga coronata_Yellow-rumped Warbler",
        },
        'bird_colors': {
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
    }
}


def get_species_mapping_for_model(model_path: str) -> str:
    """
    Map model filename to its corresponding species mapping name.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Species mapping name (e.g., 'Hawaii', 'Western-US')
        
    Raises:
        ValueError: If model name doesn't match any known species mapping
    """
    model_name = Path(model_path).stem.lower()
    
    # Map model names to species mappings
    if 'hawaii' in model_name:
        if 'subset' in model_name:
            return 'Hawaii_subset'
        return 'Hawaii'
    elif 'western-us' in model_name or 'western_us' in model_name or model_name == 'western-us':
        return 'Western-US'
    elif 'northeastern-us' in model_name or 'northeastern_us' in model_name:
        if 'subset' in model_name:
            return 'Northeastern-US_subset'
        return 'Northeastern-US'
    elif 'sierra' in model_name or 'southern-sierra' in model_name:
        return 'Southern-Sierra-Nevada'
    else:
        # If we can't determine, show available options
        raise ValueError(
            f"Cannot determine species mapping for model: {Path(model_path).name}\n"
            f"Model name should contain: 'Hawaii', 'Western-US', 'Northeastern-US', or 'Sierra'"
        )


def get_species_mapping(species_mapping_name: str) -> Dict:
    """
    Get configuration for a specific species mapping.
    
    This function provides a clean interface to access species mapping configurations
    without mutating global state. All species mappings are defined as static data in SPECIES_MAPPING.
    
    Args:
        species_mapping_name: Name of the species mapping (e.g., 'Hawaii', 'Western-US')
        
    Returns:
        Dictionary containing:
            - 'id_to_ebird': Mapping from class IDs to eBird species codes
            - 'ebird_to_name': Mapping from eBird codes to full bird names
            - 'bird_colors': Mapping from class IDs to RGB color tuples
            - 'abbreviation': Species mapping abbreviation
            - 'clip_length': CLIP_LENGTH constant
            - 'height_width': HEIGHT_AND_WIDTH_IN_PIXELS constant
            - 'pcen_segment_length': PCEN_SEGMENT_LENGTH constant
            
    Raises:
        ValueError: If species mapping name is invalid
    """
    if species_mapping_name not in SPECIES_MAPPING:
        valid_mappings = ', '.join(sorted(SPECIES_MAPPING.keys()))
        raise ValueError(
            f"Invalid species mapping name: {species_mapping_name}\n"
            f"Available species mappings: {valid_mappings}"
        )
    
    mapping_data = SPECIES_MAPPING[species_mapping_name]
    
    return {
        'id_to_ebird': mapping_data['id_to_ebird'].copy(),
        'ebird_to_name': mapping_data['ebird_to_name'].copy(),
        'bird_colors': mapping_data['bird_colors'].copy(),
        'abbreviation': mapping_data['abbreviation'],
        'clip_length': CLIP_LENGTH,
        'height_width': HEIGHT_AND_WIDTH_IN_PIXELS,
        'pcen_segment_length': PCEN_SEGMENT_LENGTH
    }
