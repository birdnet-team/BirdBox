# SPDX-FileCopyrightText: 2025-2026 BirdNET-Team
# SPDX-License-Identifier: MIT
#################################################################################################################
### This file contains the configuration for the inference as well as for the evaluation of the detections.   ###
### All species mappings are defined as static data structures - no runtime mutations.                        ###
#################################################################################################################

from typing import Dict
from pathlib import Path

# Global constants (same for all datasets)
HEIGHT_AND_WIDTH_IN_PIXELS: int = 256
CLIP_LENGTH: int = 3
PCEN_SEGMENT_LENGTH: int = 60

# Species mapping configurations - all defined as static dictionaries
SPECIES_MAPPING = {
    'All-In-One': {
        'abbreviation': 'ALL',
        'id_to_ebird': {
            0: "????",
            1: "amabaw1",
            2: "amapyo1",
            3: "astgna1",
            4: "baffal1",
            5: "barant1",
            6: "bartin2",
            7: "batman1",
            8: "blacar1",
            9: "blbthr1",
            10: "blcbec1",
            11: "blctro1",
            12: "blfant1",
            13: "blfcot1",
            14: "blfjac1",
            15: "blfnun1",
            16: "blgdov1",
            17: "blhpar1",
            18: "bltant2",
            19: "blttro1",
            20: "bobfly1",
            21: "brratt1",
            22: "bsbeye1",
            23: "btfgle1",
            24: "bubgro2",
            25: "bubwre1",
            26: "bucmot4",
            27: "buffal1",
            28: "butsal1",
            29: "butwoo1",
            30: "chwfog1",
            31: "cinmou1",
            32: "cintin1",
            33: "citwoo1",
            34: "coffal1",
            35: "coltro1",
            36: "compot1",
            37: "cowpar1",
            38: "crfgle1",
            39: "ducatt1",
            40: "ducfly",
            41: "ducgre1",
            42: "duhpar",
            43: "dutant2",
            44: "elewoo1",
            45: "eulfly1",
            46: "fasant1",
            47: "fepowl",
            48: "forela1",
            49: "garkin1",
            50: "gilbar1",
            51: "gnbtro1",
            52: "gocspa1",
            53: "goeant1",
            54: "gogwoo1",
            55: "gramou1",
            56: "grasal3",
            57: "grcfly1",
            58: "greant1",
            59: "greibi1",
            60: "gretin1",
            61: "grfdov1",
            62: "gryant1",
            63: "gryant2",
            64: "gycfly1",
            65: "gycwor1",
            66: "hauthr1",
            67: "horscr1",
            68: "letbar1",
            69: "littin1",
            70: "litwoo2",
            71: "lobwoo1",
            72: "lowant1",
            73: "meapar",
            74: "muswre2",
            75: "olioro1",
            76: "oliwoo1",
            77: "partan1",
            78: "pavpig2",
            79: "pirfly1",
            80: "plbwoo1",
            81: "pltant1",
            82: "pluant1",
            83: "plupig2",
            84: "plwant1",
            85: "puteup1",
            86: "putfru1",
            87: "pygant1",
            88: "rcatan1",
            89: "rebmac2",
            90: "renwoo1",
            91: "rinant2",
            92: "rinkin1",
            93: "rinwoo1",
            94: "royfly1",
            95: "ruboro1",
            96: "rucant2",
            97: "rudpig",
            98: "rufant3",
            99: "ruftof1",
            100: "ruqdov",
            101: "scapig2",
            102: "scbwoo5",
            103: "scrpih1",
            104: "sobcac1",
            105: "specha3",
            106: "spigua1",
            107: "spwant2",
            108: "squcuc1",
            109: "stbwoo2",
            110: "strcuc1",
            111: "strwoo2",
            112: "strxen1",
            113: "stwqua1",
            114: "tabsco1",
            115: "thlwre1",
            116: "undtin1",
            117: "viotro3",
            118: "wespuf1",
            119: "whbtot1",
            120: "whcspa1",
            121: "whfant2",
            122: "whltyr1",
            123: "whnrob1",
            124: "whrsir1",
            125: "whttou1",
            126: "whtwoo2",
            127: "whwbec1",
            128: "wibpip1",
            129: "yectyr1",
            130: "yemfly1",
            131: "yercac1",
            132: "yetwoo2",
            133: "hawama",
            134: "ercfra",
            135: "jabwar",
            136: "apapan",
            137: "skylar",
            138: "houfin",
            139: "blkfra",
            140: "yefcan",
            141: "reblei",
            142: "warwhe1",
            143: "kalphe",
            144: "elepai",
            145: "hawhaw",
            146: "melthr",
            147: "iiwi",
            148: "calqua",
            149: "norcar",
            150: "wiltur",
            151: "chukar",
            152: "palila",
            153: "hawcre",
            154: "comwax",
            155: "hawpet1",
            156: "barpet",
            157: "akepa1",
            158: "hawgoo",
            159: "omao",
            160: "aldfly",
            161: "amecro",
            162: "amegfi",
            163: "amered",
            164: "amerob",
            165: "amewoo",
            166: "balori",
            167: "bcnher",
            168: "belkin1",
            169: "bkbwar",
            170: "bkcchi",
            171: "blujay",
            172: "bnhcow",
            173: "boboli",
            174: "brdowl",
            175: "brncre",
            176: "btnwar",
            177: "buhvir",
            178: "buwwar",
            179: "cangoo",
            180: "cedwax",
            181: "chswar",
            182: "comgra",
            183: "comrav",
            184: "comyel",
            185: "coohaw",
            186: "daejun",
            187: "dowwoo",
            188: "easblu",
            189: "easkin",
            190: "easpho",
            191: "eastow",
            192: "eawpew",
            193: "eursta",
            194: "gockin",
            195: "grbher3",
            196: "grcfly",
            197: "grycat",
            198: "haiwoo",
            199: "herthr",
            200: "hoowar",
            201: "houwre",
            202: "killde",
            203: "mallar3",
            204: "moudov",
            205: "naswar",
            206: "norfli",
            207: "norwat",
            208: "ovenbi1",
            209: "pilwoo",
            210: "pinsis",
            211: "purfin",
            212: "rebnut",
            213: "rebwoo",
            214: "redcro",
            215: "reevir1",
            216: "rewbla",
            217: "ribgul",
            218: "robgro",
            219: "ruckin",
            220: "rusbla",
            221: "scatan",
            222: "snogoo",
            223: "solsan",
            224: "sonspa",
            225: "swaspa",
            226: "treswa",
            227: "tuftit",
            228: "tunswa",
            229: "veery",
            230: "warvir",
            231: "whbnut",
            232: "whtspa",
            233: "wooduc",
            234: "woothr",
            235: "yebsap",
            236: "yelwar",
            237: "yerwar",
            238: "yetvir",
            239: "amepip",
            240: "brebla",
            241: "casfin",
            242: "clanut",
            243: "dusfly",
            244: "foxspa",
            245: "gcrfin",
            246: "moublu",
            247: "mouchi",
            248: "orcwar",
            249: "rocwre",
            250: "sposan",
            251: "whcspa",
            252: "acowoo",
            253: "annhum",
            254: "batpig1",
            255: "bewwre",
            256: "bkhgro",
            257: "btywar",
            258: "casvir",
            259: "chbchi",
            260: "evegro",
            261: "gnttow",
            262: "hamfly",
            263: "herwar",
            264: "hutvir",
            265: "lazbun",
            266: "linspa",
            267: "macwar",
            268: "mouqua",
            269: "olsfly",
            270: "pasfly",
            271: "spotow",
            272: "stejay",
            273: "swathr",
            274: "towsol",
            275: "towwar",
            276: "vesspa",
            277: "westan",
            278: "wewpew",
            279: "whhwoo",
            280: "wilsap",
            281: "wlswar",
        },
        'ebird_to_name': {
            "????": "Unknown",
            "acowoo": "Melanerpes formicivorus_Acorn Woodpecker",
            "akepa1": "Loxops coccineus_Hawaii Akepa",
            "aldfly": "Empidonax alnorum_Alder Flycatcher",
            "amabaw1": "Dendrocolaptes certhia_Amazonian Barred-Woodcreeper",
            "amapyo1": "Glaucidium hardyi_Amazonian Pygmy-Owl",
            "amecro": "Corvus brachyrhynchos_American Crow",
            "amegfi": "Spinus tristis_American Goldfinch",
            "amepip": "Anthus rubescens_American Pipit",
            "amered": "Setophaga ruticilla_American Redstart",
            "amerob": "Turdus migratorius_American Robin",
            "amewoo": "Scolopax minor_American Woodcock",
            "annhum": "Calypte anna_Anna's Hummingbird",
            "apapan": "Himatione sanguinea_Apapane",
            "astgna1": "Conopophaga peruviana_Ash-throated Gnateater",
            "baffal1": "Micrastur ruficollis_Barred Forest-Falcon",
            "balori": "Icterus galbula_Baltimore Oriole",
            "barant1": "Thamnophilus doliatus_Barred Antshrike",
            "barpet": "Hydrobates castro_Band-rumped Storm-Petrel",
            "bartin2": "Crypturellus bartletti_Bartlett's Tinamou",
            "batman1": "Pipra fasciicauda_Band-tailed Manakin",
            "batpig1": "Patagioenas fasciata_Band-tailed Pigeon",
            "bcnher": "Nycticorax nycticorax_Black-crowned Night-Heron",
            "belkin1": "Megaceryle alcyon_Belted Kingfisher",
            "bewwre": "Thryomanes bewickii_Bewick's Wren",
            "bkbwar": "Setophaga fusca_Blackburnian Warbler",
            "bkcchi": "Poecile atricapillus_Black-capped Chickadee",
            "bkhgro": "Pheucticus melanocephalus_Black-headed Grosbeak",
            "blacar1": "Daptrius ater_Black Caracara",
            "blbthr1": "Turdus ignobilis_Black-billed Thrush",
            "blcbec1": "Pachyramphus marginatus_Black-capped Becard",
            "blctro1": "Trogon curucui_Blue-crowned Trogon",
            "blfant1": "Formicarius analis_Black-faced Antthrush",
            "blfcot1": "Conioptilon mcilhennyi_Black-faced Cotinga",
            "blfjac1": "Galbula cyanescens_Bluish-fronted Jacamar",
            "blfnun1": "Monasa nigrifrons_Black-fronted Nunbird",
            "blgdov1": "Claravis pretiosa_Blue Ground Dove",
            "blhpar1": "Pionus menstruus_Blue-headed Parrot",
            "blkfra": "Francolinus francolinus_Black Francolin",
            "bltant2": "Myrmophylax atrothorax_Black-throated Antbird",
            "blttro1": "Trogon melanurus_Black-tailed Trogon",
            "blujay": "Cyanocitta cristata_Blue Jay",
            "bnhcow": "Molothrus ater_Brown-headed Cowbird",
            "bobfly1": "Megarynchus pitangua_Boat-billed Flycatcher",
            "boboli": "Dolichonyx oryzivorus_Bobolink",
            "brdowl": "Strix varia_Barred Owl",
            "brebla": "Euphagus cyanocephalus_Brewer's Blackbird",
            "brncre": "Certhia americana_Brown Creeper",
            "brratt1": "Attila spadiceus_Bright-rumped Attila",
            "bsbeye1": "Phlegopsis nigromaculata_Black-spotted Bare-eye",
            "btfgle1": "Automolus ochrolaemus_Buff-throated Foliage-gleaner",
            "btnwar": "Setophaga virens_Black-throated Green Warbler",
            "btywar": "Setophaga nigrescens_Black-throated Gray Warbler",
            "bubgro2": "Cyanoloxia rothschildii_Amazonian Grosbeak",
            "bubwre1": "Cantorchilus leucotis_Buff-breasted Wren",
            "bucmot4": "Momotus momota_Amazonian Motmot",
            "buffal1": "Micrastur buckleyi_Buckley's Forest-Falcon",
            "buhvir": "Vireo solitarius_Blue-headed Vireo",
            "butsal1": "Saltator maximus_Buff-throated Saltator",
            "butwoo1": "Xiphorhynchus guttatus_Buff-throated Woodcreeper",
            "buwwar": "Vermivora cyanoptera_Blue-winged Warbler",
            "calqua": "Callipepla californica_California Quail",
            "cangoo": "Branta canadensis_Canada Goose",
            "casfin": "Haemorhous cassinii_Cassin's Finch",
            "casvir": "Vireo cassinii_Cassin's Vireo",
            "cedwax": "Bombycilla cedrorum_Cedar Waxwing",
            "chbchi": "Poecile rufescens_Chestnut-backed Chickadee",
            "chswar": "Setophaga pensylvanica_Chestnut-sided Warbler",
            "chukar": "Alectoris chukar_Chukar",
            "chwfog1": "Dendroma erythroptera_Chestnut-winged Foliage-gleaner",
            "cinmou1": "Laniocera hypopyrra_Cinereous Mourner",
            "cintin1": "Crypturellus cinereus_Cinereous Tinamou",
            "citwoo1": "Dendrexetastes rufigula_Cinnamon-throated Woodcreeper",
            "clanut": "Nucifraga columbiana_Clark's Nutcracker",
            "coffal1": "Micrastur semitorquatus_Collared Forest-Falcon",
            "coltro1": "Trogon collaris_Collared Trogon",
            "comgra": "Quiscalus quiscula_Common Grackle",
            "compot1": "Nyctibius griseus_Common Potoo",
            "comrav": "Corvus corax_Common Raven",
            "comwax": "Estrilda astrild_Common Waxbill",
            "comyel": "Geothlypis trichas_Common Yellowthroat",
            "coohaw": "Accipiter cooperii_Cooper's Hawk",
            "cowpar1": "Brotogeris cyanoptera_Cobalt-winged Parakeet",
            "crfgle1": "Philydor pyrrhodes_Cinnamon-rumped Foliage-gleaner",
            "daejun": "Junco hyemalis_Dark-eyed Junco",
            "dowwoo": "Dryobates pubescens_Downy Woodpecker",
            "ducatt1": "Attila bolivianus_Dull-capped Attila",
            "ducfly": "Myiarchus tuberculifer_Dusky-capped Flycatcher",
            "ducgre1": "Pachysylvia hypoxantha_Dusky-capped Greenlet",
            "duhpar": "Aratinga weddellii_Dusky-headed Parakeet",
            "dusfly": "Empidonax oberholseri_Dusky Flycatcher",
            "dutant2": "Thamnomanes ardesiacus_Dusky-throated Antshrike",
            "easblu": "Sialia sialis_Eastern Bluebird",
            "easkin": "Tyrannus tyrannus_Eastern Kingbird",
            "easpho": "Sayornis phoebe_Eastern Phoebe",
            "eastow": "Pipilo erythrophthalmus_Eastern Towhee",
            "eawpew": "Contopus virens_Eastern Wood-Pewee",
            "elepai": "Chasiempis sandwichensis_Hawaii Elepaio",
            "elewoo1": "Xiphorhynchus elegans_Elegant Woodcreeper",
            "ercfra": "Pternistis erckelii_Erckel's Francolin",
            "eulfly1": "Lathrotriccus euleri_Euler's Flycatcher",
            "eursta": "Sturnus vulgaris_European Starling",
            "evegro": "Coccothraustes vespertinus_Evening Grosbeak",
            "fasant1": "Cymbilaimus lineatus_Fasciated Antshrike",
            "fepowl": "Glaucidium brasilianum_Ferruginous Pygmy-Owl",
            "forela1": "Myiopagis gaimardii_Forest Elaenia",
            "foxspa": "Passerella iliaca_Fox Sparrow",
            "garkin1": "Chloroceryle inda_Green-and-rufous Kingfisher",
            "gcrfin": "Leucosticte tephrocotis_Gray-crowned Rosy-Finch",
            "gilbar1": "Capito auratus_Gilded Barbet",
            "gnbtro1": "Trogon viridis_Green-backed Trogon",
            "gnttow": "Pipilo chlorurus_Green-tailed Towhee",
            "gockin": "Regulus satrapa_Golden-crowned Kinglet",
            "gocspa1": "Platyrinchus coronatus_Golden-crowned Spadebill",
            "goeant1": "Akletos goeldii_Goeldi's Antbird",
            "gogwoo1": "Piculus chrysochloros_Golden-green Woodpecker",
            "gramou1": "Rhytipterna simplex_Grayish Mourner",
            "grasal3": "Saltator coerulescens_Blue-gray Saltator",
            "grbher3": "Ardea herodias_Great Blue Heron",
            "grcfly": "Myiarchus crinitus_Great Crested Flycatcher",
            "grcfly1": "Myiozetetes granadensis_Gray-capped Flycatcher",
            "greant1": "Taraba major_Great Antshrike",
            "greibi1": "Mesembrinibis cayennensis_Green Ibis",
            "gretin1": "Tinamus major_Great Tinamou",
            "grfdov1": "Leptotila rufaxilla_Gray-fronted Dove",
            "gryant1": "Myrmotherula menetriesii_Gray Antwren",
            "gryant2": "Cercomacra cinerascens_Gray Antbird",
            "grycat": "Dumetella carolinensis_Gray Catbird",
            "gycfly1": "Tolmomyias poliocephalus_Gray-crowned Flycatcher",
            "gycwor1": "Aramides cajaneus_Gray-cowled Wood-Rail",
            "haiwoo": "Dryobates villosus_Hairy Woodpecker",
            "hamfly": "Empidonax hammondii_Hammond's Flycatcher",
            "hauthr1": "Turdus hauxwelli_Hauxwell's Thrush",
            "hawama": "Chlorodrepanis virens_Hawaii Amakihi",
            "hawcre": "Loxops mana_Hawaii Creeper",
            "hawgoo": "Branta sandvicensis_Hawaiian Goose",
            "hawhaw": "Buteo solitarius_Hawaiian Hawk",
            "hawpet1": "Pterodroma sandwichensis_Hawaiian Petrel",
            "herthr": "Catharus guttatus_Hermit Thrush",
            "herwar": "Setophaga occidentalis_Hermit Warbler",
            "hoowar": "Setophaga citrina_Hooded Warbler",
            "horscr1": "Anhima cornuta_Horned Screamer",
            "houfin": "Haemorhous mexicanus_House Finch",
            "houwre": "Troglodytes aedon_House Wren",
            "hutvir": "Vireo huttoni_Hutton's Vireo",
            "iiwi": "Drepanis coccinea_Iiwi",
            "jabwar": "Horornis diphone_Japanese Bush Warbler",
            "kalphe": "Lophura leucomelanos_Kalij Pheasant",
            "killde": "Charadrius vociferus_Killdeer",
            "lazbun": "Passerina amoena_Lazuli Bunting",
            "letbar1": "Eubucco richardsoni_Lemon-throated Barbet",
            "linspa": "Melospiza lincolnii_Lincoln's Sparrow",
            "littin1": "Crypturellus soui_Little Tinamou",
            "litwoo2": "Dryobates passerinus_Little Woodpecker",
            "lobwoo1": "Nasica longirostris_Long-billed Woodcreeper",
            "lowant1": "Myrmotherula longipennis_Long-winged Antwren",
            "macwar": "Geothlypis tolmiei_MacGillivray's Warbler",
            "mallar3": "Anas platyrhynchos_Mallard",
            "meapar": "Amazona farinosa_Mealy Parrot",
            "melthr": "Garrulax canorus_Chinese Hwamei",
            "moublu": "Sialia currucoides_Mountain Bluebird",
            "mouchi": "Poecile gambeli_Mountain Chickadee",
            "moudov": "Zenaida macroura_Mourning Dove",
            "mouqua": "Oreortyx pictus_Mountain Quail",
            "muswre2": "Cyphorhinus arada_Musician Wren",
            "naswar": "Leiothlypis ruficapilla_Nashville Warbler",
            "norcar": "Cardinalis cardinalis_Northern Cardinal",
            "norfli": "Colaptes auratus_Northern Flicker",
            "norwat": "Parkesia noveboracensis_Northern Waterthrush",
            "olioro1": "Psarocolius bifasciatus_Olive Oropendola",
            "oliwoo1": "Sittasomus griseicapillus_Olivaceous Woodcreeper",
            "olsfly": "Contopus cooperi_Olive-sided Flycatcher",
            "omao": "Myadestes obscurus_Omao",
            "orcwar": "Leiothlypis celata_Orange-crowned Warbler",
            "ovenbi1": "Seiurus aurocapilla_Ovenbird",
            "palila": "Loxioides bailleui_Palila",
            "partan1": "Tangara chilensis_Paradise Tanager",
            "pasfly": "Empidonax difficilis_Pacific-slope Flycatcher",
            "pavpig2": "Patagioenas cayennensis_Pale-vented Pigeon",
            "pilwoo": "Dryocopus pileatus_Pileated Woodpecker",
            "pinsis": "Spinus pinus_Pine Siskin",
            "pirfly1": "Legatus leucophaius_Piratic Flycatcher",
            "plbwoo1": "Dendrocincla fuliginosa_Plain-brown Woodcreeper",
            "pltant1": "Isleria hauxwelli_Plain-throated Antwren",
            "pluant1": "Myrmelastes hyperythrus_Plumbeous Antbird",
            "plupig2": "Patagioenas plumbea_Plumbeous Pigeon",
            "plwant1": "Thamnophilus schistaceus_Plain-winged Antshrike",
            "purfin": "Haemorhous purpureus_Purple Finch",
            "puteup1": "Euphonia chlorotica_Purple-throated Euphonia",
            "putfru1": "Querula purpurata_Purple-throated Fruitcrow",
            "pygant1": "Myrmotherula brachyura_Pygmy Antwren",
            "rcatan1": "Habia rubica_Red-crowned Ant-Tanager",
            "reblei": "Leiothrix lutea_Red-billed Leiothrix",
            "rebmac2": "Orthopsittaca manilatus_Red-bellied Macaw",
            "rebnut": "Sitta canadensis_Red-breasted Nuthatch",
            "rebwoo": "Melanerpes carolinus_Red-bellied Woodpecker",
            "redcro": "Loxia curvirostra_Red Crossbill",
            "reevir1": "Vireo olivaceus_Red-eyed Vireo",
            "renwoo1": "Campephilus rubricollis_Red-necked Woodpecker",
            "rewbla": "Agelaius phoeniceus_Red-winged Blackbird",
            "ribgul": "Larus delawarensis_Ring-billed Gull",
            "rinant2": "Corythopis torquatus_Ringed Antpipit",
            "rinkin1": "Megaceryle torquata_Ringed Kingfisher",
            "rinwoo1": "Celeus torquatus_Ringed Woodpecker",
            "robgro": "Pheucticus ludovicianus_Rose-breasted Grosbeak",
            "rocwre": "Salpinctes obsoletus_Rock Wren",
            "royfly1": "Onychorhynchus coronatus_Royal Flycatcher",
            "ruboro1": "Psarocolius angustifrons_Russet-backed Oropendola",
            "rucant2": "Formicarius colma_Rufous-capped Antthrush",
            "ruckin": "Corthylio calendula_Ruby-crowned Kinglet",
            "rudpig": "Patagioenas subvinacea_Ruddy Pigeon",
            "rufant3": "Formicarius rufifrons_Rufous-fronted Antthrush",
            "ruftof1": "Poecilotriccus latirostris_Rusty-fronted Tody-Flycatcher",
            "ruqdov": "Geotrygon montana_Ruddy Quail-Dove",
            "rusbla": "Euphagus carolinus_Rusty Blackbird",
            "scapig2": "Patagioenas speciosa_Scaled Pigeon",
            "scatan": "Piranga olivacea_Scarlet Tanager",
            "scbwoo5": "Celeus grammicus_Scale-breasted Woodpecker",
            "scrpih1": "Lipaugus vociferans_Screaming Piha",
            "skylar": "Alauda arvensis_Eurasian Skylark",
            "snogoo": "Anser caerulescens_Snow Goose",
            "sobcac1": "Cacicus solitarius_Solitary Black Cacique",
            "solsan": "Tringa solitaria_Solitary Sandpiper",
            "sonspa": "Melospiza melodia_Song Sparrow",
            "specha3": "Ortalis guttata_Speckled Chachalaca",
            "spigua1": "Penelope jacquacu_Spix's Guan",
            "sposan": "Actitis macularius_Spotted Sandpiper",
            "spotow": "Pipilo maculatus_Spotted Towhee",
            "spwant2": "Pygiptila stellaris_Spot-winged Antshrike",
            "squcuc1": "Piaya cayana_Squirrel Cuckoo",
            "stbwoo2": "Dendroplex picus_Straight-billed Woodcreeper",
            "stejay": "Cyanocitta stelleri_Steller's Jay",
            "strcuc1": "Tapera naevia_Striped Cuckoo",
            "strwoo2": "Xiphorhynchus obsoletus_Striped Woodcreeper",
            "strxen1": "Xenops rutilans_Streaked Xenops",
            "stwqua1": "Odontophorus stellatus_Starred Wood-Quail",
            "swaspa": "Melospiza georgiana_Swamp Sparrow",
            "swathr": "Catharus ustulatus_Swainson's Thrush",
            "tabsco1": "Megascops watsonii_Tawny-bellied Screech-Owl",
            "thlwre1": "Campylorhynchus turdinus_Thrush-like Wren",
            "towsol": "Myadestes townsendi_Townsend's Solitaire",
            "towwar": "Setophaga townsendi_Townsend's Warbler",
            "treswa": "Tachycineta bicolor_Tree Swallow",
            "tuftit": "Baeolophus bicolor_Tufted Titmouse",
            "tunswa": "Cygnus columbianus_Tundra Swan",
            "undtin1": "Crypturellus undulatus_Undulated Tinamou",
            "veery": "Catharus fuscescens_Veery",
            "vesspa": "Pooecetes gramineus_Vesper Sparrow",
            "viotro3": "Trogon ramonianus_Amazonian Trogon",
            "warvir": "Vireo gilvus_Warbling Vireo",
            "warwhe1": "Zosterops japonicus_Warbling White-eye",
            "wespuf1": "Nystalus obamai_Western Striolated-Puffbird",
            "westan": "Piranga ludoviciana_Western Tanager",
            "wewpew": "Contopus sordidulus_Western Wood-Pewee",
            "whbnut": "Sitta carolinensis_White-breasted Nuthatch",
            "whbtot1": "Hemitriccus griseipectus_White-bellied Tody-Tyrant",
            "whcspa": "Zonotrichia leucophrys_White-crowned Sparrow",
            "whcspa1": "Platyrinchus platyrhynchos_White-crested Spadebill",
            "whfant2": "Myrmotherula axillaris_White-flanked Antwren",
            "whhwoo": "Dryobates albolarvatus_White-headed Woodpecker",
            "whltyr1": "Ornithion inerme_White-lored Tyrannulet",
            "whnrob1": "Turdus albicollis_White-necked Thrush",
            "whrsir1": "Sirystes albocinereus_White-rumped Sirystes",
            "whtspa": "Zonotrichia albicollis_White-throated Sparrow",
            "whttou1": "Ramphastos tucanus_White-throated Toucan",
            "whtwoo2": "Piculus leucolaemus_White-throated Woodpecker",
            "whwbec1": "Pachyramphus polychopterus_White-winged Becard",
            "wibpip1": "Piprites chloris_Wing-barred Piprites",
            "wilsap": "Sphyrapicus thyroideus_Williamson's Sapsucker",
            "wiltur": "Meleagris gallopavo_Wild Turkey",
            "wlswar": "Cardellina pusilla_Wilson's Warbler",
            "wooduc": "Aix sponsa_Wood Duck",
            "woothr": "Hylocichla mustelina_Wood Thrush",
            "yebsap": "Sphyrapicus varius_Yellow-bellied Sapsucker",
            "yectyr1": "Tyrannulus elatus_Yellow-crowned Tyrannulet",
            "yefcan": "Crithagra mozambica_Yellow-fronted Canary",
            "yelwar": "Setophaga petechia_Yellow Warbler",
            "yemfly1": "Tolmomyias assimilis_Yellow-margined Flycatcher",
            "yercac1": "Cacicus cela_Yellow-rumped Cacique",
            "yerwar": "Setophaga coronata_Yellow-rumped Warbler",
            "yetvir": "Vireo flavifrons_Yellow-throated Vireo",
            "yetwoo2": "Melanerpes cruentatus_Yellow-tufted Woodpecker",
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
            56:  (255, 165, 0),     # Orange
            57:  (138, 221, 45),    # Chartreuse
            58:  (70, 240, 240),    # Turquoise
            59:  (0, 128, 255),     # Azure
            60:  (204, 0, 204),     # Purple
            61:  (255, 204, 153),   # Peach
            62:  (255, 51, 153),    # Pink
            63:  (153, 204, 50),    # YellowGreen2
            64:  (0, 255, 200),     # Aqua
            65:  (255, 102, 0),     # BrightOrange
            66:  (255, 0, 127),     # Rose
            67:  (64, 224, 208),    # Turquoise2
            68:  (210, 105, 30),    # Chocolate
            69:  (176, 224, 230),   # PowderBlue
            70:  (255, 228, 181),   # Moccasin
            71:  (240, 128, 128),   # LightCoral
            72:  (255, 99, 255),    # VioletPink
            73:  (124, 252, 0),     # LawnGreen
            74:  (255, 239, 0),     # BrightYellow
            75:  (139, 0, 139),     # DarkMagenta
            76:  (0, 139, 139),     # DarkCyan
            77:  (65, 105, 225),    # RoyalBlue
            78:  (205, 92, 92),     # IndianRed
            79:  (255, 165, 0),     # Orange
            80:  (138, 221, 45),    # Chartreuse
            81:  (70, 240, 240),    # Turquoise
            82:  (0, 128, 255),     # Azure
            83:  (204, 0, 204),     # Purple
            84:  (255, 204, 153),   # Peach
            85:  (255, 51, 153),    # Pink
            86:  (153, 204, 50),    # YellowGreen2
            87:  (0, 255, 200),     # Aqua
            88:  (255, 102, 0),     # BrightOrange
            89:  (255, 0, 127),     # Rose
            90:  (64, 224, 208),    # Turquoise2
            91:  (210, 105, 30),    # Chocolate
            92:  (176, 224, 230),   # PowderBlue
            93:  (255, 228, 181),   # Moccasin
            94:  (240, 128, 128),   # LightCoral
            95:  (255, 99, 255),    # VioletPink
            96:  (124, 252, 0),     # LawnGreen
            97:  (255, 239, 0),     # BrightYellow
            98:  (139, 0, 139),     # DarkMagenta
            99:  (0, 139, 139),     # DarkCyan
            100:  (65, 105, 225),    # RoyalBlue
            101:  (205, 92, 92),     # IndianRed
            102:  (255, 165, 0),     # Orange
            103:  (138, 221, 45),    # Chartreuse
            104:  (70, 240, 240),    # Turquoise
            105:  (0, 128, 255),     # Azure
            106:  (204, 0, 204),     # Purple
            107:  (255, 204, 153),   # Peach
            108:  (255, 51, 153),    # Pink
            109:  (153, 204, 50),    # YellowGreen2
            110:  (0, 255, 200),     # Aqua
            111:  (255, 102, 0),     # BrightOrange
            112:  (255, 0, 127),     # Rose
            113:  (64, 224, 208),    # Turquoise2
            114:  (210, 105, 30),    # Chocolate
            115:  (176, 224, 230),   # PowderBlue
            116:  (255, 228, 181),   # Moccasin
            117:  (240, 128, 128),   # LightCoral
            118:  (255, 99, 255),    # VioletPink
            119:  (124, 252, 0),     # LawnGreen
            120:  (255, 239, 0),     # BrightYellow
            121:  (139, 0, 139),     # DarkMagenta
            122:  (0, 139, 139),     # DarkCyan
            123:  (65, 105, 225),    # RoyalBlue
            124:  (205, 92, 92),     # IndianRed
            125:  (255, 165, 0),     # Orange
            126:  (138, 221, 45),    # Chartreuse
            127:  (70, 240, 240),    # Turquoise
            128:  (0, 128, 255),     # Azure
            129:  (204, 0, 204),     # Purple
            130:  (255, 204, 153),   # Peach
            131:  (255, 51, 153),    # Pink
            132:  (153, 204, 50),    # YellowGreen2
            133:  (0, 255, 200),     # Aqua
            134:  (255, 102, 0),     # BrightOrange
            135:  (255, 0, 127),     # Rose
            136:  (64, 224, 208),    # Turquoise2
            137:  (210, 105, 30),    # Chocolate
            138:  (176, 224, 230),   # PowderBlue
            139:  (255, 228, 181),   # Moccasin
            140:  (240, 128, 128),   # LightCoral
            141:  (255, 99, 255),    # VioletPink
            142:  (124, 252, 0),     # LawnGreen
            143:  (255, 239, 0),     # BrightYellow
            144:  (139, 0, 139),     # DarkMagenta
            145:  (0, 139, 139),     # DarkCyan
            146:  (65, 105, 225),    # RoyalBlue
            147:  (205, 92, 92),     # IndianRed
            148:  (255, 165, 0),     # Orange
            149:  (138, 221, 45),    # Chartreuse
            150:  (70, 240, 240),    # Turquoise
            151:  (0, 128, 255),     # Azure
            152:  (204, 0, 204),     # Purple
            153:  (255, 204, 153),   # Peach
            154:  (255, 51, 153),    # Pink
            155:  (153, 204, 50),    # YellowGreen2
            156:  (0, 255, 200),     # Aqua
            157:  (255, 102, 0),     # BrightOrange
            158:  (255, 0, 127),     # Rose
            159:  (64, 224, 208),    # Turquoise2
            160:  (210, 105, 30),    # Chocolate
            161:  (176, 224, 230),   # PowderBlue
            162:  (255, 228, 181),   # Moccasin
            163:  (240, 128, 128),   # LightCoral
            164:  (255, 99, 255),    # VioletPink
            165:  (124, 252, 0),     # LawnGreen
            166:  (255, 239, 0),     # BrightYellow
            167:  (139, 0, 139),     # DarkMagenta
            168:  (0, 139, 139),     # DarkCyan
            169:  (65, 105, 225),    # RoyalBlue
            170:  (205, 92, 92),     # IndianRed
            171:  (255, 165, 0),     # Orange
            172:  (138, 221, 45),    # Chartreuse
            173:  (70, 240, 240),    # Turquoise
            174:  (0, 128, 255),     # Azure
            175:  (204, 0, 204),     # Purple
            176:  (255, 204, 153),   # Peach
            177:  (255, 51, 153),    # Pink
            178:  (153, 204, 50),    # YellowGreen2
            179:  (0, 255, 200),     # Aqua
            180:  (255, 102, 0),     # BrightOrange
            181:  (255, 0, 127),     # Rose
            182:  (64, 224, 208),    # Turquoise2
            183:  (210, 105, 30),    # Chocolate
            184:  (176, 224, 230),   # PowderBlue
            185:  (255, 228, 181),   # Moccasin
            186:  (240, 128, 128),   # LightCoral
            187:  (255, 99, 255),    # VioletPink
            188:  (124, 252, 0),     # LawnGreen
            189:  (255, 239, 0),     # BrightYellow
            190:  (139, 0, 139),     # DarkMagenta
            191:  (0, 139, 139),     # DarkCyan
            192:  (65, 105, 225),    # RoyalBlue
            193:  (205, 92, 92),     # IndianRed
            194:  (255, 165, 0),     # Orange
            195:  (138, 221, 45),    # Chartreuse
            196:  (70, 240, 240),    # Turquoise
            197:  (0, 128, 255),     # Azure
            198:  (204, 0, 204),     # Purple
            199:  (255, 204, 153),   # Peach
            200:  (255, 51, 153),    # Pink
            201:  (153, 204, 50),    # YellowGreen2
            202:  (0, 255, 200),     # Aqua
            203:  (255, 102, 0),     # BrightOrange
            204:  (255, 0, 127),     # Rose
            205:  (64, 224, 208),    # Turquoise2
            206:  (210, 105, 30),    # Chocolate
            207:  (176, 224, 230),   # PowderBlue
            208:  (255, 228, 181),   # Moccasin
            209:  (240, 128, 128),   # LightCoral
            210:  (255, 99, 255),    # VioletPink
            211:  (124, 252, 0),     # LawnGreen
            212:  (255, 239, 0),     # BrightYellow
            213:  (139, 0, 139),     # DarkMagenta
            214:  (0, 139, 139),     # DarkCyan
            215:  (65, 105, 225),    # RoyalBlue
            216:  (205, 92, 92),     # IndianRed
            217:  (255, 165, 0),     # Orange
            218:  (138, 221, 45),    # Chartreuse
            219:  (70, 240, 240),    # Turquoise
            220:  (0, 128, 255),     # Azure
            221:  (204, 0, 204),     # Purple
            222:  (255, 204, 153),   # Peach
            223:  (255, 51, 153),    # Pink
            224:  (153, 204, 50),    # YellowGreen2
            225:  (0, 255, 200),     # Aqua
            226:  (255, 102, 0),     # BrightOrange
            227:  (255, 0, 127),     # Rose
            228:  (64, 224, 208),    # Turquoise2
            229:  (210, 105, 30),    # Chocolate
            230:  (176, 224, 230),   # PowderBlue
            231:  (255, 228, 181),   # Moccasin
            232:  (240, 128, 128),   # LightCoral
            233:  (255, 99, 255),    # VioletPink
            234:  (124, 252, 0),     # LawnGreen
            235:  (255, 239, 0),     # BrightYellow
            236:  (139, 0, 139),     # DarkMagenta
            237:  (0, 139, 139),     # DarkCyan
            238:  (65, 105, 225),    # RoyalBlue
            239:  (205, 92, 92),     # IndianRed
            240:  (255, 165, 0),     # Orange
            241:  (138, 221, 45),    # Chartreuse
            242:  (70, 240, 240),    # Turquoise
            243:  (0, 128, 255),     # Azure
            244:  (204, 0, 204),     # Purple
            245:  (255, 204, 153),   # Peach
            246:  (255, 51, 153),    # Pink
            247:  (153, 204, 50),    # YellowGreen2
            248:  (0, 255, 200),     # Aqua
            249:  (255, 102, 0),     # BrightOrange
            250:  (255, 0, 127),     # Rose
            251:  (64, 224, 208),    # Turquoise2
            252:  (210, 105, 30),    # Chocolate
            253:  (176, 224, 230),   # PowderBlue
            254:  (255, 228, 181),   # Moccasin
            255:  (240, 128, 128),   # LightCoral
            256:  (255, 99, 255),    # VioletPink
            257:  (124, 252, 0),     # LawnGreen
            258:  (255, 239, 0),     # BrightYellow
            259:  (139, 0, 139),     # DarkMagenta
            260:  (0, 139, 139),     # DarkCyan
            261:  (65, 105, 225),    # RoyalBlue
            262:  (205, 92, 92),     # IndianRed
            263:  (255, 165, 0),     # Orange
            264:  (138, 221, 45),    # Chartreuse
            265:  (70, 240, 240),    # Turquoise
            266:  (0, 128, 255),     # Azure
            267:  (204, 0, 204),     # Purple
            268:  (255, 204, 153),   # Peach
            269:  (255, 51, 153),    # Pink
            270:  (153, 204, 50),    # YellowGreen2
            271:  (0, 255, 200),     # Aqua
            272:  (255, 102, 0),     # BrightOrange
            273:  (255, 0, 127),     # Rose
            274:  (64, 224, 208),    # Turquoise2
            275:  (210, 105, 30),    # Chocolate
            276:  (176, 224, 230),   # PowderBlue
            277:  (255, 228, 181),   # Moccasin
            278:  (240, 128, 128),   # LightCoral
            279:  (255, 99, 255),    # VioletPink
            280:  (124, 252, 0),     # LawnGreen
            281:  (255, 239, 0),     # BrightYellow
        }
    },

    'Just-Bird': {
        'abbreviation': 'BIRD',
        'id_to_ebird': {
            0: "bird",
        },
        'ebird_to_name': {
            "bird": "Bird_Bird",
        },
        'bird_colors': {
            0: (255, 215, 0),  # Gold
        }
    }
}


def get_species_mapping_for_model(model_path: str) -> str:
    """
    Map model filename to its corresponding species mapping name.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Species mapping name ('All-In-One' or 'Just-Bird')
        
    Raises:
        ValueError: If model name doesn't match any known species mapping
    """
    model_name = Path(model_path).stem.lower().replace('_', '-')
    
    model_name_to_species_mapping = {
        'all-in-one': 'All-In-One',
        'just-bird': 'Just-Bird',
    }

    for key, display_name in model_name_to_species_mapping.items():
        if key in model_name:
            return display_name

    # If we can't determine, show available options
    raise ValueError(
        f"Cannot determine species mapping for model: {Path(model_path).name}\n"
        f"Model name should contain one of: {', '.join(sorted(model_name_to_species_mapping.keys()))}"
    )


def get_species_mapping(species_mapping_name: str) -> Dict:
    """
    Get configuration for a specific species mapping.
    
    This function provides a clean interface to access species mapping configurations
    without mutating global state. All species mappings are defined as static data in SPECIES_MAPPING.
    
    Args:
        species_mapping_name: Name of the species mapping ('All-In-One' or 'Just-Bird')
        
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
