"""
Resolve free text typed by a person into one of the 345 things G-Doodle can draw.

This is the layer that makes "wpisz co chcesz" honest. The model itself has no
language in it at all — it is conditioned on a single category token, and its
whole world is the 345 classes QuickDraw contains. So the text box cannot be a
prompt in the Stable Diffusion sense, and pretending otherwise would produce the
worst possible failure: you ask for a rocket, you get a confident helicopter, and
nothing tells you the model never knew what a rocket was.

Instead: Polish (and English) text is matched against an alias table, and the
result always carries *how* it matched. An exact hit draws immediately. A fuzzy
hit draws but says what it understood. A miss draws nothing and offers the
nearest things that do exist. The UI is expected to show that distinction —
same rule as [[gzowo-ai]]: never fake capability you do not have.

Exported to the browser as a JSON table by `--export`, so the site does this
lookup locally with no model and no network.
"""

import json
import re
import sys
import unicodedata
from difflib import SequenceMatcher

sys.path.insert(0, __file__.rsplit('/', 2)[0])

from data.categories import CATEGORIES

# Polish names for every category. First entry is the canonical label the UI
# shows back to the user; the rest are synonyms people actually type.
PL = {
    'The Eiffel Tower': ['wieża eiffla', 'eiffel'],
    'The Great Wall of China': ['wielki mur chiński', 'mur chiński'],
    'The Mona Lisa': ['mona lisa'],
    'aircraft carrier': ['lotniskowiec'],
    'airplane': ['samolot'],
    'alarm clock': ['budzik'],
    'ambulance': ['karetka', 'ambulans'],
    'angel': ['anioł'],
    'animal migration': ['migracja zwierząt', 'stado ptaków', 'klucz ptaków'],
    'ant': ['mrówka'],
    'anvil': ['kowadło'],
    'apple': ['jabłko'],
    'arm': ['ramię'],
    'asparagus': ['szparagi', 'szparag'],
    'axe': ['siekiera', 'topór'],
    'backpack': ['plecak'],
    'banana': ['banan'],
    'bandage': ['bandaż', 'plaster'],
    'barn': ['stodoła'],
    'baseball': ['piłka baseballowa', 'baseball'],
    'baseball bat': ['kij baseballowy'],
    'basket': ['koszyk', 'kosz'],
    'basketball': ['piłka do kosza', 'koszykówka'],
    'bat': ['nietoperz'],
    'bathtub': ['wanna'],
    'beach': ['plaża'],
    'bear': ['niedźwiedź'],
    'beard': ['broda'],
    'bed': ['łóżko'],
    'bee': ['pszczoła'],
    'belt': ['pasek'],
    'bench': ['ławka'],
    'bicycle': ['rower'],
    'binoculars': ['lornetka'],
    'bird': ['ptak'],
    'birthday cake': ['tort urodzinowy', 'tort'],
    'blackberry': ['jeżyna'],
    'blueberry': ['borówka', 'jagoda'],
    'book': ['książka'],
    'boomerang': ['bumerang'],
    'bottlecap': ['kapsel', 'nakrętka'],
    'bowtie': ['mucha', 'muszka'],
    'bracelet': ['bransoletka'],
    'brain': ['mózg'],
    'bread': ['chleb'],
    'bridge': ['most'],
    'broccoli': ['brokuł', 'brokuły'],
    'broom': ['miotła'],
    'bucket': ['wiadro'],
    'bulldozer': ['buldożer', 'spychacz'],
    'bus': ['autobus'],
    'bush': ['krzak', 'krzew'],
    'butterfly': ['motyl'],
    'cactus': ['kaktus'],
    'cake': ['ciasto'],
    'calculator': ['kalkulator'],
    'calendar': ['kalendarz'],
    'camel': ['wielbłąd'],
    'camera': ['aparat', 'aparat fotograficzny'],
    'camouflage': ['kamuflaż', 'moro'],
    'campfire': ['ognisko'],
    'candle': ['świeca', 'świeczka'],
    'cannon': ['armata', 'działo'],
    'canoe': ['kajak', 'kanu'],
    'car': ['samochód', 'auto'],
    'carrot': ['marchewka'],
    'castle': ['zamek'],
    'cat': ['kot', 'kotek', 'kicia'],
    'ceiling fan': ['wentylator sufitowy'],
    'cell phone': ['komórka', 'telefon komórkowy'],
    'cello': ['wiolonczela'],
    'chair': ['krzesło'],
    'chandelier': ['żyrandol'],
    'church': ['kościół'],
    'circle': ['koło', 'okrąg', 'kółko'],
    'clarinet': ['klarnet'],
    'clock': ['zegar'],
    'cloud': ['chmura', 'chmurka'],
    'coffee cup': ['filiżanka', 'kubek kawy'],
    'compass': ['kompas'],
    'computer': ['komputer'],
    'cookie': ['ciastko'],
    'cooler': ['lodówka turystyczna'],
    'couch': ['kanapa', 'sofa'],
    'cow': ['krowa'],
    'crab': ['krab'],
    'crayon': ['kredka'],
    'crocodile': ['krokodyl'],
    'crown': ['korona'],
    'cruise ship': ['statek', 'statek wycieczkowy'],
    'cup': ['kubek'],
    'diamond': ['diament', 'romb'],
    'dishwasher': ['zmywarka'],
    'diving board': ['trampolina'],
    'dog': ['pies', 'piesek'],
    'dolphin': ['delfin'],
    'donut': ['pączek', 'donut'],
    'door': ['drzwi'],
    'dragon': ['smok'],
    'dresser': ['komoda'],
    'drill': ['wiertarka'],
    'drums': ['perkusja', 'bębny'],
    'duck': ['kaczka'],
    'dumbbell': ['hantla', 'sztangielka'],
    'ear': ['ucho'],
    'elbow': ['łokieć'],
    'elephant': ['słoń'],
    'envelope': ['koperta'],
    'eraser': ['gumka'],
    'eye': ['oko'],
    'eyeglasses': ['okulary'],
    'face': ['twarz'],
    'fan': ['wentylator', 'wachlarz'],
    'feather': ['pióro'],
    'fence': ['płot', 'ogrodzenie'],
    'finger': ['palec'],
    'fire hydrant': ['hydrant'],
    'fireplace': ['kominek'],
    'firetruck': ['wóz strażacki', 'straż pożarna'],
    'fish': ['ryba'],
    'flamingo': ['flaming'],
    'flashlight': ['latarka'],
    'flip flops': ['klapki', 'japonki'],
    'floor lamp': ['lampa stojąca', 'lampa'],
    'flower': ['kwiat', 'kwiatek'],
    'flying saucer': ['latający spodek', 'ufo', 'spodek'],
    'foot': ['stopa'],
    'fork': ['widelec'],
    'frog': ['żaba'],
    'frying pan': ['patelnia'],
    'garden': ['ogród'],
    'garden hose': ['wąż ogrodowy'],
    'giraffe': ['żyrafa'],
    'goatee': ['kozia bródka'],
    'golf club': ['kij golfowy'],
    'grapes': ['winogrona'],
    'grass': ['trawa'],
    'guitar': ['gitara'],
    'hamburger': ['hamburger', 'burger'],
    'hammer': ['młotek'],
    'hand': ['dłoń', 'ręka'],
    'harp': ['harfa'],
    'hat': ['kapelusz', 'czapka'],
    'headphones': ['słuchawki'],
    'hedgehog': ['jeż'],
    'helicopter': ['helikopter', 'śmigłowiec'],
    'helmet': ['kask', 'hełm'],
    'hexagon': ['sześciokąt'],
    'hockey puck': ['krążek hokejowy'],
    'hockey stick': ['kij hokejowy'],
    'horse': ['koń'],
    'hospital': ['szpital'],
    'hot air balloon': ['balon'],
    'hot dog': ['hot dog', 'parówka w bułce'],
    'hot tub': ['jacuzzi'],
    'hourglass': ['klepsydra'],
    'house': ['dom'],
    'house plant': ['roślina doniczkowa', 'kwiatek w doniczce'],
    'hurricane': ['huragan'],
    'ice cream': ['lody'],
    'jacket': ['kurtka'],
    'jail': ['więzienie', 'kraty'],
    'kangaroo': ['kangur'],
    'key': ['klucz'],
    'keyboard': ['klawiatura'],
    'knee': ['kolano'],
    'knife': ['nóż'],
    'ladder': ['drabina'],
    'lantern': ['lampion', 'latarnia'],
    'laptop': ['laptop'],
    'leaf': ['liść'],
    'leg': ['noga'],
    'light bulb': ['żarówka'],
    'lighter': ['zapalniczka'],
    'lighthouse': ['latarnia morska'],
    'lightning': ['błyskawica', 'piorun'],
    'line': ['linia', 'kreska'],
    'lion': ['lew'],
    'lipstick': ['szminka'],
    'lobster': ['homar'],
    'lollipop': ['lizak'],
    'mailbox': ['skrzynka pocztowa'],
    'map': ['mapa'],
    'marker': ['marker', 'flamaster'],
    'matches': ['zapałki'],
    'megaphone': ['megafon'],
    'mermaid': ['syrena'],
    'microphone': ['mikrofon'],
    'microwave': ['mikrofalówka'],
    'monkey': ['małpa'],
    'moon': ['księżyc'],
    'mosquito': ['komar'],
    'motorbike': ['motocykl', 'motor'],
    'mountain': ['góra', 'góry'],
    'mouse': ['mysz'],
    'moustache': ['wąsy'],
    'mouth': ['usta'],
    'mug': ['kubek z uchem'],
    'mushroom': ['grzyb'],
    'nail': ['gwóźdź'],
    'necklace': ['naszyjnik'],
    'nose': ['nos'],
    'ocean': ['ocean', 'morze'],
    'octagon': ['ośmiokąt'],
    'octopus': ['ośmiornica'],
    'onion': ['cebula'],
    'oven': ['piekarnik'],
    'owl': ['sowa'],
    'paint can': ['puszka farby'],
    'paintbrush': ['pędzel'],
    'palm tree': ['palma'],
    'panda': ['panda'],
    'pants': ['spodnie'],
    'paper clip': ['spinacz'],
    'parachute': ['spadochron'],
    'parrot': ['papuga'],
    'passport': ['paszport'],
    'peanut': ['orzeszek', 'fistaszek'],
    'pear': ['gruszka'],
    'peas': ['groszek', 'groch'],
    'pencil': ['ołówek'],
    'penguin': ['pingwin'],
    'piano': ['pianino', 'fortepian'],
    'pickup truck': ['pickup'],
    'picture frame': ['ramka', 'rama obrazu'],
    'pig': ['świnia'],
    'pillow': ['poduszka'],
    'pineapple': ['ananas'],
    'pizza': ['pizza'],
    'pliers': ['kombinerki', 'szczypce'],
    'police car': ['radiowóz', 'policja'],
    'pond': ['staw'],
    'pool': ['basen'],
    'popsicle': ['lód na patyku'],
    'postcard': ['pocztówka'],
    'potato': ['ziemniak'],
    'power outlet': ['gniazdko', 'kontakt'],
    'purse': ['torebka'],
    'rabbit': ['królik'],
    'raccoon': ['szop'],
    'radio': ['radio'],
    'rain': ['deszcz'],
    'rainbow': ['tęcza'],
    'rake': ['grabie'],
    'remote control': ['pilot'],
    'rhinoceros': ['nosorożec'],
    'rifle': ['karabin', 'strzelba'],
    'river': ['rzeka'],
    'roller coaster': ['kolejka górska', 'rollercoaster'],
    'rollerskates': ['wrotki'],
    'sailboat': ['żaglówka'],
    'sandwich': ['kanapka'],
    'saw': ['piła'],
    'saxophone': ['saksofon'],
    'school bus': ['autobus szkolny'],
    'scissors': ['nożyczki'],
    'scorpion': ['skorpion'],
    'screwdriver': ['śrubokręt', 'wkrętak'],
    'sea turtle': ['żółw morski', 'żółw'],
    'see saw': ['huśtawka wagowa'],
    'shark': ['rekin'],
    'sheep': ['owca'],
    'shoe': ['but'],
    'shorts': ['szorty', 'krótkie spodenki'],
    'shovel': ['łopata'],
    'sink': ['zlew', 'umywalka'],
    'skateboard': ['deskorolka'],
    'skull': ['czaszka'],
    'skyscraper': ['wieżowiec', 'drapacz chmur'],
    'sleeping bag': ['śpiwór'],
    'smiley face': ['buźka', 'uśmiech'],
    'snail': ['ślimak'],
    'snake': ['wąż'],
    'snorkel': ['rurka do nurkowania', 'fajka'],
    'snowflake': ['płatek śniegu'],
    'snowman': ['bałwan'],
    'soccer ball': ['piłka nożna', 'piłka'],
    'sock': ['skarpetka'],
    'speedboat': ['motorówka'],
    'spider': ['pająk'],
    'spoon': ['łyżka'],
    'spreadsheet': ['arkusz kalkulacyjny', 'tabelka'],
    'square': ['kwadrat'],
    'squiggle': ['zawijas', 'bazgroł'],
    'squirrel': ['wiewiórka'],
    'stairs': ['schody'],
    'star': ['gwiazda', 'gwiazdka'],
    'steak': ['stek'],
    'stereo': ['wieża stereo', 'głośnik'],
    'stethoscope': ['stetoskop'],
    'stitches': ['szwy'],
    'stop sign': ['znak stop'],
    'stove': ['kuchenka'],
    'strawberry': ['truskawka'],
    'streetlight': ['latarnia uliczna'],
    'string bean': ['fasolka szparagowa'],
    'submarine': ['łódź podwodna'],
    'suitcase': ['walizka'],
    'sun': ['słońce'],
    'swan': ['łabędź'],
    'sweater': ['sweter'],
    'swing set': ['huśtawka'],
    'sword': ['miecz'],
    'syringe': ['strzykawka'],
    't-shirt': ['koszulka', 't-shirt'],
    'table': ['stół'],
    'teapot': ['czajniczek', 'dzbanek'],
    'teddy-bear': ['miś', 'pluszowy miś'],
    'telephone': ['telefon'],
    'television': ['telewizor'],
    'tennis racquet': ['rakieta tenisowa'],
    'tent': ['namiot'],
    'tiger': ['tygrys'],
    'toaster': ['toster'],
    'toe': ['palec u nogi'],
    'toilet': ['toaleta', 'kibel'],
    'tooth': ['ząb'],
    'toothbrush': ['szczoteczka do zębów'],
    'toothpaste': ['pasta do zębów'],
    'tornado': ['tornado', 'trąba powietrzna'],
    'tractor': ['traktor'],
    'traffic light': ['sygnalizacja świetlna', 'światła'],
    'train': ['pociąg'],
    'tree': ['drzewo', 'choinka'],
    'triangle': ['trójkąt'],
    'trombone': ['puzon'],
    'truck': ['ciężarówka'],
    'trumpet': ['trąbka'],
    'umbrella': ['parasol'],
    'underwear': ['bielizna', 'majtki'],
    'van': ['furgonetka', 'van'],
    'vase': ['wazon'],
    'violin': ['skrzypce'],
    'washing machine': ['pralka'],
    'watermelon': ['arbuz'],
    'waterslide': ['zjeżdżalnia wodna'],
    'whale': ['wieloryb'],
    'wheel': ['koło od auta', 'opona'],
    'windmill': ['wiatrak'],
    'wine bottle': ['butelka wina'],
    'wine glass': ['kieliszek'],
    'wristwatch': ['zegarek'],
    'yoga': ['joga'],
    'zebra': ['zebra'],
    'zigzag': ['zygzak'],
}

# Things people will ask for that QuickDraw genuinely does not contain. Saying
# so beats drawing the nearest neighbour and letting the user think the model
# tried. Values are the honest fallbacks to offer.
#
# Only genuinely absent concepts belong here. A compound request whose head noun
# *does* exist ("rower górski", "auto wyścigowe") must not be listed — the
# substring matcher already resolves those to the base category, and listing
# them here would shadow it and refuse a drawing the model can perfectly well
# make.
NOT_IN_DATASET = {
    'rakieta': ['helicopter', 'flying saucer', 'hot air balloon'],
    'robot': ['computer', 'television', 'face'],
    'dinozaur': ['dragon', 'crocodile'],
    'jednorożec': ['horse', 'zebra'],
    'człowiek': ['face', 'smiley face', 'arm'],
    'ludzik': ['face', 'smiley face'],
    'serce': ['diamond', 'circle'],
    'polska': ['map'],
    'flaga': ['map', 'postcard'],
    'litera': ['line', 'squiggle'],
    'gwiazda śmierci': ['circle', 'moon'],
}

_WORD = re.compile(r'[^a-z0-9 ]+')
_FILLER = {'narysuj', 'narysuj mi', 'pokaz', 'pokaż', 'chce', 'chcę', 'prosze',
           'proszę', 'jakis', 'jakiś', 'jakas', 'jakaś', 'mi', 'a', 'draw',
           'please', 'the', 'an'}


def normalize(text):
    """Lowercase, strip Polish diacritics and punctuation, drop filler words."""
    text = unicodedata.normalize('NFKD', str(text).lower())
    text = text.replace('ł', 'l')
    text = ''.join(c for c in text if not unicodedata.combining(c))
    text = _WORD.sub(' ', text)
    words = [w for w in text.split() if w and w not in _FILLER]
    return ' '.join(words)


def _index():
    """alias (normalised) -> category, longest alias first so 'kij golfowy'
    is not shadowed by 'kij'."""
    table = {}
    for category in CATEGORIES:
        table.setdefault(normalize(category), category)
        for alias in PL.get(category, []):
            table.setdefault(normalize(alias), category)
    return table


INDEX = _index()


def resolve(text, threshold=0.72):
    """Map free text to a category.

    Returns a dict with `category` (None when nothing is close enough), `match`
    ('exact' | 'contains' | 'fuzzy' | 'missing' | 'none'), `label` — the Polish
    name to echo back — and `alternatives`.
    """
    q = normalize(text)
    if not q:
        return {'category': None, 'match': 'none', 'label': None, 'alternatives': []}

    if q in INDEX:
        cat = INDEX[q]
        return {'category': cat, 'match': 'exact', 'label': label_of(cat),
                'score': 1.0, 'alternatives': []}

    for miss, suggestions in NOT_IN_DATASET.items():
        if normalize(miss) == q:
            return {'category': None, 'match': 'missing', 'label': None,
                    'alternatives': suggestions}

    contained = [(alias, cat) for alias, cat in INDEX.items()
                 if alias in q or q in alias]
    if contained:
        alias, cat = max(contained, key=lambda kv: len(kv[0]))
        return {'category': cat, 'match': 'contains', 'label': label_of(cat),
                'score': 0.9, 'alternatives': []}

    scored = sorted(
        ((SequenceMatcher(None, q, alias).ratio(), cat) for alias, cat in INDEX.items()),
        reverse=True,
    )
    best_score, best_cat = scored[0]
    seen, alts = set(), []
    for score, cat in scored:
        if cat not in seen:
            seen.add(cat)
            alts.append(cat)
        if len(alts) == 4:
            break

    if best_score >= threshold:
        return {'category': best_cat, 'match': 'fuzzy', 'label': label_of(best_cat),
                'score': round(best_score, 3), 'alternatives': alts[1:]}
    return {'category': None, 'match': 'none', 'label': None,
            'score': round(best_score, 3), 'alternatives': alts}


def label_of(category):
    """The Polish name to show a person for an English category id."""
    aliases = PL.get(category)
    return aliases[0] if aliases else category


def export(path):
    """Write the lookup table the browser uses, so the site needs no Python."""
    payload = {
        'categories': CATEGORIES,
        'labels': {c: label_of(c) for c in CATEGORIES},
        'aliases': {alias: cat for alias, cat in INDEX.items()},
        'missing': NOT_IN_DATASET,
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    return payload


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == '--export':
        p = export(sys.argv[2])
        print(f"exported {len(p['aliases'])} aliases for {len(p['categories'])} categories "
              f"-> {sys.argv[2]}")
        raise SystemExit(0)

    missing = [c for c in CATEGORIES if c not in PL]
    print(f'categories without a Polish name: {len(missing)} {missing[:5]}')
    print(f'alias index: {len(INDEX)} entries\n')

    for probe in ['kot', 'narysuj mi smoka', 'pianino', 'rower górski', 'ośmiornica',
                  'rakieta', 'zamek', 'kotek', 'samochud', 'pizza', 'wieża eiffla',
                  'jakiś ptak', 'blablabla']:
        r = resolve(probe)
        got = r['category'] or '—'
        print(f"{probe:<20} -> {got:<18} [{r['match']}] {r.get('alternatives') or ''}")
