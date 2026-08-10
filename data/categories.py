"""
Every category G-Doodle is trained on — all 345 in Google's QuickDraw release.

The list is committed rather than fetched at run time so a training run stays
reproducible even if the bucket changes underneath it. It was read from the
bucket listing, not typed from memory, and the order is the sorted order the
tokenizer assigns category tokens in — reordering it invalidates every existing
checkpoint.

Worth knowing before anyone asks for something else: **QuickDraw has no `rocket`.**
It was the obvious first pick for this project and it is simply not among the
345 classes Google collected. `helicopter`, `flying saucer`, `hot air balloon`
and `parachute` are the nearest things that fly.

The 345 are the model's entire world. Free text typed into the site is resolved
to one of these by `runtime/prompt.py`; nothing outside the list can be drawn,
and the UI is expected to say so rather than silently draw the wrong thing.
"""

CATEGORIES = [
    'The Eiffel Tower', 'The Great Wall of China', 'The Mona Lisa',
    'aircraft carrier', 'airplane', 'alarm clock', 'ambulance', 'angel',
    'animal migration', 'ant', 'anvil', 'apple', 'arm', 'asparagus', 'axe',
    'backpack', 'banana', 'bandage', 'barn', 'baseball', 'baseball bat',
    'basket', 'basketball', 'bat', 'bathtub', 'beach', 'bear', 'beard', 'bed',
    'bee', 'belt', 'bench', 'bicycle', 'binoculars', 'bird', 'birthday cake',
    'blackberry', 'blueberry', 'book', 'boomerang', 'bottlecap', 'bowtie',
    'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom', 'bucket',
    'bulldozer', 'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator',
    'calendar', 'camel', 'camera', 'camouflage', 'campfire', 'candle', 'cannon',
    'canoe', 'car', 'carrot', 'castle', 'cat', 'ceiling fan', 'cell phone',
    'cello', 'chair', 'chandelier', 'church', 'circle', 'clarinet', 'clock',
    'cloud', 'coffee cup', 'compass', 'computer', 'cookie', 'cooler', 'couch',
    'cow', 'crab', 'crayon', 'crocodile', 'crown', 'cruise ship', 'cup',
    'diamond', 'dishwasher', 'diving board', 'dog', 'dolphin', 'donut', 'door',
    'dragon', 'dresser', 'drill', 'drums', 'duck', 'dumbbell', 'ear', 'elbow',
    'elephant', 'envelope', 'eraser', 'eye', 'eyeglasses', 'face', 'fan',
    'feather', 'fence', 'finger', 'fire hydrant', 'fireplace', 'firetruck',
    'fish', 'flamingo', 'flashlight', 'flip flops', 'floor lamp', 'flower',
    'flying saucer', 'foot', 'fork', 'frog', 'frying pan', 'garden',
    'garden hose', 'giraffe', 'goatee', 'golf club', 'grapes', 'grass',
    'guitar', 'hamburger', 'hammer', 'hand', 'harp', 'hat', 'headphones',
    'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey puck', 'hockey stick',
    'horse', 'hospital', 'hot air balloon', 'hot dog', 'hot tub', 'hourglass',
    'house', 'house plant', 'hurricane', 'ice cream', 'jacket', 'jail',
    'kangaroo', 'key', 'keyboard', 'knee', 'knife', 'ladder', 'lantern',
    'laptop', 'leaf', 'leg', 'light bulb', 'lighter', 'lighthouse', 'lightning',
    'line', 'lion', 'lipstick', 'lobster', 'lollipop', 'mailbox', 'map',
    'marker', 'matches', 'megaphone', 'mermaid', 'microphone', 'microwave',
    'monkey', 'moon', 'mosquito', 'motorbike', 'mountain', 'mouse', 'moustache',
    'mouth', 'mug', 'mushroom', 'nail', 'necklace', 'nose', 'ocean', 'octagon',
    'octopus', 'onion', 'oven', 'owl', 'paint can', 'paintbrush', 'palm tree',
    'panda', 'pants', 'paper clip', 'parachute', 'parrot', 'passport', 'peanut',
    'pear', 'peas', 'pencil', 'penguin', 'piano', 'pickup truck',
    'picture frame', 'pig', 'pillow', 'pineapple', 'pizza', 'pliers',
    'police car', 'pond', 'pool', 'popsicle', 'postcard', 'potato',
    'power outlet', 'purse', 'rabbit', 'raccoon', 'radio', 'rain', 'rainbow',
    'rake', 'remote control', 'rhinoceros', 'rifle', 'river', 'roller coaster',
    'rollerskates', 'sailboat', 'sandwich', 'saw', 'saxophone', 'school bus',
    'scissors', 'scorpion', 'screwdriver', 'sea turtle', 'see saw', 'shark',
    'sheep', 'shoe', 'shorts', 'shovel', 'sink', 'skateboard', 'skull',
    'skyscraper', 'sleeping bag', 'smiley face', 'snail', 'snake', 'snorkel',
    'snowflake', 'snowman', 'soccer ball', 'sock', 'speedboat', 'spider',
    'spoon', 'spreadsheet', 'square', 'squiggle', 'squirrel', 'stairs', 'star',
    'steak', 'stereo', 'stethoscope', 'stitches', 'stop sign', 'stove',
    'strawberry', 'streetlight', 'string bean', 'submarine', 'suitcase', 'sun',
    'swan', 'sweater', 'swing set', 'sword', 'syringe', 't-shirt', 'table',
    'teapot', 'teddy-bear', 'telephone', 'television', 'tennis racquet', 'tent',
    'tiger', 'toaster', 'toe', 'toilet', 'tooth', 'toothbrush', 'toothpaste',
    'tornado', 'tractor', 'traffic light', 'train', 'tree', 'triangle',
    'trombone', 'truck', 'trumpet', 'umbrella', 'underwear', 'van', 'vase',
    'violin', 'washing machine', 'watermelon', 'waterslide', 'whale', 'wheel',
    'windmill', 'wine bottle', 'wine glass', 'wristwatch', 'yoga', 'zebra',
    'zigzag',
]

assert len(CATEGORIES) == 345, f'expected 345 categories, got {len(CATEGORIES)}'
assert len(CATEGORIES) == len(set(CATEGORIES)), 'duplicate category'
assert CATEGORIES == sorted(CATEGORIES, key=lambda s: s), 'order must stay sorted'
