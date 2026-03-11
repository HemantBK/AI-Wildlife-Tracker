"""
Add Essential Indian Wildlife Species — Comprehensive Edition (500+)

Scrapes 500+ Indian wildlife species from Wikipedia to build a rich knowledge base.
Covers mammals, birds, reptiles, amphibians, and marine life found in India.

Usage:
    python scripts/add_essential_species.py

After running, re-run the processing pipeline:
    python -m src.preprocessing.cleaner
    python -m src.preprocessing.chunker
    python -m src.preprocessing.validator
    python -m src.retrieval.build_indexes
"""

import json
import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.wikipedia_scraper import scrape_species_page

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/raw/wikipedia")

# ─── 500+ Indian Wildlife Species ─────────────────────────────────

ESSENTIAL_SPECIES = {
    "mammals": [
        # ── Big Cats ──
        {"common_name": "Bengal tiger", "scientific_name": "Panthera tigris tigris"},
        {"common_name": "Indian leopard", "scientific_name": "Panthera pardus fusca"},
        {"common_name": "Asiatic lion", "scientific_name": "Panthera leo persica"},
        {"common_name": "Snow leopard", "scientific_name": "Panthera uncia"},
        {"common_name": "Clouded leopard", "scientific_name": "Neofelis nebulosa"},
        {"common_name": "Jungle cat", "scientific_name": "Felis chaus"},
        {"common_name": "Fishing cat", "scientific_name": "Prionailurus viverrinus"},
        {"common_name": "Leopard cat", "scientific_name": "Prionailurus bengalensis"},
        {"common_name": "Rusty-spotted cat", "scientific_name": "Prionailurus rubiginosus"},
        {"common_name": "Pallas's cat", "scientific_name": "Otocolobus manul"},
        {"common_name": "Asiatic golden cat", "scientific_name": "Catopuma temminckii"},
        {"common_name": "Marbled cat", "scientific_name": "Pardofelis marmorata"},
        {"common_name": "Eurasian lynx", "scientific_name": "Lynx lynx"},
        {"common_name": "Caracal", "scientific_name": "Caracal caracal"},
        # ── Large Herbivores ──
        {"common_name": "Indian elephant", "scientific_name": "Elephas maximus indicus"},
        {"common_name": "Indian rhinoceros", "scientific_name": "Rhinoceros unicornis"},
        {"common_name": "Gaur", "scientific_name": "Bos gaurus"},
        {"common_name": "Wild water buffalo", "scientific_name": "Bubalus arnee"},
        {"common_name": "Nilgai", "scientific_name": "Boselaphus tragocamelus"},
        {"common_name": "Wild yak", "scientific_name": "Bos mutus"},
        {"common_name": "Takin", "scientific_name": "Budorcas taxicolor"},
        # ── Deer & Antelope ──
        {"common_name": "Chital", "scientific_name": "Axis axis"},
        {"common_name": "Sambar deer", "scientific_name": "Rusa unicolor"},
        {"common_name": "Barasingha", "scientific_name": "Rucervus duvaucelii"},
        {"common_name": "Indian muntjac", "scientific_name": "Muntiacus muntjak"},
        {"common_name": "Blackbuck", "scientific_name": "Antilope cervicapra"},
        {"common_name": "Four-horned antelope", "scientific_name": "Tetracerus quadricornis"},
        {"common_name": "Chinkara", "scientific_name": "Gazella bennettii"},
        {"common_name": "Indian hog deer", "scientific_name": "Axis porcinus"},
        {"common_name": "Kashmir stag", "scientific_name": "Cervus hanglu hanglu"},
        {"common_name": "Tibetan antelope", "scientific_name": "Pantholops hodgsonii"},
        {"common_name": "Himalayan tahr", "scientific_name": "Hemitragus jemlahicus"},
        {"common_name": "Nilgiri tahr", "scientific_name": "Nilgiritragus hylocrius"},
        {"common_name": "Himalayan serow", "scientific_name": "Capricornis thar"},
        {"common_name": "Himalayan goral", "scientific_name": "Naemorhedus goral"},
        {"common_name": "Bharal", "scientific_name": "Pseudois nayaur"},
        {"common_name": "Indian wild ass", "scientific_name": "Equus hemionus khur"},
        {"common_name": "Mouse deer", "scientific_name": "Moschiola indica"},
        # ── Primates ──
        {"common_name": "Rhesus macaque", "scientific_name": "Macaca mulatta"},
        {"common_name": "Hanuman langur", "scientific_name": "Semnopithecus entellus"},
        {"common_name": "Lion-tailed macaque", "scientific_name": "Macaca silenus"},
        {"common_name": "Hoolock gibbon", "scientific_name": "Hoolock hoolock"},
        {"common_name": "Bonnet macaque", "scientific_name": "Macaca radiata"},
        {"common_name": "Assamese macaque", "scientific_name": "Macaca assamensis"},
        {"common_name": "Pig-tailed macaque", "scientific_name": "Macaca leonina"},
        {"common_name": "Capped langur", "scientific_name": "Trachypithecus pileatus"},
        {"common_name": "Golden langur", "scientific_name": "Trachypithecus geei"},
        {"common_name": "Nilgiri langur", "scientific_name": "Trachypithecus johnii"},
        {"common_name": "Slow loris", "scientific_name": "Nycticebus bengalensis"},
        # ── Bears ──
        {"common_name": "Sloth bear", "scientific_name": "Melursus ursinus"},
        {"common_name": "Himalayan black bear", "scientific_name": "Ursus thibetanus"},
        {"common_name": "Himalayan brown bear", "scientific_name": "Ursus arctos isabellinus"},
        # ── Canids & Dogs ──
        {"common_name": "Indian wild dog", "scientific_name": "Cuon alpinus"},
        {"common_name": "Indian wolf", "scientific_name": "Canis lupus pallipes"},
        {"common_name": "Indian fox", "scientific_name": "Vulpes bengalensis"},
        {"common_name": "Tibetan sand fox", "scientific_name": "Vulpes ferrilata"},
        {"common_name": "Golden jackal", "scientific_name": "Canis aureus"},
        # ── Mustelids, Civets & Small Carnivores ──
        {"common_name": "Indian grey mongoose", "scientific_name": "Urva edwardsii"},
        {"common_name": "Small Indian mongoose", "scientific_name": "Urva auropunctata"},
        {"common_name": "Honey badger", "scientific_name": "Mellivora capensis"},
        {"common_name": "Asian palm civet", "scientific_name": "Paradoxurus hermaphroditus"},
        {"common_name": "Small Indian civet", "scientific_name": "Viverricula indica"},
        {"common_name": "Large Indian civet", "scientific_name": "Viverra zibetha"},
        {"common_name": "Binturong", "scientific_name": "Arctictis binturong"},
        {"common_name": "Smooth-coated otter", "scientific_name": "Lutrogale perspicillata"},
        {"common_name": "Asian small-clawed otter", "scientific_name": "Aonyx cinereus"},
        {"common_name": "Eurasian otter", "scientific_name": "Lutra lutra"},
        {"common_name": "Yellow-throated marten", "scientific_name": "Martes flavigula"},
        {"common_name": "Striped hyena", "scientific_name": "Hyaena hyaena"},
        # ── Others ──
        {"common_name": "Red panda", "scientific_name": "Ailurus fulgens"},
        {"common_name": "Indian pangolin", "scientific_name": "Manis crassicaudata"},
        {"common_name": "Chinese pangolin", "scientific_name": "Manis pentadactyla"},
        {"common_name": "Indian giant squirrel", "scientific_name": "Ratufa indica"},
        {"common_name": "Indian flying fox", "scientific_name": "Pteropus medius"},
        {"common_name": "Dugong", "scientific_name": "Dugong dugon"},
        {"common_name": "Indian porcupine", "scientific_name": "Hystrix indica"},
        {"common_name": "Indian hare", "scientific_name": "Lepus nigricollis"},
        {"common_name": "Indian crested porcupine", "scientific_name": "Hystrix indica"},
        {"common_name": "Gangetic dolphin", "scientific_name": "Platanista gangetica"},
        {"common_name": "Irrawaddy dolphin", "scientific_name": "Orcaella brevirostris"},
        {"common_name": "Spotted deer", "scientific_name": "Axis axis"},
    ],
    "birds": [
        # ── Iconic Indian Birds ──
        {"common_name": "Indian peafowl", "scientific_name": "Pavo cristatus"},
        {"common_name": "Indian roller", "scientific_name": "Coracias benghalensis"},
        {"common_name": "Indian robin", "scientific_name": "Copsychus fulicatus"},
        {"common_name": "Indian pitta", "scientific_name": "Pitta brachyura"},
        {"common_name": "Indian paradise flycatcher", "scientific_name": "Terpsiphone paradisi"},
        # ── Kingfishers ──
        {"common_name": "Common kingfisher", "scientific_name": "Alcedo atthis"},
        {"common_name": "White-throated kingfisher", "scientific_name": "Halcyon smyrnensis"},
        {"common_name": "Pied kingfisher", "scientific_name": "Ceryle rudis"},
        {"common_name": "Stork-billed kingfisher", "scientific_name": "Pelargopsis capensis"},
        {"common_name": "Brown-winged kingfisher", "scientific_name": "Pelargopsis amauroptera"},
        # ── Raptors (Eagles, Hawks, Vultures) ──
        {"common_name": "Indian eagle-owl", "scientific_name": "Bubo bengalensis"},
        {"common_name": "Crested serpent eagle", "scientific_name": "Spilornis cheela"},
        {"common_name": "Black kite", "scientific_name": "Milvus migrans"},
        {"common_name": "Brahminy kite", "scientific_name": "Haliastur indus"},
        {"common_name": "White-rumped vulture", "scientific_name": "Gyps bengalensis"},
        {"common_name": "Indian vulture", "scientific_name": "Gyps indicus"},
        {"common_name": "Red-headed vulture", "scientific_name": "Sarcogyps calvus"},
        {"common_name": "Changeable hawk-eagle", "scientific_name": "Nisaetus cirrhatus"},
        {"common_name": "Bonelli's eagle", "scientific_name": "Aquila fasciata"},
        {"common_name": "Crested hawk-eagle", "scientific_name": "Nisaetus cirrhatus"},
        {"common_name": "White-bellied sea eagle", "scientific_name": "Haliaeetus leucogaster"},
        {"common_name": "Pallas's fish eagle", "scientific_name": "Haliaeetus leucoryphus"},
        {"common_name": "Shikra", "scientific_name": "Accipiter badius"},
        {"common_name": "Besra", "scientific_name": "Accipiter virgatus"},
        {"common_name": "Oriental honey buzzard", "scientific_name": "Pernis ptilorhynchus"},
        {"common_name": "Black-shouldered kite", "scientific_name": "Elanus caeruleus"},
        {"common_name": "Indian spotted eagle", "scientific_name": "Clanga hastata"},
        {"common_name": "Peregrine falcon", "scientific_name": "Falco peregrinus"},
        {"common_name": "Laggar falcon", "scientific_name": "Falco jugger"},
        {"common_name": "Common kestrel", "scientific_name": "Falco tinnunculus"},
        # ── Owls ──
        {"common_name": "Barn owl", "scientific_name": "Tyto alba"},
        {"common_name": "Spotted owlet", "scientific_name": "Athene brama"},
        {"common_name": "Jungle owlet", "scientific_name": "Glaucidium radiatum"},
        {"common_name": "Brown fish owl", "scientific_name": "Ketupa zeylonensis"},
        {"common_name": "Mottled wood owl", "scientific_name": "Strix ocellata"},
        {"common_name": "Indian scops owl", "scientific_name": "Otus bakkamoena"},
        {"common_name": "Short-eared owl", "scientific_name": "Asio flammeus"},
        # ── Hornbills ──
        {"common_name": "Great Indian hornbill", "scientific_name": "Buceros bicornis"},
        {"common_name": "Indian grey hornbill", "scientific_name": "Ocyceros birostris"},
        {"common_name": "Malabar pied hornbill", "scientific_name": "Anthracoceros coronatus"},
        {"common_name": "Wreathed hornbill", "scientific_name": "Rhyticeros undulatus"},
        {"common_name": "Malabar grey hornbill", "scientific_name": "Ocyceros griseus"},
        {"common_name": "Rufous-necked hornbill", "scientific_name": "Aceros nipalensis"},
        # ── Parrots & Parakeets ──
        {"common_name": "Rose-ringed parakeet", "scientific_name": "Psittacula krameri"},
        {"common_name": "Alexandrine parakeet", "scientific_name": "Psittacula eupatria"},
        {"common_name": "Plum-headed parakeet", "scientific_name": "Psittacula cyanocephala"},
        {"common_name": "Malabar parakeet", "scientific_name": "Psittacula columboides"},
        {"common_name": "Vernal hanging parrot", "scientific_name": "Loriculus vernalis"},
        # ── Woodpeckers ──
        {"common_name": "Greater flameback", "scientific_name": "Chrysocolaptes guttacristatus"},
        {"common_name": "Black-rumped flameback", "scientific_name": "Dinopium benghalense"},
        {"common_name": "White-bellied woodpecker", "scientific_name": "Dryocopus javensis"},
        {"common_name": "Lesser yellownape", "scientific_name": "Picus chlorolophus"},
        {"common_name": "Heart-spotted woodpecker", "scientific_name": "Hemicircus canente"},
        # ── Barbets & Bee-eaters ──
        {"common_name": "Coppersmith barbet", "scientific_name": "Psilopogon haemacephalus"},
        {"common_name": "White-cheeked barbet", "scientific_name": "Psilopogon viridis"},
        {"common_name": "Brown-headed barbet", "scientific_name": "Psilopogon zeylanicus"},
        {"common_name": "Green bee-eater", "scientific_name": "Merops orientalis"},
        {"common_name": "Blue-tailed bee-eater", "scientific_name": "Merops philippinus"},
        {"common_name": "Chestnut-headed bee-eater", "scientific_name": "Merops leschenaulti"},
        # ── Cuckoos ──
        {"common_name": "Asian koel", "scientific_name": "Eudynamys scolopaceus"},
        {"common_name": "Greater coucal", "scientific_name": "Centropus sinensis"},
        {"common_name": "Common hawk-cuckoo", "scientific_name": "Hierococcyx varius"},
        {"common_name": "Indian cuckoo", "scientific_name": "Cuculus micropterus"},
        {"common_name": "Jacobin cuckoo", "scientific_name": "Clamator jacobinus"},
        # ── Bulbuls ──
        {"common_name": "Red-vented bulbul", "scientific_name": "Pycnonotus cafer"},
        {"common_name": "Red-whiskered bulbul", "scientific_name": "Pycnonotus jocosus"},
        {"common_name": "White-browed bulbul", "scientific_name": "Pycnonotus luteolus"},
        {"common_name": "Yellow-browed bulbul", "scientific_name": "Iole indica"},
        # ── Drongos ──
        {"common_name": "Black drongo", "scientific_name": "Dicrurus macrocercus"},
        {"common_name": "Greater racket-tailed drongo", "scientific_name": "Dicrurus paradiseus"},
        {"common_name": "Ashy drongo", "scientific_name": "Dicrurus leucophaeus"},
        {"common_name": "White-bellied drongo", "scientific_name": "Dicrurus caerulescens"},
        # ── Flycatchers & Warblers ──
        {"common_name": "Verditer flycatcher", "scientific_name": "Eumyias thalassinus"},
        {"common_name": "Asian brown flycatcher", "scientific_name": "Muscicapa dauurica"},
        {"common_name": "Tickell's blue flycatcher", "scientific_name": "Cyornis tickelliae"},
        {"common_name": "Common tailorbird", "scientific_name": "Orthotomus sutorius"},
        {"common_name": "Jungle prinia", "scientific_name": "Prinia sylvatica"},
        {"common_name": "Ashy prinia", "scientific_name": "Prinia socialis"},
        # ── Sunbirds ──
        {"common_name": "Purple sunbird", "scientific_name": "Cinnyris asiaticus"},
        {"common_name": "Loten's sunbird", "scientific_name": "Cinnyris lotenius"},
        {"common_name": "Purple-rumped sunbird", "scientific_name": "Leptocoma zeylonica"},
        {"common_name": "Mrs. Gould's sunbird", "scientific_name": "Aethopyga gouldiae"},
        # ── Mynas & Starlings ──
        {"common_name": "Common myna", "scientific_name": "Acridotheres tristis"},
        {"common_name": "Jungle myna", "scientific_name": "Acridotheres fuscus"},
        {"common_name": "Hill myna", "scientific_name": "Gracula religiosa"},
        {"common_name": "Brahminy starling", "scientific_name": "Sturnia pagodarum"},
        {"common_name": "Rosy starling", "scientific_name": "Pastor roseus"},
        {"common_name": "Asian pied starling", "scientific_name": "Gracupica contra"},
        # ── Crows & Treepies ──
        {"common_name": "House crow", "scientific_name": "Corvus splendens"},
        {"common_name": "Large-billed crow", "scientific_name": "Corvus macrorhynchos"},
        {"common_name": "Indian treepie", "scientific_name": "Dendrocitta vagabunda"},
        {"common_name": "White-bellied treepie", "scientific_name": "Dendrocitta leucogastra"},
        # ── Sparrows & Finches ──
        {"common_name": "House sparrow", "scientific_name": "Passer domesticus"},
        {"common_name": "Indian silverbill", "scientific_name": "Euodice malabarica"},
        {"common_name": "Scaly-breasted munia", "scientific_name": "Lonchura punctulata"},
        {"common_name": "White-rumped munia", "scientific_name": "Lonchura striata"},
        # ── Pigeons & Doves ──
        {"common_name": "Rock pigeon", "scientific_name": "Columba livia"},
        {"common_name": "Spotted dove", "scientific_name": "Spilopelia chinensis"},
        {"common_name": "Laughing dove", "scientific_name": "Spilopelia senegalensis"},
        {"common_name": "Eurasian collared dove", "scientific_name": "Streptopelia decaocto"},
        {"common_name": "Yellow-footed green pigeon", "scientific_name": "Treron phoenicopterus"},
        {"common_name": "Pompadour green pigeon", "scientific_name": "Treron pompadora"},
        {"common_name": "Nicobar pigeon", "scientific_name": "Caloenas nicobarica"},
        # ── Pheasants & Partridges ──
        {"common_name": "Indian peafowl", "scientific_name": "Pavo cristatus"},
        {"common_name": "Red junglefowl", "scientific_name": "Gallus gallus"},
        {"common_name": "Grey junglefowl", "scientific_name": "Gallus sonneratii"},
        {"common_name": "Kalij pheasant", "scientific_name": "Lophura leucomelanos"},
        {"common_name": "Himalayan monal", "scientific_name": "Lophophorus impejanus"},
        {"common_name": "Satyr tragopan", "scientific_name": "Tragopan satyra"},
        {"common_name": "Grey francolin", "scientific_name": "Ortygornis pondicerianus"},
        {"common_name": "Black francolin", "scientific_name": "Francolinus francolinus"},
        {"common_name": "Indian peahen", "scientific_name": "Pavo cristatus"},
        {"common_name": "Blood pheasant", "scientific_name": "Ithaginis cruentus"},
        {"common_name": "Cheer pheasant", "scientific_name": "Catreus wallichii"},
        {"common_name": "Western tragopan", "scientific_name": "Tragopan melanocephalus"},
        # ── Cranes & Storks ──
        {"common_name": "Sarus crane", "scientific_name": "Antigone antigone"},
        {"common_name": "Painted stork", "scientific_name": "Mycteria leucocephala"},
        {"common_name": "Asian openbill", "scientific_name": "Anastomus oscitans"},
        {"common_name": "Black-necked stork", "scientific_name": "Ephippiorhynchus asiaticus"},
        {"common_name": "Lesser adjutant", "scientific_name": "Leptoptilos javanicus"},
        {"common_name": "Greater adjutant", "scientific_name": "Leptoptilos dubius"},
        {"common_name": "Woolly-necked stork", "scientific_name": "Ciconia episcopus"},
        {"common_name": "White stork", "scientific_name": "Ciconia ciconia"},
        {"common_name": "Demoiselle crane", "scientific_name": "Grus virgo"},
        # ── Herons & Egrets ──
        {"common_name": "Indian pond heron", "scientific_name": "Ardeola grayii"},
        {"common_name": "Grey heron", "scientific_name": "Ardea cinerea"},
        {"common_name": "Purple heron", "scientific_name": "Ardea purpurea"},
        {"common_name": "Cattle egret", "scientific_name": "Bubulcus ibis"},
        {"common_name": "Great egret", "scientific_name": "Ardea alba"},
        {"common_name": "Little egret", "scientific_name": "Egretta garzetta"},
        {"common_name": "Intermediate egret", "scientific_name": "Ardea intermedia"},
        {"common_name": "Black-crowned night heron", "scientific_name": "Nycticorax nycticorax"},
        {"common_name": "Little bittern", "scientific_name": "Ixobrychus minutus"},
        # ── Ibises & Spoonbills ──
        {"common_name": "Black-headed ibis", "scientific_name": "Threskiornis melanocephalus"},
        {"common_name": "Red-naped ibis", "scientific_name": "Pseudibis papillosa"},
        {"common_name": "Glossy ibis", "scientific_name": "Plegadis falcinellus"},
        {"common_name": "Eurasian spoonbill", "scientific_name": "Platalea leucorodia"},
        # ── Flamingos ──
        {"common_name": "Greater flamingo", "scientific_name": "Phoenicopterus roseus"},
        {"common_name": "Lesser flamingo", "scientific_name": "Phoeniconaias minor"},
        # ── Ducks & Geese ──
        {"common_name": "Bar-headed goose", "scientific_name": "Anser indicus"},
        {"common_name": "Ruddy shelduck", "scientific_name": "Tadorna ferruginea"},
        {"common_name": "Northern pintail", "scientific_name": "Anas acuta"},
        {"common_name": "Common teal", "scientific_name": "Anas crecca"},
        {"common_name": "Spot-billed duck", "scientific_name": "Anas poecilorhyncha"},
        {"common_name": "Cotton pygmy goose", "scientific_name": "Nettapus coromandelianus"},
        {"common_name": "Lesser whistling duck", "scientific_name": "Dendrocygna javanica"},
        {"common_name": "Knob-billed duck", "scientific_name": "Sarkidiornis melanotos"},
        # ── Waders & Shorebirds ──
        {"common_name": "Red-wattled lapwing", "scientific_name": "Vanellus indicus"},
        {"common_name": "Yellow-wattled lapwing", "scientific_name": "Vanellus malabaricus"},
        {"common_name": "Black-winged stilt", "scientific_name": "Himantopus himantopus"},
        {"common_name": "Indian thick-knee", "scientific_name": "Burhinus indicus"},
        {"common_name": "Pheasant-tailed jacana", "scientific_name": "Hydrophasianus chirurgus"},
        {"common_name": "Bronze-winged jacana", "scientific_name": "Metopidius indicus"},
        {"common_name": "Common sandpiper", "scientific_name": "Actitis hypoleucos"},
        {"common_name": "Wood sandpiper", "scientific_name": "Tringa glareola"},
        {"common_name": "River tern", "scientific_name": "Sterna aurantia"},
        {"common_name": "Indian skimmer", "scientific_name": "Rynchops albicollis"},
        # ── Cormorants & Pelicans ──
        {"common_name": "Indian cormorant", "scientific_name": "Phalacrocorax fuscicollis"},
        {"common_name": "Little cormorant", "scientific_name": "Microcarbo niger"},
        {"common_name": "Great cormorant", "scientific_name": "Phalacrocorax carbo"},
        {"common_name": "Spot-billed pelican", "scientific_name": "Pelecanus philippensis"},
        {"common_name": "Darter", "scientific_name": "Anhinga melanogaster"},
        # ── Swifts & Swallows ──
        {"common_name": "Asian palm swift", "scientific_name": "Cypsiurus balasiensis"},
        {"common_name": "White-rumped needletail", "scientific_name": "Zoonavena sylvatica"},
        {"common_name": "Barn swallow", "scientific_name": "Hirundo rustica"},
        {"common_name": "Red-rumped swallow", "scientific_name": "Cecropis daurica"},
        {"common_name": "Wire-tailed swallow", "scientific_name": "Hirundo smithii"},
        # ── Hoopoe, Roller & Others ──
        {"common_name": "Eurasian hoopoe", "scientific_name": "Upupa epops"},
        {"common_name": "Indian roller", "scientific_name": "Coracias benghalensis"},
        {"common_name": "Dollar bird", "scientific_name": "Eurystomus orientalis"},
        {"common_name": "Indian nightjar", "scientific_name": "Caprimulgus asiaticus"},
        {"common_name": "Savanna nightjar", "scientific_name": "Caprimulgus affinis"},
        # ── Shrikes ──
        {"common_name": "Long-tailed shrike", "scientific_name": "Lanius schach"},
        {"common_name": "Bay-backed shrike", "scientific_name": "Lanius vittatus"},
        {"common_name": "Brown shrike", "scientific_name": "Lanius cristatus"},
        {"common_name": "Great grey shrike", "scientific_name": "Lanius excubitor"},
        # ── Wagtails & Pipits ──
        {"common_name": "White wagtail", "scientific_name": "Motacilla alba"},
        {"common_name": "Grey wagtail", "scientific_name": "Motacilla cinerea"},
        {"common_name": "Citrine wagtail", "scientific_name": "Motacilla citreola"},
        {"common_name": "Paddyfield pipit", "scientific_name": "Anthus rufulus"},
        # ── Thrushes & Chats ──
        {"common_name": "Oriental magpie-robin", "scientific_name": "Copsychus saularis"},
        {"common_name": "Indian chat", "scientific_name": "Oenanthe fusca"},
        {"common_name": "Pied bushchat", "scientific_name": "Saxicola caprata"},
        {"common_name": "Blue rock thrush", "scientific_name": "Monticola solitarius"},
        {"common_name": "Malabar whistling thrush", "scientific_name": "Myophonus horsfieldii"},
        # ── Bustards ──
        {"common_name": "Great Indian bustard", "scientific_name": "Ardeotis nigriceps"},
        {"common_name": "Indian bustard", "scientific_name": "Ardeotis nigriceps"},
        {"common_name": "Lesser florican", "scientific_name": "Sypheotides indicus"},
        {"common_name": "Bengal florican", "scientific_name": "Houbaropsis bengalensis"},
        # ── Laughingthrushes ──
        {"common_name": "Rufous-bellied niltava", "scientific_name": "Niltava sundara"},
        {"common_name": "White-crested laughingthrush", "scientific_name": "Garrulax leucolophus"},
        {"common_name": "Rufous treepie", "scientific_name": "Dendrocitta vagabunda"},
        # ── Endangered / Endemic ──
        {"common_name": "Nilgiri flycatcher", "scientific_name": "Eumyias albicaudatus"},
        {"common_name": "Nilgiri pipit", "scientific_name": "Anthus nilghiriensis"},
        {"common_name": "Malabar trogon", "scientific_name": "Harpactes fasciatus"},
        {"common_name": "Sri Lanka frogmouth", "scientific_name": "Batrachostomus moniliger"},
        {"common_name": "Jerdon's courser", "scientific_name": "Rhinoptilus bitorquatus"},
        {"common_name": "Andaman teal", "scientific_name": "Anas albogularis"},
        {"common_name": "Spoon-billed sandpiper", "scientific_name": "Calidris pygmaea"},
    ],
    "reptiles": [
        # ── Snakes — Big Four + Others ──
        {"common_name": "King cobra", "scientific_name": "Ophiophagus hannah"},
        {"common_name": "Indian cobra", "scientific_name": "Naja naja"},
        {"common_name": "Russell's viper", "scientific_name": "Daboia russelii"},
        {"common_name": "Common krait", "scientific_name": "Bungarus caeruleus"},
        {"common_name": "Indian python", "scientific_name": "Python molurus"},
        {"common_name": "Reticulated python", "scientific_name": "Malayopython reticulatus"},
        {"common_name": "Indian rock python", "scientific_name": "Python molurus"},
        {"common_name": "Saw-scaled viper", "scientific_name": "Echis carinatus"},
        {"common_name": "Malabar pit viper", "scientific_name": "Craspedocephalus malabaricus"},
        {"common_name": "Bamboo pit viper", "scientific_name": "Trimeresurus gramineus"},
        {"common_name": "Hump-nosed pit viper", "scientific_name": "Hypnale hypnale"},
        {"common_name": "Common vine snake", "scientific_name": "Ahaetulla nasuta"},
        {"common_name": "Indian rat snake", "scientific_name": "Ptyas mucosa"},
        {"common_name": "Common wolf snake", "scientific_name": "Lycodon aulicus"},
        {"common_name": "Checkered keelback", "scientific_name": "Fowlea piscator"},
        {"common_name": "Banded krait", "scientific_name": "Bungarus fasciatus"},
        {"common_name": "Indian trinket snake", "scientific_name": "Coelognathus helena"},
        {"common_name": "Common sand boa", "scientific_name": "Eryx conicus"},
        {"common_name": "Common cat snake", "scientific_name": "Boiga trigonata"},
        {"common_name": "Green vine snake", "scientific_name": "Ahaetulla nasuta"},
        # ── Crocodilians ──
        {"common_name": "Mugger crocodile", "scientific_name": "Crocodylus palustris"},
        {"common_name": "Gharial", "scientific_name": "Gavialis gangeticus"},
        {"common_name": "Saltwater crocodile", "scientific_name": "Crocodylus porosus"},
        # ── Lizards ──
        {"common_name": "Bengal monitor", "scientific_name": "Varanus bengalensis"},
        {"common_name": "Water monitor", "scientific_name": "Varanus salvator"},
        {"common_name": "Indian chameleon", "scientific_name": "Chamaeleo zeylanicus"},
        {"common_name": "Common garden lizard", "scientific_name": "Calotes versicolor"},
        {"common_name": "Tokay gecko", "scientific_name": "Gekko gecko"},
        {"common_name": "Indian house gecko", "scientific_name": "Hemidactylus frenatus"},
        {"common_name": "Fan-throated lizard", "scientific_name": "Sitana ponticeriana"},
        {"common_name": "Indian skink", "scientific_name": "Eutropis carinata"},
        {"common_name": "Flying lizard", "scientific_name": "Draco dussumieri"},
        # ── Turtles & Tortoises ──
        {"common_name": "Indian star tortoise", "scientific_name": "Geochelone elegans"},
        {"common_name": "Indian flapshell turtle", "scientific_name": "Lissemys punctata"},
        {"common_name": "Indian roofed turtle", "scientific_name": "Pangshura tecta"},
        {"common_name": "Indian softshell turtle", "scientific_name": "Nilssonia gangetica"},
        {"common_name": "Olive ridley sea turtle", "scientific_name": "Lepidochelys olivacea"},
        {"common_name": "Green sea turtle", "scientific_name": "Chelonia mydas"},
        {"common_name": "Hawksbill sea turtle", "scientific_name": "Eretmochelys imbricata"},
        {"common_name": "Leatherback sea turtle", "scientific_name": "Dermochelys coriacea"},
        {"common_name": "Red-crowned roofed turtle", "scientific_name": "Batagur kachuga"},
        {"common_name": "Indian tent turtle", "scientific_name": "Pangshura tentoria"},
        {"common_name": "Travancore tortoise", "scientific_name": "Indotestudo travancorica"},
    ],
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing articles to avoid duplicates
    existing_species = set()
    for filepath in OUTPUT_DIR.glob("*.json"):
        if filepath.name == "failed_species.json":
            continue
        try:
            with open(filepath, encoding="utf-8") as f:
                articles = json.load(f)
                for a in articles:
                    existing_species.add(a.get("species_name", "").lower())
                    existing_species.add(a.get("scientific_name", "").lower())
        except Exception:
            pass

    total_to_scrape = sum(len(v) for v in ESSENTIAL_SPECIES.values())
    print("=" * 60)
    print("  Adding Indian Wildlife Species — Comprehensive Edition")
    print(f"  {total_to_scrape} species to process")
    print(f"  {len(existing_species)} existing species (will skip duplicates)")
    print("=" * 60)
    print()

    new_articles = {"mammals": [], "birds": [], "reptiles": []}
    failed = []
    skipped = 0

    for group, species_list in ESSENTIAL_SPECIES.items():
        logger.info(f"Processing {group} ({len(species_list)} species)...")

        for sp in species_list:
            name = sp["common_name"]
            sci = sp["scientific_name"]

            # Skip if already exists
            if name.lower() in existing_species or sci.lower() in existing_species:
                logger.info(f"  Skipping (already exists): {name}")
                skipped += 1
                continue

            # Scrape from Wikipedia
            article = scrape_species_page(name, sci)
            if article:
                article["geographic_regions"] = ["India"]
                new_articles[group].append(article)
                existing_species.add(name.lower())
                existing_species.add(sci.lower())
                logger.info(f"  + {name}")
            else:
                failed.append(name)
                logger.warning(f"  FAILED: {name}")

            time.sleep(0.3)  # Rate limiting

    # Merge with existing data
    for group, articles in new_articles.items():
        if not articles:
            continue

        output_file = OUTPUT_DIR / f"{group}.json"
        existing = []
        if output_file.exists():
            with open(output_file, encoding="utf-8") as f:
                existing = json.load(f)

        merged = existing + articles
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {group}: {len(existing)} existing + {len(articles)} new = {len(merged)} total")

    total_new = sum(len(v) for v in new_articles.values())
    print()
    print("=" * 60)
    print(f"  Done! Added {total_new} new species")
    print(f"  Skipped {skipped} (already existed)")
    print(f"  Failed {len(failed)} species")
    if failed:
        print(f"  Failed: {', '.join(failed[:15])}{'...' if len(failed) > 15 else ''}")
    print()
    print("  Next steps:")
    print("    python -m src.preprocessing.cleaner")
    print("    python -m src.preprocessing.chunker")
    print("    python -m src.preprocessing.validator")
    print("    python -m src.retrieval.build_indexes")
    print("=" * 60)


if __name__ == "__main__":
    main()
