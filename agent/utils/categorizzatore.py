"""
Categorizzazione interventi di manutenzione.

Classifica descrizioni di interventi in 15 macro-categorie
usando pattern regex.
"""

import re
from typing import List

# 15 macro-categorie con pattern di riconoscimento
# Ordine importante: le categorie più specifiche vanno prima di quelle generiche

CATEGORIE = [
    ("01. PNEUMATICI", (
        r"pneum|pneu|gomm[ae]|gome\b|goma\b|jinyu|pirelli|bridgestone|goodyear|sava\b|"
        r"goodrich|michelin|rappezzo|foratur|inversione.*ruot|rotazione.*ruot|"
        r"valvola pneum|ruota di scorta|ruota bucata|ruot[ae].*scorta|"
        r"mont.*ruota.*scorta|stacco.*ruota|cerchio|"
        r"385/65|315/80|295/80|295/60|385/55|315/70|315/60|"
        r"12[\.,]00|13[\.,]00|r22[\.,]?5|r19[\.,]?5|r17[\.,]?5|"
        r"montaggio.*pneum|smontaggio.*pneum|smont\..*mont.*pneum|"
        r"gomme|cambio ruota|it\s*90|itineris|trailer\s*90|"
        r"penumatic|pneuamatic|"
        r"sostituzione.*ruot|cam\s*gom|allineamento.*ruot|"
        r"riparaz\w*\s*penu|riparaz\w*\s*pneum|"
        r"cam\.\s*goma|"
        r"copertone|inversione.*cerch|cerch"
    )),
    ("02. IMPIANTO FRENANTE", (
        r"fren[io]|feni\b|pastigl|disco\s*fren|disci\b|dischi\s*\d.*ass|brake|diapress|torples|"
        r"cilindro\s*fren|cilindro.*membran|cilindro.*molla|registrazione\s*fren|cavo\s*freno|soffiett.*fren|"
        r"prova\s*fren|freno\s*a\s*mano|freno\s*a\s*tamburo|ganasc|prova.*rulli|"
        r"camera\s*fren|brake\s*chamber|zanche|manovella|manov\.|reggiman|"
        r"soffione|pinza\s*fren|regolat.*fren|caliper|"
        r"giunto.*iso|attac[hc][io]?\s*aria|spirale.*aria|gomini.*aria|tubo\s*spirale|"
        r"corr\.?\s*fren|modulat.*ebs|"
        r"sost.*dischi\s*\d|sost\s*feni|somtaggio\s*disci|"
        r"load.*sens.*relay|"
        r"rep\s*tubi\s*aria|tubi\s*aria|"
        r"rubinetto.*sfrenat|sfrenatur|"
        r"rettifica.*disc|"
        r"trailer.*control.*valve|"
        r"soffioen|"
        r"presa.*ebs|"
        r"cilindro.*tristop|tristop|"
        r"cartuccia.*coalescenz|"
        r"valvol[ae].*scar.*cond|"
        r"valvol[oa].*control.*pression|"
        r"champio|ecoair"
    )),
    ("03. SOSPENSIONI E AMMORTIZZATORI", (
        r"moll[ae]\s*aria|air\s*sp[ir]|ammort|amort|ammo\b|silent.?block|balestra|bal\.\s*comp|"
        r"sospensi|suspens|susp\b|sosp\b|boccol[ea]|piastre usura|rondell.*sospen|"
        r"cv\s*shock|assale|asale|barra\s*stabiliz|stabiliz[z]?at|tampone|tamponi|"
        r"moll[ae].*pneumat|moll[ae].*asse|spring\s*seat|"
        r"cam\s+susp|sost\s*susp|mont\s*susp|sost\s*sosp|sost\s*sospe|"
        r"cam\s+\d*\s*amort|cam\s*amort|sost\s*ammo|"
        r"serraggio\s*balest|"
        r"raise.*lower.*valve|levelling.*valve|"
        r"elemento\s*elastic|ammorizzat|"
        r"5\s*ammorizzat"
    )),
    ("04. CARROZZERIA CONTAINER / CASSE MOBILI", (
        r"paviment|pianal[ei]?|pioanal|multistrato|betulla|tavol[eao]|"
        r"stecc[ha]|stecch|stequ|steche|"
        r"ruggin|ruigg|rugine|"
        r"port[ea].*(?:blocc|intern|dx|sx|post)|sblocc?.*port|chiusur.*port|"
        r"porte\s*blocc|sblocc?.*asta|serrattur|serratur|"
        r"chiusur[ae]?\s*cass|"
        r"travers|trav\.|ravers[ae]|"
        r"cernier[ae]|"
        r"ganc[io]|gancett|gangio|"
        r"pern[io](?!.*ruota)|"
        r"botol[ea]|"
        r"montant[ie]|longher|"
        r"j.?bar|"
        r"manigli[ae]|"
        r"corrimano|"
        r"ventol[ae]|"
        r"scalett[ae]|scala\s*sal|"
        r"cancell[oi]|"
        r"colp[io].*lat|colp[io].*[ds]x|colp[io].*lato|\bcolp[io]\b|"
        r"raddrizzat|raddriz|raddr\.|radd[r]?\.|rad\.\s*asta|"
        r"buc[ho].*tetto|buc[ho].*paret|buc[ho].*[ds]x|tetto.*buc[ho]|"
        r"pannell[io](?!.*catarifrang|.*reflex)|kit\s*pannello|"
        r"verniciatur|vernicatur|trattamento ruggine|antiruggine|"
        r"guarnizion|gurnizion|"
        r"cass[ea].*acciaio|cass[ea].*ast|cass[ea]\s*mobil|container|\bcont\.\b|"
        r"sald\w*.*montant|sald\w*.*cernier|sald\w*.*pern|"
        r"sald\w*.*stecca|sald\w*.*tetto|sald\w*.*ganc|"
        r"angolar[ie]|lamier[ae].*cass|"
        r"porte.*post|sede.*manigl|"
        r"arpion[ei]|guid[ae].*port|asta.*port|aletta|basculant|"
        r"\bcsc\b|"
        r"ferma.*port|"
        r"taglio.*paret|paret[ie].*[ds]x|"
        r"preparaz\w*\s*cass|riparaz\w*\s*cass|prep\s*cass|"
        r"preparaz\w*\s*tetto|preparaione\s*tetto|"
        r"riparaz\w*\s*port[ae]|"
        r"centina|centin[ae]|"
        r"tinteggiatur|"
        r"\bbott[ae]\b|\bbotte\b|"
        r"plastica\s*intern|"
        r"perdita.*dentro|"
        r"rinforzam|"
        r"collar[ie]|portamanichett|portatub|porta\s*tub|caseta\s*tub|"
        r"bucat[oi].*later|later.*bucat|"
        r"cassone|"
        r"preparazione.*chiusur|chiusura\b"
    )),
    ("05. TELONI E COPERTURE", (
        r"telon[ei]|pezz[ea].*telon|cucitur|"
        r"cop\.\s*cass|copertur[ae].*cass|coperchio.*cass|coperta.*cass|"
        r"cop\.\s*casa|cop\s+cass|cop\s+casa|copeta\s*cas|coperta\s*cas|"
        r"cavo tir|cavett.*(?:nylon|tir)|"
        r"barra avvolgimento|"
        r"archett[io]|"
        r"tiparaz|rip\w*.*telon|"
        r"\bcricchett|cinghia.*cricchett|"
        r"sbloccaggio.*copertur|copertura\b|sacco\b|"
        r"teli\s*copri|telo\b|topp[ea].*tel|rip\w*.*telo|"
        r"riparaz\w*\s*telo"
    )),
    ("06. IMPIANTO ELETTRICO E LUCI", (
        r"\bluc[ie]\b|lucii|fanal[ei]|fanalin|lampad|cablaggio|"
        r"pres[ea].*elettr|pres[ea].*poli|pres[ea].*2p|presa\s*rimorchio|"
        r"24v|"
        r"spirale\s*elettr|spirale\s*plast|spirale\s*eletr|"
        r"interruttore|centralina.*frecc|"
        r"massa\s*a\s*terra|messa\s*a\s*terra|catarifrang|rifletten|"
        r"imp\.?\s*luci|eutopoint|ecopoint|"
        r"led\s|led\b|bianco.*led|arancio.*led|"
        r"pannell[io].*catarifrang|paneli?\s*reflex|panel\s*reflex|"
        r"imp\w*\s*elettr|elettric|eletric|electri|elec\b|"
        r"strisc[ie].*ross|segnalet|"
        r"rev\w*\s*elettr|rev\s*sist\s*electr|rep\s*sist.*elet|"
        r"gemma|portatarga|targa\s*ripetit|"
        r"videocamer|alimentator[ei]|"
        r"rele\b|rel[eè]|fusibil|"
        r"selettore|prossimit|"
        r"cavo.*scarico.*terra|cavo\s*messa|"
        r"spazzol[ea].*tergicristall|tergicristall|"
        r"collegam\w*.*cavi|montaggio.*alimentat|"
        r"filo\s*rimorchi|attacco\s*corrent|"
        r"spina\b|tabbelle|"
        r"montaggio\s*panelli.*luc|"
        r"spirale\s*ebs|"
        r"europoint|gemm[ae]|"
        r"strisci[ae]|"
        r"power\s*relay"
    )),
    ("07. MOTORE E MECCANICA MOTRICE", (
        r"tagliand|filtr[io].*olio|filtr[io].*carburant|filtr[io].*aria(?!.*silos)|"
        r"filtr[io].*antipolline|filtri\b|sost\s*filtr|\bfiltro\b|"
        r"cambio olio|perdita olio|rabbocco\s*olio|olio\s*atf|"
        r"guarnizion.*copp|guarnizion.*test|"
        r"sensor[ei]|kitas|"
        r"rigenerazion|catalizzat|"
        r"bulb[io].*olio|"
        r"additivo|"
        r"blocchetto.*accension|"
        r"servomaster|frizion[ei]|"
        r"travaso|travasi|"
        r"depotenziam|"
        r"controllo.*a/?c|condizionat|"
        r"differenzial|"
        r"motore|turbina|cinghia(?!.*cricchett)|cinghie\s*dentat|radiator|ventola\s*mot|"
        r"diagnos[it]|centralin[ae](?!.*frecc)|"
        r"avviamento|alternator|compressor[ei](?!.*silos)|"
        r"marmitta|scappamento|tubo.*gas|"
        r"coppa olio|asta olio|liquido refrig|antigelo|"
        r"pompa gasolio|pompa acqua|pompa\s*idraul|iniett|"
        r"shunt.*valve|valvola.*marc|"
        r"puleggi[ae]|cint[ea]\b|distributore(?!.*silos)|"
        r"olio.*riduttor|riduttore|"
        r"pistola\s*aria|pistola\s*soffiag|soffiaggio\s*cabin|"
        r"silenziator[ei]|raccord[io].*sprint|"
        r"presa.*di.*forza|albero.*presa|canna.*olio|tubo.*olio|tubazion.*olio|"
        r"tubo.*aspiraz|tubo.*scaric(?!.*silos)|tubi\s*scaric|"
        r"aria\s*scania|"
        r"scarico\s*memoria|memoria\s*di\s*massa|"
        r"sost\s*olio\s*comp|sost\s*olio\b|sost\s*tubi\b|"
        r"testine|flang[ia]|"
        r"paraolio|cam\s*paraolio|"
        r"controllo\s*cambio|cambio\s*attacco\s*olio|olio\s*cambio|"
        r"attacco\s*aria\s*abs|"
        r"barra\s*sterzo|sterzo|"
        r"specchio.*retrovis|retrovis|"
        r"pompa\s*alza|alza\s*cabin|"
        r"cooling\s*sis|"
        r"blocco\s*trazion|"
        r"convertitor|"
        r"motore?\s*idraul|danfoss|"
        r"cinghie\s*dentel|"
        r"organi\s*di\s*comando|"
        r"6\s*pk|"
        r"repair\s*kit|"
        r"manutenzioni\s*vari"
    )),
    ("08. MOZZI E RUOTE", (
        r"mozzo|mozzi|muzzo|colonn[ea]tt[ea]|colonn?in[ea]|dad[io].*ruot|dadi\b|"
        r"perno ruota|disco limitat|"
        r"cuscinett|cucinett?i|registrazione\s*mozz|registro\s*cuci|"
        r"co\.\s*c.*ruota|coppia.*zanche|"
        r"hub\s*assembl|"
        r"cam\s*colonin|sost\s*\d*\s*colon[en]|"
        r"rotocamera|cam\s*rotocam"
    )),
    ("09. REVISIONE E CONTROLLI PERIODICI", (
        r"revision[ei]|revizione|movimentazione.*revision|"
        r"check.*list|preparaz.*revision|"
        r"calibrazione|tachigrafo|dtco|cronotachigraf|"
        r"prova fumi|"
        r"tabell[ea].*adesiv|adesiv[io].*normat|"
        r"prova.*usl|"
        r"collaudo|"
        r"dichiarazione.*conform|libretto.*manutenzione|"
        r"linea\s*vita|conformit|"
        r"prerevis|pre.?revis|pre.?revizion|"
        r"rev\s*pressio|rev\s*telaio|"
        r"controlli\s*pre.?revis|"
        r"verifica\s*periodic|recipiente\s*a\s*pression|"
        r"controllo\s*funzional"
    )),
    ("10. ATTREZZATURE SILOS / CISTERNA", (
        r"silos|cistern|"
        r"tubo.*scarico|tubo.*mandata|"
        r"filtro.*aria.*silos|filtro.*aria.*compressor|"
        r"valvola.*farfall|valvola.*europa|valvola.*stell|valvola.*sicurezz|"
        r"valvola.*livettat|valvola\s*universal|"
        r"storz|portagomma|"
        r"manometr|manomet\b|glicerin|"
        r"termometr|"
        r"coperchio.*doppio|delrin|coperchio\b|riraraz.*coperchi|"
        r"pompa.*ingranag|"
        r"nippli|nipplo|curv[ea].*inox|"
        r"maniglia.*valvola|valvola.*sirca|"
        r"volantino.*zincat|volantio|"
        r"convogliat|"
        r"soffiante|"
        r"DIN.?150|"
        r"rubinett|robineto|piatto.*scaric|"
        r"raccord[io](?!.*sprint)|"
        r"bonded|"
        r"ralla|"
        r"fascett[ae]|"
        r"ghier[ae]|bussola.*arrest|cannocchial|"
        r"calott[ea]|tiranti.*calott|"
        r"sollevator[ei]|soll\.\s*[ds]x|"
        r"portatub|porta\s*tub|"
        r"pistone.*ribalt|"
        r"innesto.*rapid|giunzione|giunto\b|"
        r"adattator[ei]|"
        r"o.ring|"
        r"sup\.\s*aria|"
        r"guarniz\s*semicircol|"
        r"piedini|"
        r"escort\b|"
        r"scarico\s*aria|"
        r"tramoggi[ae]|"
        r"valvola\s*a\s*farbal|farballa|"
        r"riduzione\s*zin[cg]at|"
        r"tappo\b|"
        r"gommino|"
        r"tubo\s*termoplast"
    )),
    ("11. ROTOCELLA E TWIST LOCK", (
        r"rotocell|rotocela|twist|twister|"
        r"pulsantier[ae]|"
        r"cavo.*pulsant|"
        r"blocchetto.*guida|"
        r"jost|menci"
    )),
    ("12. SOCCORSO E INTERVENTI FUORI SEDE", (
        r"officina.*mobile|"
        r"soccorso|"
        r"fuori.*sede|"
        r"trasfert|"
        r"presso.*client|c/o\s|novamont|fiorenzuola|"
        r"smontaggio.*zampe|"
        r"sul.*posto|in.*francia|hamecher|"
        r"tiro\s*gru|gru\s*ferar|"
        r"baselle|"
        r"messa\s*in\s*moto|"
        r"presso\s*vs\s*sede|"
        r"trasporto.*rottura"
    )),
    ("13. MATERIALI DI CONSUMO E FLUIDI", (
        r"adblue|ad\s*blue|repostare\s*ad|scarico\s*ad|olio\s*atf|"
        r"grass[oa]|graso|lubrificaz|ingrassag|"
        r"olio idraul|"
        r"olio.*compressor.*mobil|rarus|"
        r"olio.*motor.*shell|rimula|olio motore|"
        r"material[ei].*consumo|mat\.?\s*consumo|"
        r"batter[iy]|batteria|"
        r"manodopera|"
        r"viti.*rondell|"
        r"pulizia"
    )),
    ("14. STRUTTURA METALLICA E SALDATURE", (
        r"sagomati.*lamier|taglio.*laser|pressopieg|"
        r"tubolar[ie].*zincat|tubolar[ei]|"
        r"lamier[ae]|"
        r"anell[io].*acciaio|anell[io].*inox|"
        r"saldatur[ae]|saldat[oi]|soldatur|"
        r"sald\.(?!.*montant|.*cernier|.*pern|.*stecc|.*tetto|.*ganc)|"
        r"paragfanc|parafang|paraganghi|paraurti|paraschi|paracicl|"
        r"staff[ea]|staffa|"
        r"piattaform|pedan|"
        r"struttur.*metall|"
        r"fazzoletto|fazzoletti|"
        r"supporto|serbatoio|cassett[aio]|"
        r"capannin[ae]|anello|"
        r"barra.*paraincastr|barra\s*parainc|"
        r"frame.*mount.*bracket|"
        r"cassetta.*zinc|"
        r"cassetta.*attrez|"
        r"piastr[ae].*lamier|piastra\s*sagomat|piastra\s*rinforz|"
        r"cavallott|"
        r"piedi\b|sostituz.*piedi|"
        r"saldametall|"
        r"lastra\b|telarubbertex|"
        r"prolungh[ae]"
    )),
    ("15. ALLESTIMENTO E PERSONALIZZAZIONE", (
        r"kit.*satellit|installaz.*satellit|satellit|"
        r"scritt[ea].*cabin|"
        r"porta.*doc|"
        r"montaggio.*liner|liner\b|lainer\b|"
        r"adesiv[io].*letter|letter[ea].*adesiv|"
        r"messa.*su.*strada|"
        r"allestiment|"
        r"tasca.*porta|"
        r"cavo.*nero|"
        r"numero\b|sost.*numero|"
        r"semiaccoppi|"
        r"impianto.*as24|as24\b|"
        r"remix|oso\s*\d|"
        r"truck.*don\b|truck.*sahara|tubo.*aria.*truck|"
        r"montaggio\s*zampa|"
        r"cassetta\s*zinc|"
        r"gps\b|"
        r"scrite\b|"
        r"letter[ea]\s*mancan|numeri\b|numerazione|"
        r"targhett[ae]|"
        r"toll\s*collect"
    )),
]

# Pattern compilati per performance
CATEGORIE_COMPILED = [
    (nome, re.compile(pattern, re.IGNORECASE))
    for nome, pattern in CATEGORIE
]


def pulisci_testo(testo: str) -> str:
    """Normalizza il testo rimuovendo rumore."""
    if not testo:
        return ""
    testo = str(testo).strip()
    if testo.lower() == "none":
        return ""
    # Rimuovi caratteri di controllo Excel
    testo = testo.replace("_x000D_", "").replace("\r", "")
    # Rimuovi prezzi (€ 190,00)
    testo = re.sub(r"€\s*[\d.,]+", "", testo)
    # Rimuovi date isolate
    testo = re.sub(r"\bdel\s+\d{1,2}/\d{1,2}(?:/\d{2,4})?\b", "", testo, flags=re.IGNORECASE)
    testo = re.sub(r"\be\s+\d{1,2}/\d{1,2}(?:/\d{2,4})?\b", "", testo, flags=re.IGNORECASE)
    # Rimuovi numeri container
    testo = re.sub(r"\bcont\.?\s*\d+", "", testo, flags=re.IGNORECASE)
    # Rimuovi "NR" / "N." prefix con numero
    testo = re.sub(r"\bn\.?\s*r?\.?\s*(?=\d)", "", testo, flags=re.IGNORECASE)
    # Rimuovi matricole isolate
    testo = re.sub(r"\bmatricol[ae]?:?\s*[\w\d/\-]+", "", testo, flags=re.IGNORECASE)
    # Rimuovi codici fornitori
    testo = re.sub(r"\bcod\.?\s*[\w\d.]+", "", testo, flags=re.IGNORECASE)
    return testo.strip()


def is_solo_codice(testo: str) -> bool:
    """Controlla se il testo è solo un codice numerico/seriale senza contenuto."""
    t = testo.strip()
    if not t:
        return True
    # Solo numeri e spazi
    if re.match(r"^[\d\s/\-\.]+$", t):
        return True
    # Codice tipo AZJ21136 R
    if re.match(r"^[A-Z]{2,4}\s*\d{4,}.*$", t, re.IGNORECASE) and len(t) < 25:
        return True
    return False


def splitta_attivita(testo: str) -> List[str]:
    """Divide il testo in attività elementari."""
    if not testo:
        return []
    parti = re.split(r'\+|\n', testo)
    risultato = []
    for parte in parti:
        parte = parte.strip()
        if parte and len(parte) > 2:
            risultato.append(parte)
    return risultato if risultato else [testo]


def classifica(testo: str) -> List[str]:
    """Classifica un testo in una o più categorie. Ritorna lista di categorie."""
    if not testo or len(testo.strip()) < 3:
        return ["NON CLASSIFICATO"]

    if is_solo_codice(testo):
        return ["NON CLASSIFICATO"]

    categorie_trovate = []
    for nome, pattern in CATEGORIE_COMPILED:
        if pattern.search(testo):
            categorie_trovate.append(nome)

    return categorie_trovate if categorie_trovate else ["NON CLASSIFICATO"]


def categorizza_riga(descrizione: str, dettaglio: str) -> List[str]:
    """
    Classifica una singola riga (descrizione + dettaglio) nelle macro-categorie.

    Args:
        descrizione: testo del campo Descrizione
        dettaglio: testo del campo Dettaglio

    Returns:
        Lista ordinata di categorie (es. ["01. PNEUMATICI"]),
        oppure ["NON CLASSIFICATO"] se nessuna categoria trovata.
    """
    descrizione = pulisci_testo(descrizione)
    dettaglio = pulisci_testo(dettaglio)

    testo_completo = f"{descrizione} {dettaglio}".strip() if dettaglio else descrizione

    # 1) Classifica sul testo intero combinato
    categorie_riga = set()
    for c in classifica(testo_completo):
        categorie_riga.add(c)

    # 2) Splitta e classifica ogni frammento
    for att in splitta_attivita(testo_completo):
        for c in classifica(att):
            categorie_riga.add(c)

    # Rimuovi NON CLASSIFICATO se almeno una categoria reale trovata
    if len(categorie_riga) > 1:
        categorie_riga.discard("NON CLASSIFICATO")

    return sorted(categorie_riga)


# Mapping delle categorie AdHoc alle categorie del modello TipoGuasto
CATEGORIA_TO_TIPO_GUASTO = {
    "01. PNEUMATICI": "pneumatici",
    "02. IMPIANTO FRENANTE": "freni",
    "03. SOSPENSIONI E AMMORTIZZATORI": "sospensioni",
    "04. CARROZZERIA CONTAINER / CASSE MOBILI": "carrozzeria",
    "05. TELONI E COPERTURE": "carrozzeria",
    "06. IMPIANTO ELETTRICO E LUCI": "elettrico",
    "07. MOTORE E MECCANICA MOTRICE": "motore",
    "08. MOZZI E RUOTE": "pneumatici",
    "09. REVISIONE E CONTROLLI PERIODICI": "revisione",
    "10. ATTREZZATURE SILOS / CISTERNA": "idraulico",
    "11. ROTOCELLA E TWIST LOCK": "altro",
    "12. SOCCORSO E INTERVENTI FUORI SEDE": "altro",
    "13. MATERIALI DI CONSUMO E FLUIDI": "tagliando",
    "14. STRUTTURA METALLICA E SALDATURE": "carrozzeria",
    "15. ALLESTIMENTO E PERSONALIZZAZIONE": "altro",
    "NON CLASSIFICATO": "altro",
}


def categoria_to_tipo_guasto(categoria: str) -> str:
    """
    Converte una categoria AdHoc (es. "01. PNEUMATICI") in un TipoGuasto.

    Args:
        categoria: Nome categoria AdHoc

    Returns:
        Stringa compatibile con TipoGuasto
    """
    return CATEGORIA_TO_TIPO_GUASTO.get(categoria, "altro")
