# chunking.py
"""
Chunking utilitaire pour le RAG.

- Découpe en paragraphes
- Re-colle en blocs d'environ `chunk_size` caractères
- Ajoute un overlap (chevauchement) pour garder le contexte
- Smart chunking pour sections EASA avec contexte préservé
- Chunking adaptatif basé sur la densité du contenu
- Chunks hiérarchiques parent-enfant
- Augmentation des chunks avec mots-clés
"""

import re
import math
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# =====================================================================
#  CONSTANTES POUR CHUNKING ADAPTATIF
# =====================================================================

# Tailles de chunks selon la densité du contenu
CHUNK_SIZES = {
    "very_dense": 800,      # Code, formules, tableaux
    "dense": 1200,          # Technique, spécifications
    "normal": 1500,         # Texte standard
    "sparse": 2000,         # Narratif, introductions
}

# Mots-clés techniques pour détecter la densité (~750 termes aéronautiques)
TECHNICAL_INDICATORS = {
    # =========================================================================
    # TERMES NORMATIFS ET RÉGLEMENTAIRES
    # =========================================================================
    # Termes normatifs EASA/FAA
    "shall", "must", "should", "may", "require", "compliance", "applicable",
    "mandatory", "compliant", "specification", "requirement", "acceptable",
    "approved", "authorization", "certificate", "certified", "approval",
    "exemption", "deviation", "waiver", "amendment", "revision", "supersede",
    "effective", "obsolete", "canceled", "suspended", "revoked",
    # Codes de référence réglementaires
    "cs", "amc", "gm", "far", "jar", "easa", "icao", "faa", "tcds", "ata",
    "etso", "tso", "jtso", "caa", "anac", "dgac", "caac", "jcab", "casa",
    "tcca", "gcaa", "sacaa", "caav", "aesa", "enac", "lba", "bazl",
    # Chapitres ATA
    "ata21", "ata22", "ata23", "ata24", "ata25", "ata26", "ata27", "ata28",
    "ata29", "ata30", "ata31", "ata32", "ata33", "ata34", "ata35", "ata36",
    "ata38", "ata45", "ata49", "ata51", "ata52", "ata53", "ata54", "ata55",
    "ata56", "ata57", "ata70", "ata71", "ata72", "ata73", "ata74", "ata75",
    "ata76", "ata77", "ata78", "ata79", "ata80", "ata91", "ata92",
    # Documents réglementaires
    "osd", "mmel", "cdl", "poh", "afm", "mel", "ddg", "fcom", "qrh",
    "srm", "amm", "ipc", "cmm", "esm", "wdm", "ssm", "fim", "tsm",
    "mpd", "mrbr", "cpcp", "srmm", "ndtm", "fctm", "frmm", "swpm",
    "aipc", "trem", "eipc", "cir", "lir", "etm", "cfm", "mm",
    # Directives et bulletins
    "ad", "sb", "stc", "pma", "doa", "poa", "moa", "camo", "asl",
    "sil", "ain", "aoc", "awn", "cn", "tcn", "ssb", "aob", "oeb",
    # Licences et qualifications
    "atpl", "cpl", "ppl", "mpl", "ir", "me", "se", "type", "rating",
    "endorsement", "recurrent", "proficiency", "currency", "medical",
    "class1", "class2", "class3", "lapl", "bpl", "spl",

    # =========================================================================
    # STRUCTURES AÉRONEF
    # =========================================================================
    # Structure principale
    "fuselage", "wing", "empennage", "nacelle", "pylon", "fairing",
    "radome", "tailcone", "nosecone", "belly", "keel", "crown",
    "horizontal", "vertical", "stabilizer", "fin", "tailplane",
    "centerbox", "wingbox", "torsionbox", "leadingedge", "trailingedge",
    # Éléments structuraux détaillés
    "longeron", "spar", "rib", "stringer", "frame", "bulkhead", "former",
    "skin", "panel", "doubler", "reinforcement", "fitting", "bracket",
    "shear", "web", "flange", "cap", "beam", "truss", "intercostal",
    "splice", "joint", "seam", "lap", "butt", "scarf", "joggle",
    "cutout", "access", "inspection", "door", "panel", "closeout",
    "fillet", "radius", "joggle", "beading", "dimple", "countersink",
    # Surfaces de contrôle
    "aileron", "elevator", "rudder", "flap", "slat", "spoiler", "airbrake",
    "speedbrake", "tab", "trimtab", "servo", "horn", "hinge", "actuator",
    "flaperon", "elevon", "ruddervator", "canard", "winglet", "sharklet",
    "krueger", "fowler", "plain", "split", "slotted", "droop",
    "inboard", "outboard", "midspan", "root", "tip", "fence", "vortex",
    # Train d'atterrissage détaillé
    "landing", "gear", "undercarriage", "strut", "oleo", "shock",
    "wheel", "tire", "tyre", "brake", "antiskid", "retract", "extend",
    "uplock", "downlock", "door", "fairing", "bogie", "truck",
    "shimmy", "damper", "steering", "towing", "jacking", "axle",
    "torque", "link", "scissors", "drag", "brace", "side", "trunnion",
    "pintle", "swivel", "castor", "caster", "fuse", "plug", "seal",
    "hubcap", "bearing", "spacer", "inflation", "deflation", "creep",
    # Portes et ouvertures
    "door", "hatch", "window", "windshield", "windscreen", "canopy",
    "emergency", "exit", "escape", "slide", "raft", "evacuation",
    "cargo", "freight", "bulk", "forward", "aft", "overwing",
    "passenger", "service", "galley", "lavatory", "crew", "cockpit",
    "plug", "outward", "inward", "gull", "airstair", "ventral",

    # =========================================================================
    # PROPULSION ET APU
    # =========================================================================
    # Moteurs - Général
    "engine", "powerplant", "motor", "propeller", "prop", "blade",
    "thrust", "power", "torque", "rpm", "n1", "n2", "n3", "egt", "itt",
    "tgt", "tit", "np", "ng", "nf", "nh", "nl", "epr", "ff",
    # Turboréacteur/Turbofan détaillé
    "turbine", "compressor", "combustor", "combustion", "chamber", "burner",
    "fan", "bypass", "core", "spool", "shaft", "bearing", "seal",
    "nozzle", "exhaust", "afterburner", "reheat", "reverser", "blocker",
    "cascade", "cowl", "nacelle", "inlet", "intake", "diffuser",
    "stator", "rotor", "vane", "disk", "disc", "blisk", "bling",
    "lpt", "hpt", "lpc", "hpc", "ipc", "booster", "splitter",
    "shroud", "casing", "containment", "liner", "dome", "swirler",
    "igniter", "exciter", "plug", "harness", "lead", "cable",
    "vsv", "vbv", "igv", "ogv", "strut", "fairing", "plug", "centerbody",
    # Turbopropulseur détaillé
    "turboprop", "turboshaft", "propfan", "gearbox", "reduction", "rgb",
    "torquemeter", "feather", "feathering", "beta", "reverse", "groundfine",
    "governor", "overspeed", "underspeed", "pitch", "blade", "angle",
    "counterweight", "spinner", "dome", "hub", "retention", "deice",
    # APU (Auxiliary Power Unit) détaillé
    "apu", "auxiliary", "starter", "generator", "bleed", "pneumatic",
    "apuegt", "apurpm", "loadcompressor", "inletdoor", "exhaustdoor",
    "firewall", "shroud", "mount", "vibration", "isolator", "duct",
    "surge", "stall", "flameout", "hotstart", "hungstart", "nostart",
    # Carburant détaillé
    "fuel", "tank", "pump", "valve", "filter", "line", "feed",
    "boost", "transfer", "jettison", "dump", "vent", "surge",
    "quantity", "densitometer", "probe", "capacitance", "stick",
    "collector", "sump", "baffle", "rib", "stringer", "access",
    "defuel", "refuel", "overwing", "underwing", "single", "pressure",
    "gravity", "jeta1", "jeta", "jetb", "jp8", "jp5", "jp4", "avgas",
    "contamination", "water", "particulate", "microbial", "fungal",
    # Huile et lubrification détaillé
    "oil", "lubricant", "lubrication", "scavenge", "breather",
    "chip", "detector", "magnetic", "filter", "cooler", "heat",
    "pressure", "temperature", "quantity", "consumption", "analysis",
    "spectrometric", "ferrographic", "viscosity", "tbn", "tan",
    # Défaillances moteur
    "surge", "stall", "compressor", "flameout", "rollback", "runaway",
    "overspeed", "overtemp", "overtorque", "vibration", "unbalance",
    "fod", "foreignobject", "damage", "ingestion", "birdstrike",
    "contained", "uncontained", "failure", "separation", "liberation",

    # =========================================================================
    # SYSTÈMES AVION
    # =========================================================================
    # Hydraulique détaillé
    "hydraulic", "hyd", "pump", "reservoir", "accumulator", "actuator",
    "servo", "cylinder", "piston", "manifold", "priority", "ptu",
    "rat", "ramair", "edp", "emdp", "acmp", "ehp", "ebha", "eha",
    "green", "blue", "yellow", "left", "right", "center", "standby",
    "pressure", "return", "case", "drain", "supply", "demand",
    "filter", "bypass", "thermal", "relief", "check", "restrictor",
    "skydrol", "ld4", "hyjet", "phosphate", "ester", "synthetic",
    # Pneumatique détaillé
    "pneumatic", "bleed", "precooler", "valve", "duct", "manifold",
    "pack", "mixer", "trim", "zone", "recirculation", "outflow",
    "crossbleed", "isolation", "starter", "wing", "antiice", "wai",
    "pressure", "regulator", "sensor", "switch", "overpressure",
    "overtemperature", "leak", "detection", "indication",
    # Électrique détaillé
    "electrical", "electric", "power", "generator", "alternator",
    "battery", "bus", "busbar", "breaker", "contactor", "relay",
    "transformer", "rectifier", "inverter", "converter", "apugen",
    "idg", "vscf", "vfg", "rag", "emergency", "essential", "shed",
    "apu", "gpu", "ground", "external", "ac", "dc", "hz", "volt",
    "amp", "watt", "kva", "var", "pf", "phase", "neutral", "ground",
    "nicad", "leadacid", "liion", "lithium", "thermal", "runaway",
    "tru", "atru", "btru", "sspc", "elcu", "gcr", "bcl", "bcu",
    # Éclairage
    "lighting", "light", "beacon", "strobe", "navigation", "position",
    "landing", "taxi", "runway", "turnoff", "logo", "wing", "ice",
    "cabin", "emergency", "exit", "floor", "path", "escape",
    "cockpit", "dome", "reading", "flood", "panel", "integral",
    "led", "halogen", "incandescent", "xenon", "hid",
    # Climatisation et pressurisation détaillé
    "airconditioning", "conditioning", "pressurization", "cabin",
    "altitude", "differential", "outflow", "safety", "dump",
    "pack", "mixer", "gasper", "recirculation", "hepa", "filter",
    "temp", "temperature", "zone", "cockpit", "cargo", "avionics",
    "controller", "selector", "sensor", "duct", "riser", "diffuser",
    "blower", "fan", "ejector", "venturi", "bootstrap", "cycle",
    # Protection givre détaillé
    "ice", "icing", "antiice", "deice", "probe", "pitot", "static",
    "aoa", "tat", "sat", "heating", "bleed", "pneumatic", "electric",
    "boots", "weeping", "wing", "engine", "windshield", "window",
    "inlet", "cowl", "spinner", "propeller", "tail", "stabilizer",
    "tks", "glycol", "fluid", "panel", "mat", "blanket", "spray",
    # Oxygène détaillé
    "oxygen", "mask", "regulator", "diluter", "crew", "passenger",
    "chemical", "generator", "gaseous", "psi", "duration", "flow",
    "portable", "therapeutic", "firstaid", "walkaround", "quickdon",
    "continuous", "demand", "pressure", "altitude", "deployment",
    # Détection incendie et extinction détaillé
    "fire", "detection", "warning", "extinguisher", "bottle", "agent",
    "halon", "loop", "overheat", "smoke", "cargo", "lavatory",
    "apu", "engine", "squib", "discharge", "zone", "bay", "nacelle",
    "wheelwell", "bleed", "duct", "pneumatic", "continuous", "fenwal",
    "kidde", "graviner", "meggitt", "dual", "single", "shot",
    # Eau et déchets
    "water", "potable", "waste", "lavatory", "galley", "drain",
    "tank", "heater", "pump", "valve", "filter", "quantity",
    "vacuum", "flush", "recirculating", "servicing", "panel",

    # =========================================================================
    # AVIONIQUE ET NAVIGATION
    # =========================================================================
    # Instruments de vol détaillé
    "pfd", "nd", "eicas", "ecam", "mfd", "hud", "efis", "isis",
    "attitude", "horizon", "altimeter", "airspeed", "vsi", "hsi",
    "aoa", "slip", "turn", "coordinator", "compass", "rmi", "adf",
    "fma", "fd", "flightdirector", "ils", "raw", "data", "tape",
    "dme", "arc", "rose", "plan", "map", "range", "weather",
    "terrain", "traffic", "waypoint", "constraint", "prediction",
    # Systèmes de données air détaillé
    "adiru", "adc", "ahrs", "iru", "irs", "adirs", "saaru", "isfd",
    "pitot", "static", "aoa", "tat", "sat", "mach", "cas", "tas",
    "altitude", "vertical", "speed", "baro", "radio", "decision",
    "minimums", "probe", "heating", "drain", "alternate", "standby",
    # Navigation détaillée
    "navigation", "nav", "gps", "gnss", "ins", "irs", "adirs", "fms",
    "vor", "dme", "ils", "loc", "glide", "glideslope", "marker",
    "ndb", "tacan", "rnav", "rnp", "lnav", "vnav", "approach",
    "waypoint", "fix", "radial", "bearing", "track", "heading",
    "route", "airway", "sid", "star", "iap", "missed", "hold",
    "sbas", "gbas", "waas", "egnos", "gagan", "msas", "raim", "fde",
    "lpv", "lnav", "vnav", "apv", "baro", "cat1", "cat2", "cat3",
    "autoland", "flare", "rollout", "goaround", "toga", "windshear",
    # Communication détaillée
    "communication", "comm", "radio", "vhf", "hf", "uhf", "satcom",
    "acars", "cpdlc", "ads", "transponder", "tcas", "mode",
    "squawk", "ident", "interphone", "pa", "selcal", "elt", "cvr",
    "datalink", "fans", "atc", "aoc", "airline", "operational",
    "audio", "boom", "mask", "handset", "speaker", "jack", "headset",
    "ptt", "transmit", "receive", "monitor", "guard", "emergency",
    # Pilote automatique et commandes de vol détaillé
    "autopilot", "autothrottle", "autothrust", "fcc", "fmgc", "fac",
    "elac", "sec", "fcpc", "fbw", "flybywire", "sidestick", "yoke",
    "column", "pedal", "trim", "stab", "stabilizer", "servo",
    "engage", "disengage", "disconnect", "takeover", "override",
    "cmd", "cws", "cfs", "alt", "vs", "hdg", "nav", "loc", "app",
    "speed", "mach", "flex", "climb", "descent", "cruise", "hold",
    # Radar et détection détaillée
    "radar", "weather", "wx", "terrain", "taws", "gpws", "egpws",
    "tcas", "pwc", "windshear", "predictive", "reactive", "warning",
    "caution", "alert", "resolution", "advisory", "traffic",
    "intruder", "threat", "target", "bearing", "range", "closure",
    "vertical", "separation", "climb", "descend", "crossing",
    # Enregistreurs détaillés
    "fdr", "cvr", "qar", "dfdr", "ssfdr", "recorder", "blackbox",
    "dau", "fdau", "dfdau", "aids", "bite", "cfds", "acms", "hums",
    "parameter", "exceedance", "trend", "monitoring", "download",

    # =========================================================================
    # AÉRODYNAMIQUE ET PERFORMANCES
    # =========================================================================
    # Forces aérodynamiques
    "lift", "drag", "thrust", "weight", "portance", "trainee",
    "induced", "parasite", "profile", "wave", "interference",
    "pressure", "friction", "form", "skin", "boundary", "layer",
    "laminar", "turbulent", "transition", "separation", "stall",
    "vortex", "wake", "downwash", "upwash", "wingtip", "trailing",
    # Coefficients et aérodynamique
    "coefficient", "cl", "cd", "cm", "cy", "cn", "alpha", "aoa",
    "incidence", "angle", "attack", "sideslip", "beta", "gamma",
    "aspect", "ratio", "taper", "sweep", "dihedral", "anhedral",
    "twist", "washout", "washin", "camber", "thickness", "chord",
    "span", "area", "loading", "ar", "mac", "mgc", "smc",
    # Caractéristiques de vol
    "stall", "buffet", "mmo", "vmo", "vne", "vno", "vfe", "vle",
    "vlo", "vs", "vs0", "vs1", "vref", "vapp", "v1", "v2", "vr",
    "vmcg", "vmca", "vmcl", "vyse", "vxse", "vx", "vy", "va", "vb",
    "vd", "vc", "vdf", "vmdf", "vfc", "mfc", "vra", "vmo",
    "green", "amber", "white", "red", "arc", "barber", "pole",
    # Performances détaillées
    "performance", "takeoff", "landing", "climb", "cruise", "descent",
    "range", "endurance", "payload", "mtow", "mlw", "mzfw", "oew",
    "cg", "centerofgravity", "mac", "lemac", "balance", "trim",
    "gradient", "obstacle", "clearway", "stopway", "toda", "tora",
    "asda", "lda", "acceleratestop", "acceleratego", "v1cut",
    "rto", "rejected", "abort", "continue", "tofl", "field",
    "screen", "height", "path", "segment", "gross", "net",
    "isa", "deviation", "oat", "density", "altitude", "pressure",
    "wet", "dry", "contaminated", "runway", "condition", "rcr",
    # Atmosphère et météo aviation
    "altitude", "pressure", "density", "temperature", "isa",
    "standard", "deviation", "qnh", "qfe", "qne", "transition",
    "flight", "level", "ceiling", "service", "absolute", "density",
    "metar", "taf", "sigmet", "airmet", "pirep", "notam", "atis",
    "wind", "gust", "shear", "turbulence", "cat", "convective",
    "thunderstorm", "cb", "tcu", "fog", "mist", "haze", "smoke",
    "rain", "snow", "ice", "freezing", "rime", "clear", "mixed",
    "visibility", "rvr", "ceiling", "overcast", "broken", "scattered",
    "ifr", "vfr", "mvfr", "lifr", "svfr", "imc", "vmc",

    # =========================================================================
    # MATÉRIAUX ET FABRICATION
    # =========================================================================
    # Métaux détaillés
    "aluminum", "aluminium", "alloy", "steel", "titanium", "inconel",
    "magnesium", "copper", "nickel", "chromium", "cadmium", "zinc",
    "2024", "7075", "7050", "6061", "2014", "clad", "alclad",
    "t3", "t4", "t6", "t73", "t76", "t77", "temper", "heat", "treat",
    "annealed", "normalized", "quenched", "aged", "solution",
    "304", "316", "321", "347", "17-4ph", "15-5ph", "a286", "4130",
    "ti6al4v", "ti64", "6-4", "5553", "beta", "alpha", "grade",
    # Composites détaillés
    "composite", "carbon", "fiber", "fibre", "cfrp", "gfrp", "kevlar",
    "aramid", "honeycomb", "sandwich", "laminate", "ply", "layup",
    "resin", "epoxy", "matrix", "prepreg", "autoclave", "cure",
    "unidirectional", "woven", "fabric", "tape", "tow", "roving",
    "symmetric", "balanced", "quasiisotropic", "orientation",
    "nomex", "korex", "cell", "core", "adhesive", "film", "paste",
    "cocure", "cobond", "secondary", "bond", "scarf", "step", "lap",
    "vbo", "ooa", "rtm", "vartm", "infusion", "filament", "winding",
    # Défauts et dommages détaillés
    "fatigue", "crack", "corrosion", "erosion", "delamination",
    "disbond", "porosity", "void", "inclusion", "dent", "scratch",
    "gouge", "nick", "score", "fretting", "pitting", "exfoliation",
    "stress", "strain", "yield", "ultimate", "fracture", "creep",
    "intergranular", "transgranular", "galvanic", "crevice", "filiform",
    "scc", "stress", "corrosion", "cracking", "hydrogen", "embrittlement",
    "impact", "blunt", "sharp", "barely", "visible", "bvid", "vid",
    "hail", "lightning", "strike", "bird", "tire", "debris",
    # Traitements détaillés
    "anodize", "alodine", "chromate", "conversion", "primer", "paint",
    "sealant", "corrosion", "protection", "inhibitor", "treatment",
    "coating", "epoxy", "polyurethane", "topcoat", "basecoat",
    "shot", "peen", "glass", "bead", "blast", "grit", "chemical",
    "etch", "pickle", "clean", "degrease", "solvent", "aqueous",
    # Assemblage détaillé
    "rivet", "bolt", "nut", "washer", "fastener", "bushing", "grommet",
    "bonding", "adhesive", "weld", "braze", "torque", "preload",
    "solid", "blind", "cherrymax", "huckbolt", "lockbolt", "taper",
    "countersunk", "protruding", "flush", "universal", "brazier",
    "hi-lok", "hi-lite", "eddie", "jo", "nut", "plate", "anchor",
    "floating", "gang", "channel", "clip", "shear", "tension",

    # =========================================================================
    # MAINTENANCE ET INSPECTION
    # =========================================================================
    # Types d'inspection
    "inspection", "check", "visit", "overhaul", "repair", "maintenance",
    "line", "base", "heavy", "acheck", "bcheck", "ccheck", "dcheck",
    "preflight", "postflight", "transit", "daily", "weekly", "overnight",
    "calendar", "cycles", "hours", "landings", "layover", "turnaround",
    "phase", "block", "equalized", "progressive", "letter", "numbered",
    # Méthodes NDT détaillées
    "ndt", "ndi", "nde", "visual", "dye", "penetrant", "magnetic",
    "particle", "eddy", "current", "ultrasonic", "radiographic", "xray",
    "thermographic", "borescope", "videoscope", "tap", "bond", "shearography",
    "holography", "acoustic", "emission", "liquid", "fluorescent",
    "fpi", "mpi", "eci", "hfec", "lfec", "paut", "tofd", "dr", "cr",
    # Actions maintenance détaillées
    "remove", "install", "replace", "repair", "overhaul", "adjust",
    "lubricate", "service", "drain", "fill", "bleed", "purge",
    "test", "check", "verify", "inspect", "examine", "measure",
    "calibrate", "rig", "trim", "balance", "track", "align",
    "disassemble", "reassemble", "clean", "strip", "mask", "protect",
    "torque", "safety", "wire", "lockwire", "cotter", "pin",
    # Documentation détaillée
    "logbook", "workcard", "taskcard", "jobcard", "signoff",
    "release", "return", "service", "airworthy", "unairworthy",
    "defer", "deferral", "mel", "cdl", "nef", "placard", "open",
    "snag", "squawk", "write", "clear", "closed", "duplicate",
    "carry", "forward", "rectify", "correct", "complete", "partial",
    # Intervalles et limites
    "interval", "threshold", "repeat", "limit", "life", "overhaul",
    "onwing", "tbo", "tso", "csl", "llp", "psl", "hard", "soft",
    "calendar", "flight", "hour", "cycle", "landing", "engine",
    "start", "dispatch", "conditional", "mandatory", "recommended",

    # =========================================================================
    # OPÉRATIONS DE VOL
    # =========================================================================
    # Phases de vol
    "flight", "taxi", "takeoff", "rotation", "climb", "cruise",
    "descent", "approach", "landing", "rollout", "parking",
    "pushback", "startup", "shutdown", "turnaround", "holding",
    "pattern", "circuit", "traffic", "downwind", "base", "final",
    "initial", "intermediate", "missed", "goaround", "diversion",
    # Équipage
    "pilot", "captain", "first", "officer", "copilot", "crew",
    "flight", "cabin", "attendant", "engineer", "dispatcher",
    "pic", "sic", "pf", "pm", "pnf", "cfi", "cfii", "mei",
    "loadmaster", "purser", "deadhead", "augmented", "reserve",
    # Procédures détaillées
    "checklist", "procedure", "normal", "abnormal", "emergency",
    "sop", "briefing", "callout", "crosscheck", "confirm",
    "memory", "item", "drill", "evacuation", "ditching", "brace",
    "decompression", "rapid", "explosive", "gradual", "smoke",
    "fire", "fumes", "electrical", "hydraulic", "engine", "failure",
    # Cockpit détaillé
    "cockpit", "flightdeck", "overhead", "pedestal", "glareshield",
    "panel", "annunciator", "warning", "caution", "advisory",
    "master", "fire", "handle", "cutoff", "shutoff", "guard",
    "selector", "switch", "knob", "lever", "button", "pushbutton",
    "rotary", "toggle", "rocker", "indicator", "display", "screen",
    # Phraséologie
    "cleared", "maintain", "climb", "descend", "turn", "heading",
    "direct", "proceed", "hold", "expect", "contact", "squawk",
    "ident", "roger", "wilco", "affirm", "negative", "standby",
    "mayday", "pan", "declaring", "emergency", "fuel", "minimum",

    # =========================================================================
    # SÉCURITÉ ET CERTIFICATION
    # =========================================================================
    # Certification détaillée
    "certification", "airworthiness", "type", "certificate", "tc",
    "stc", "supplemental", "production", "pc", "export", "coa",
    "approval", "validation", "conformity", "compliance", "showing",
    "substantiation", "analysis", "test", "ground", "flight", "rig",
    "special", "condition", "equivalent", "safety", "finding", "elos",
    "issue", "paper", "cri", "certification", "review", "item",
    # Exigences de sécurité détaillées
    "safety", "hazard", "risk", "probability", "severity",
    "catastrophic", "hazardous", "major", "minor", "dal",
    "ssa", "fha", "fmea", "fmeca", "fta", "cca", "zsa", "pssa",
    "asa", "ddi", "dci", "idal", "level", "a", "b", "c", "d", "e",
    "extremely", "improbable", "remote", "reasonably", "probable",
    "frequent", "occasional", "rare", "extremely", "unlikely",
    # Tolérance aux dommages détaillée
    "damage", "tolerance", "fail", "safe", "failsafe", "redundancy",
    "backup", "alternate", "standby", "reversion", "degraded",
    "dispatch", "etops", "lrops", "edto", "entry", "point",
    "adequate", "airport", "diversion", "time", "capability",
    "mei", "engine", "inoperative", "driftdown", "terrain",
    # Continued airworthiness
    "continued", "airworthiness", "ica", "instructions", "cmr",
    "certification", "maintenance", "requirement", "ali", "awl",
    "airworthiness", "limitation", "safe", "life", "retirement",

    # =========================================================================
    # DESIGN ET INGÉNIERIE
    # =========================================================================
    # Conception détaillée
    "design", "analysis", "evaluation", "assessment", "verification",
    "validation", "test", "testing", "measure", "measurement", "criteria",
    "specification", "requirement", "baseline", "configuration", "change",
    "modification", "mod", "embodiment", "retrofit", "incorporation",
    "drawing", "blueprint", "cad", "catia", "nx", "solidworks",
    "model", "simulation", "fem", "fea", "cfd", "nastran", "abaqus",
    # Calculs détaillés
    "formula", "equation", "coefficient", "ratio", "value", "parameter",
    "calculation", "computed", "determined", "maximum", "minimum",
    "nominal", "tolerance", "margin", "factor", "safety", "reserve",
    "fitting", "factor", "knockdown", "scatter", "reliability",
    "allowable", "strength", "ftu", "fty", "fsu", "fsy", "fcu",
    # Charges détaillées
    "load", "stress", "strain", "tension", "compression", "shear",
    "bending", "torsion", "buckling", "fatigue", "static", "dynamic",
    "limit", "ultimate", "proof", "cyclic", "gust", "maneuver",
    "symmetric", "asymmetric", "rolling", "yawing", "pitching",
    "taxi", "landing", "impact", "crash", "emergency", "cabin",
    "pressurization", "thermal", "acoustic", "vibration", "flutter",

    # =========================================================================
    # TYPES D'AÉRONEFS
    # =========================================================================
    # Avions commerciaux
    "narrowbody", "widebody", "regional", "commuter", "turboprop",
    "jet", "twinjet", "trijet", "quadjet", "freighter", "combi",
    "passenger", "cargo", "tanker", "military", "business", "vip",
    # Hélicoptères
    "helicopter", "rotorcraft", "rotor", "main", "tail", "blade",
    "collective", "cyclic", "antitorque", "pedal", "swashplate",
    "mast", "hub", "bearing", "damper", "lag", "lead", "flap",
    "autorotation", "hover", "translational", "lift", "vortex",
    "ringstate", "settlingwith", "power", "retreating", "advancing",
    # UAV/Drones
    "uav", "uas", "rpas", "drone", "unmanned", "remotely", "piloted",
    "autonomous", "semi", "automatic", "manual", "control", "link",
    "payload", "sensor", "camera", "gimbal", "lidar", "radar",

    # =========================================================================
    # INFRASTRUCTURE AÉROPORTUAIRE
    # =========================================================================
    # Piste et voies
    "runway", "taxiway", "apron", "ramp", "gate", "stand", "parking",
    "threshold", "displaced", "stopway", "clearway", "blast", "pad",
    "holding", "point", "intersection", "rapid", "exit", "turnoff",
    "pcn", "acn", "pavement", "concrete", "asphalt", "grooved",
    # Aides à la navigation sol
    "vasi", "papi", "als", "mals", "malsr", "ssalr", "alsf",
    "reil", "rabbit", "sequence", "flash", "lighting", "centerline",
    "edge", "threshold", "touchdown", "zone", "taxiway", "lead",
    "guidance", "docking", "vdgs", "marshalier", "follow", "me",
    # Services
    "fueling", "catering", "cleaning", "lavatory", "water", "gpu",
    "asu", "acu", "pushback", "tow", "tug", "stairs", "jetway",
    "bridge", "belt", "loader", "container", "pallet", "dolly",

    # =========================================================================
    # TERMES FRANÇAIS AÉRONAUTIQUES
    # =========================================================================
    # Structures FR
    "voilure", "fuselage", "empennage", "nacelle", "pylone", "carenage",
    "longeron", "nervure", "lisse", "cadre", "cloison", "panneau",
    "revetement", "doubleur", "ferrure", "eclisse", "gousset",
    # Systèmes FR
    "hydraulique", "pneumatique", "electrique", "carburant", "huile",
    "climatisation", "pressurisation", "oxygene", "incendie", "givre",
    # Commandes FR
    "aileron", "gouverne", "profondeur", "direction", "volet",
    "bec", "destructeur", "portance", "compensateur", "trim",
    # Propulsion FR
    "moteur", "reacteur", "turbine", "compresseur", "chambre",
    "combustion", "tuyere", "inverseur", "poussee", "helice",
    # Maintenance FR
    "inspection", "visite", "revision", "reparation", "entretien",
    "potentiel", "limite", "vie", "calendaire", "cyclique",
    # Performance FR
    "portance", "trainee", "poussee", "masse", "centrage",
    "decrochage", "buffet", "plafond", "autonomie", "rayon",
}

# Stopwords français et anglais pour l'extraction de mots-clés
STOPWORDS = {
    # Français
    "le", "la", "les", "un", "une", "des", "du", "de", "et", "ou", "en",
    "est", "sont", "être", "avoir", "pour", "par", "avec", "dans", "sur",
    "ce", "cette", "ces", "qui", "que", "dont", "où", "ne", "pas", "plus",
    "tout", "tous", "toute", "toutes", "autre", "autres", "même", "aussi",
    "très", "bien", "peut", "doit", "fait", "faire", "comme", "mais", "si",
    # Anglais
    "the", "a", "an", "and", "or", "is", "are", "be", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "this", "that", "these",
    "those", "it", "its", "as", "but", "if", "not", "no", "so", "can",
    "will", "would", "could", "should", "may", "must", "have", "has",
    "been", "being", "was", "were", "which", "what", "when", "where",
}


# =====================================================================
#  REGEX COMPILÉS (optimisation performance)
# =====================================================================

# Regex pour extraction de mots
_RE_WORDS = re.compile(r'\b[a-zA-ZÀ-ÿ]{2,}\b')
_RE_WORDS_3PLUS = re.compile(r'\b[a-zA-ZÀ-ÿ]{3,}\b')
# Regex pour nombres et formules
_RE_NUMBERS = re.compile(r'\d+(?:\.\d+)?')
_RE_FORMULAS = re.compile(r'[=<>≤≥±×÷∑∏√∫]')
# Regex pour phrases
_RE_SENTENCES = re.compile(r'[.!?]+')
# Regex pour listes
_RE_LIST_ITEMS = re.compile(r'^\s*[-*•]\s+|\(\s*[a-z0-9]+\s*\)|\b\d+[.)]\s+', re.MULTILINE)
# Regex pour références EASA
_RE_EASA_REFS = re.compile(r'\b(?:CS|AMC|GM)[-\s]?\d+(?:[.\-]\d+)*[A-Za-z]?', re.IGNORECASE)
# Regex pour majuscules
_RE_UPPERCASE = re.compile(r'\b[A-Z]{2,}\b')
# Regex pour parenthèses
_RE_BRACKETS = re.compile(r'[()[\]{}]')
# Regex pour key_phrases
_RE_DEFINITIONS = re.compile(r'[^.]*(?:means|is defined as|refers to|shall be|is the)[^.]*\.', re.IGNORECASE)
_RE_REQUIREMENTS = re.compile(r'[^.]*(?:shall|must|require[sd]?|mandatory)[^.]*\.', re.IGNORECASE)


# =====================================================================
#  ANALYSE DE DENSITÉ DU CONTENU
# =====================================================================

def _calculate_content_density(text: str) -> Dict[str, Any]:
    """
    Analyse la densité du contenu pour adapter la taille des chunks.

    Métriques analysées:
    - Densité de mots techniques
    - Ratio de nombres/formules
    - Longueur moyenne des phrases
    - Présence de listes/tableaux
    - Ratio de références (CS xx.xxx, etc.)

    Returns:
        Dict avec 'density_score', 'density_type', 'metrics'
    """
    if not text or len(text) < 50:
        return {
            "density_score": 0.5,
            "density_type": "normal",
            "metrics": {},
            "recommended_chunk_size": CHUNK_SIZES["normal"]
        }

    text_lower = text.lower()
    words = _RE_WORDS.findall(text_lower)
    total_words = len(words) if words else 1

    metrics = {}

    # 1. Densité de mots techniques
    technical_count = sum(1 for w in words if w in TECHNICAL_INDICATORS)
    metrics["technical_ratio"] = technical_count / total_words

    # 2. Ratio de nombres et formules
    numbers = _RE_NUMBERS.findall(text)
    formulas = _RE_FORMULAS.findall(text)
    metrics["numeric_ratio"] = (len(numbers) + len(formulas) * 3) / max(len(text), 1) * 100

    # 3. Longueur moyenne des phrases
    sentences = _RE_SENTENCES.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]
    avg_sentence_len = sum(len(s) for s in sentences) / max(len(sentences), 1)
    metrics["avg_sentence_length"] = avg_sentence_len

    # 4. Présence de listes (items numérotés ou à puces)
    list_items = len(_RE_LIST_ITEMS.findall(text))
    metrics["list_density"] = list_items / max(len(sentences), 1)

    # 5. Références EASA (CS xx.xxx, AMC, GM)
    refs = len(_RE_EASA_REFS.findall(text))
    metrics["reference_density"] = refs / total_words * 100

    # 6. Ratio majuscules (acronymes, titres)
    upper_words = len(_RE_UPPERCASE.findall(text))
    metrics["uppercase_ratio"] = upper_words / total_words

    # 7. Densité de parenthèses/crochets (souvent formules ou références)
    brackets = len(_RE_BRACKETS.findall(text))
    metrics["bracket_density"] = brackets / max(len(text), 1) * 100

    # Calculer le score de densité (0-1, plus haut = plus dense)
    density_score = (
        metrics["technical_ratio"] * 2.0 +
        min(metrics["numeric_ratio"], 5) / 5 * 1.5 +
        min(metrics["list_density"], 2) / 2 * 1.0 +
        min(metrics["reference_density"], 3) / 3 * 1.5 +
        metrics["uppercase_ratio"] * 1.0 +
        min(metrics["bracket_density"], 3) / 3 * 1.0 +
        (1 - min(avg_sentence_len, 200) / 200) * 0.5  # Phrases courtes = plus dense
    ) / 8.5

    density_score = min(max(density_score, 0), 1)

    # Déterminer le type de densité (seuils ajustés pour documents techniques)
    if density_score >= 0.5:
        density_type = "very_dense"
    elif density_score >= 0.3:
        density_type = "dense"
    elif density_score >= 0.15:
        density_type = "normal"
    else:
        density_type = "sparse"

    return {
        "density_score": round(density_score, 3),
        "density_type": density_type,
        "metrics": metrics,
        "recommended_chunk_size": CHUNK_SIZES[density_type]
    }


def _get_adaptive_chunk_size(
    text: str,
    base_size: int = 1500,
    min_size: int = 600,
    max_size: int = 2500,
) -> int:
    """
    Calcule la taille optimale de chunk basée sur la densité du contenu.

    Args:
        text: Texte à analyser
        base_size: Taille de base
        min_size: Taille minimum
        max_size: Taille maximum

    Returns:
        Taille de chunk recommandée
    """
    density_info = _calculate_content_density(text)
    recommended = density_info["recommended_chunk_size"]

    # Ajuster par rapport à la base fournie
    ratio = recommended / CHUNK_SIZES["normal"]
    adjusted_size = int(base_size * ratio)

    return max(min_size, min(adjusted_size, max_size))


# =====================================================================
#  EXTRACTION DE MOTS-CLÉS POUR AUGMENTATION
# =====================================================================

def _extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extrait les mots-clés importants d'un texte.
    Utilise des regex compilés pour la performance.
    """
    if not text or len(text) < 20:
        return []

    # Tokenisation avec regex compilé
    words = _RE_WORDS_3PLUS.findall(text.lower())

    # Filtrer les stopwords
    filtered_words = [w for w in words if w not in STOPWORDS]

    # Compter les fréquences
    word_counts = Counter(filtered_words)

    # Bonus pour les termes techniques
    scored_words = {}
    for word, count in word_counts.items():
        score = count
        if word in TECHNICAL_INDICATORS:
            score *= 2.0
        if len(word) > 8:
            score *= 1.3
        scored_words[word] = score

    # Extraire les codes de référence avec regex compilé
    refs = _RE_EASA_REFS.findall(text)
    ref_keywords = list(set(r.upper() for r in refs))[:3]

    # Trier par score et prendre les top
    sorted_words = sorted(scored_words.items(), key=lambda x: x[1], reverse=True)
    keywords = [w for w, _ in sorted_words[:max_keywords - len(ref_keywords)]]

    return ref_keywords + keywords


def _extract_key_phrases(text: str, max_phrases: int = 5) -> List[str]:
    """
    Extrait les phrases ou segments clés d'un texte.
    Utilise des regex compilés pour la performance.
    """
    if not text or len(text) < 50:
        return []

    # Chercher avec regex compilés
    definitions = _RE_DEFINITIONS.findall(text)
    requirements = _RE_REQUIREMENTS.findall(text)

    # Combiner et dédupliquer
    all_phrases = []
    seen = set()

    for phrase in definitions + requirements:
        phrase = phrase.strip()
        if 20 < len(phrase) < 300 and phrase not in seen:
            all_phrases.append(phrase)
            seen.add(phrase)
            if len(all_phrases) >= max_phrases:
                break

    return all_phrases


# =====================================================================
#  AUGMENTATION DES CHUNKS
# =====================================================================

def augment_chunk(
    chunk: Dict[str, Any],
    add_keywords: bool = True,
    add_key_phrases: bool = True,
    add_density_info: bool = True,
) -> Dict[str, Any]:
    """
    Enrichit un chunk avec des métadonnées pour améliorer la recherche.

    Args:
        chunk: Dict avec au minimum 'text'
        add_keywords: Extraire et ajouter les mots-clés
        add_key_phrases: Extraire les phrases clés
        add_density_info: Ajouter l'analyse de densité

    Returns:
        Chunk enrichi avec métadonnées
    """
    text = chunk.get("text", "")

    if add_keywords:
        keywords = _extract_keywords(text)
        chunk["keywords"] = keywords
        # Créer une version enrichie du texte pour l'embedding
        if keywords:
            keyword_str = " | ".join(keywords[:5])
            chunk["enriched_text"] = f"[Keywords: {keyword_str}]\n\n{text}"

    if add_key_phrases:
        key_phrases = _extract_key_phrases(text)
        chunk["key_phrases"] = key_phrases

    if add_density_info:
        density = _calculate_content_density(text)
        chunk["density_type"] = density["density_type"]
        chunk["density_score"] = density["density_score"]

    return chunk


def augment_chunks(chunks: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
    """
    Augmente une liste de chunks avec des métadonnées (parallélisé).
    """
    if len(chunks) < 10:
        # Pas de parallélisme pour peu de chunks
        return [augment_chunk(chunk, **kwargs) for chunk in chunks]

    # Paralléliser l'augmentation pour beaucoup de chunks
    max_workers = min(multiprocessing.cpu_count(), len(chunks), 8)

    def _augment_with_kwargs(chunk):
        return augment_chunk(chunk, **kwargs)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_augment_with_kwargs, chunks))

    return results


# =====================================================================
#  CHUNKS HIÉRARCHIQUES PARENT-ENFANT
# =====================================================================

def create_parent_child_chunks(
    text: str,
    source_file: str = "",
    parent_size: int = 3000,
    child_size: int = 800,
    child_overlap: int = 100,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Crée une structure hiérarchique de chunks parent-enfant.

    - Parent: Large contexte pour la compréhension globale
    - Enfant: Chunks précis pour la recherche

    Le parent est utilisé pour fournir du contexte après récupération
    des enfants pertinents.

    Args:
        text: Texte source
        source_file: Nom du fichier source
        parent_size: Taille des chunks parent
        child_size: Taille des chunks enfant
        child_overlap: Overlap entre chunks enfant

    Returns:
        Tuple (parent_chunks, child_chunks)
    """
    if not text or len(text) < child_size:
        # Document trop petit - un seul chunk
        chunk = {
            "text": text.strip(),
            "source": source_file,
            "chunk_id": f"{source_file}_p0",
            "is_parent": True,
            "children_ids": [],
            "chunk_index": 0,
        }
        return [chunk], []

    import hashlib

    def generate_chunk_id(content: str, prefix: str = "c") -> str:
        """Génère un ID unique pour un chunk."""
        hash_val = hashlib.md5(content[:100].encode()).hexdigest()[:8]
        return f"{source_file}_{prefix}{hash_val}"

    parent_chunks = []
    child_chunks = []

    # Créer les chunks parent (larges)
    parent_raw = simple_chunk(text, chunk_size=parent_size, overlap=200)

    for p_idx, parent_text in enumerate(parent_raw):
        parent_id = generate_chunk_id(parent_text, f"p{p_idx}_")

        # Créer les chunks enfant pour ce parent
        children_ids = []
        child_raw = simple_chunk(parent_text, chunk_size=child_size, overlap=child_overlap)

        for c_idx, child_text in enumerate(child_raw):
            child_id = generate_chunk_id(child_text, f"c{p_idx}_{c_idx}_")
            children_ids.append(child_id)

            # Extraire les mots-clés pour l'enfant
            keywords = _extract_keywords(child_text, max_keywords=8)

            child_chunk = {
                "text": child_text,
                "source": source_file,
                "chunk_id": child_id,
                "parent_id": parent_id,
                "is_parent": False,
                "chunk_index": len(child_chunks),
                "parent_index": p_idx,
                "keywords": keywords,
            }

            # Créer le texte enrichi pour l'embedding
            if keywords:
                child_chunk["enriched_text"] = f"[{' | '.join(keywords[:5])}]\n{child_text}"

            child_chunks.append(child_chunk)

        # Créer le chunk parent
        parent_chunk = {
            "text": parent_text,
            "source": source_file,
            "chunk_id": parent_id,
            "is_parent": True,
            "children_ids": children_ids,
            "chunk_index": p_idx,
        }
        parent_chunks.append(parent_chunk)

    return parent_chunks, child_chunks


def get_parent_for_child(
    child_chunk: Dict[str, Any],
    parent_chunks: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Retrouve le chunk parent d'un chunk enfant.

    Args:
        child_chunk: Chunk enfant avec 'parent_id'
        parent_chunks: Liste des chunks parent

    Returns:
        Chunk parent ou None
    """
    parent_id = child_chunk.get("parent_id")
    if not parent_id:
        return None

    for parent in parent_chunks:
        if parent.get("chunk_id") == parent_id:
            return parent

    return None


# =====================================================================
#  CHUNKING ADAPTATIF AUTOMATIQUE
# =====================================================================

def adaptive_smart_chunk(
    text: str,
    source_file: str = "",
    base_chunk_size: int = 1500,
    enable_hierarchy: bool = True,
    enable_augmentation: bool = True,
    enable_density_adaptation: bool = True,
) -> Dict[str, Any]:
    """
    Fonction principale de chunking adaptatif qui combine toutes les techniques.

    Cette fonction analyse automatiquement le contenu et applique:
    1. Adaptation de taille basée sur la densité
    2. Structure hiérarchique parent-enfant
    3. Augmentation avec mots-clés et métadonnées
    4. Smart chunking selon le type de document

    Args:
        text: Texte à chunker
        source_file: Nom du fichier source
        base_chunk_size: Taille de base (sera adaptée)
        enable_hierarchy: Créer des chunks parent-enfant
        enable_augmentation: Ajouter mots-clés et métadonnées
        enable_density_adaptation: Adapter la taille selon la densité

    Returns:
        Dict avec:
        - 'chunks': Liste des chunks principaux (ou enfants si hiérarchie)
        - 'parent_chunks': Chunks parent (si hiérarchie activée)
        - 'density_info': Info sur la densité globale
        - 'config': Configuration utilisée
    """
    if not text or len(text.strip()) < 100:
        return {
            "chunks": [],
            "parent_chunks": [],
            "density_info": None,
            "config": {"error": "Text too short"}
        }

    # Analyser la densité globale du document
    density_info = _calculate_content_density(text)

    # Adapter la taille de chunk selon la densité
    if enable_density_adaptation:
        chunk_size = _get_adaptive_chunk_size(
            text,
            base_size=base_chunk_size,
            min_size=600,
            max_size=2500
        )
    else:
        chunk_size = base_chunk_size

    # Configuration utilisée
    config = {
        "base_chunk_size": base_chunk_size,
        "adapted_chunk_size": chunk_size,
        "density_type": density_info["density_type"],
        "hierarchy_enabled": enable_hierarchy,
        "augmentation_enabled": enable_augmentation,
    }

    result = {
        "chunks": [],
        "parent_chunks": [],
        "density_info": density_info,
        "config": config
    }

    if enable_hierarchy:
        # Créer une structure hiérarchique
        parent_size = chunk_size * 2  # Parents 2x plus grands que les enfants
        child_size = chunk_size

        parent_chunks, child_chunks = create_parent_child_chunks(
            text,
            source_file=source_file,
            parent_size=parent_size,
            child_size=child_size,
        )

        if enable_augmentation:
            child_chunks = augment_chunks(child_chunks)
            parent_chunks = augment_chunks(parent_chunks, add_key_phrases=False)

        result["chunks"] = child_chunks
        result["parent_chunks"] = parent_chunks

    else:
        # Chunking smart standard
        chunks = smart_chunk_generic(
            text,
            source_file=source_file,
            chunk_size=chunk_size,
            min_chunk_size=max(200, chunk_size // 5),
        )

        if enable_augmentation:
            chunks = augment_chunks(chunks)

        result["chunks"] = chunks

    return result


def adaptive_chunk_document(
    text: str,
    source_file: str = "",
    is_easa: bool = False,
    sections: Optional[List[Dict[str, Any]]] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Point d'entrée simplifié pour le chunking adaptatif.

    Détecte automatiquement le type de document et applique
    la meilleure stratégie de chunking.

    Args:
        text: Texte à chunker
        source_file: Nom du fichier source
        is_easa: True si document EASA avec sections
        sections: Sections EASA pré-découpées (optionnel)
        **kwargs: Arguments supplémentaires pour adaptive_smart_chunk

    Returns:
        Liste de chunks prêts pour l'indexation
    """
    if is_easa and sections:
        # Document EASA - utiliser le chunking spécialisé
        chunks = chunk_easa_sections(
            sections,
            max_chunk_size=kwargs.get("base_chunk_size", 1500),
        )

        # Augmenter les chunks EASA aussi
        if kwargs.get("enable_augmentation", True):
            chunks = augment_chunks(chunks)

        return chunks

    # Document générique - chunking adaptatif complet
    result = adaptive_smart_chunk(text, source_file=source_file, **kwargs)

    # Retourner les chunks enfants (plus précis pour la recherche)
    return result["chunks"]


def _split_into_paragraphs(text: str) -> List[str]:
    """
    Découpe le texte en paragraphes (séparés par au moins une ligne vide).
    """
    text = text.replace("\r\n", "\n")
    raw_paras = re.split(r"\n\s*\n", text)
    paras: List[str] = []
    for p in raw_paras:
        p = p.strip()
        if p:
            paras.append(p)
    return paras


def _chunk_block(text: str, chunk_size: int) -> List[str]:
    """
    Découpe un bloc de texte en sous-blocs ≤ chunk_size, en essayant de couper
    sur des fins de phrase / mots.
    """
    chunks: List[str] = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        # essayer de reculer jusqu'à un point / fin de phrase
        cut = end
        for sep in [". ", "; ", ", ", " "]:
            idx = text.rfind(sep, start, end)
            if idx != -1 and idx > start + int(chunk_size * 0.4):
                cut = idx + len(sep)
                break
        chunk = text[start:cut].strip()
        if chunk:
            chunks.append(chunk)
        start = cut
    return chunks


def _add_overlap(chunks: List[str], overlap: int) -> List[str]:
    """
    Ajoute un overlap en réinjectant la fin du chunk précédent au début du suivant.
    Overlap exprimé en nombre de caractères (approx).
    """
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    new_chunks: List[str] = []
    prev_tail = ""

    for i, ch in enumerate(chunks):
        if i == 0:
            new_chunks.append(ch)
        else:
            prefix = prev_tail
            combined = (prefix + "\n" + ch).strip()
            new_chunks.append(combined)

        # mettre à jour le tail pour le prochain
        if len(ch) > overlap:
            prev_tail = ch[-overlap:]
        else:
            prev_tail = ch

    return new_chunks


def simple_chunk(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> List[str]:
    """
    Chunking simple :

      - découpe en paragraphes
      - concatène les paragraphes jusqu'à chunk_size
      - si un paragraphe est très long, on le re-découpe avec _chunk_block
      - ajoute un overlap entre chunks

    Retourne une liste de chaînes (chunks).
    """
    if not text:
        return []

    text = text.strip()
    if not text:
        return []

    paras = _split_into_paragraphs(text)
    if not paras:
        return [text] if len(text) <= chunk_size else _add_overlap(
            _chunk_block(text, chunk_size), overlap
        )

    blocks: List[str] = []
    current: List[str] = []
    current_len = 0

    for p in paras:
        L = len(p)
        if L > int(chunk_size * 1.2):
            # paragraphe énorme : flush ce qu'on a, puis découpe lui-même
            if current:
                blocks.append("\n\n".join(current))
                current = []
                current_len = 0
            sub_chunks = _chunk_block(p, chunk_size)
            blocks.extend(sub_chunks)
            continue

        if current_len + L + 2 <= chunk_size:
            current.append(p)
            current_len += L + 2
        else:
            if current:
                blocks.append("\n\n".join(current))
            current = [p]
            current_len = L

    if current:
        blocks.append("\n\n".join(current))

    chunks = _add_overlap(blocks, overlap)
    return chunks


# =====================================================================
#  SMART CHUNKING POUR DOCUMENTS GÉNÉRIQUES
# =====================================================================

# Patterns pour détecter les titres/headers
HEADER_PATTERNS = [
    # Titres numérotés courts: "1. Introduction", "1.1 Overview" (max 60 chars pour éviter faux positifs)
    re.compile(r'^(\d+(?:\.\d+)*\.?\s+[A-Z][^\n]{0,55})$', re.MULTILINE),
    # Titres en majuscules seules
    re.compile(r'^([A-Z][A-Z\s]{5,60})$', re.MULTILINE),
    # Titres avec "Chapter", "Section", "Part"
    re.compile(r'^((?:Chapter|Section|Part|Appendix|Annex)\s+[\dIVXA-Z]+[:\s\-]*[^\n]*)$', re.MULTILINE | re.IGNORECASE),
]

# Pattern pour détecter les tables markdown
TABLE_PATTERN = re.compile(r'^\|.+\|$', re.MULTILINE)
TABLE_SEPARATOR = re.compile(r'^\|[\s\-:|]+\|$', re.MULTILINE)

# Patterns pour détecter les listes
LIST_PATTERNS = [
    # Listes à puces: -, *, •
    re.compile(r'^[\s]*[-*•]\s+.+', re.MULTILINE),
    # Listes numérotées: 1., 2., a), b), (1), (a)
    re.compile(r'^[\s]*(?:\d+[.)]|\([a-z0-9]+\)|[a-z][.)]\s).+', re.MULTILINE | re.IGNORECASE),
]


def _is_header(text: str) -> bool:
    """
    Vérifie si un paragraphe est un titre/header.

    Heuristiques pour distinguer headers des items de liste:
    - Headers sont courts (< 80 chars pour numérotés)
    - Headers n'ont pas de contenu descriptif long
    - Items de liste commencent par du contenu descriptif
    """
    text = text.strip()
    if len(text) > 150 or len(text) < 3:
        return False

    # Si ça match un pattern de liste numérotée et c'est long, c'est probablement un item de liste
    numbered_match = re.match(r'^(\d+[.)]\s+)', text)
    if numbered_match:
        after_number = text[numbered_match.end():]
        # Si après le numéro il y a beaucoup de texte (> 50 chars), c'est une liste
        if len(after_number) > 50:
            return False
        # Si ça commence par un verbe/action courant, c'est une liste
        list_starters = ['the ', 'a ', 'an ', 'this ', 'that ', 'each ', 'all ',
                         'any ', 'no ', 'must ', 'shall ', 'should ', 'may ',
                         'can ', 'will ', 'limit', 'maximum', 'minimum', 'ensure',
                         'provide', 'maintain', 'prevent', 'allow', 'require']
        after_lower = after_number.lower()
        if any(after_lower.startswith(s) for s in list_starters):
            return False

    for pattern in HEADER_PATTERNS:
        if pattern.match(text):
            return True
    return False


def _is_list_item(text: str) -> bool:
    """Vérifie si un texte commence par un item de liste."""
    for pattern in LIST_PATTERNS:
        if pattern.match(text.strip()):
            return True
    return False


def _is_list_block(text: str) -> bool:
    """Vérifie si un bloc est une liste (plusieurs items)."""
    lines = text.strip().split('\n')
    if len(lines) < 2:
        return False
    list_items = sum(1 for line in lines if _is_list_item(line))
    return list_items >= len(lines) * 0.6  # Au moins 60% des lignes sont des items


def _is_table_block(text: str) -> bool:
    """
    Vérifie si un bloc est une table markdown.

    Une table markdown a:
    - Plusieurs lignes commençant et finissant par |
    - Une ligne de séparation avec des tirets |---|---|
    """
    lines = text.strip().split('\n')
    if len(lines) < 2:
        return False

    # Compter les lignes qui ressemblent à des lignes de table
    table_lines = sum(1 for line in lines if TABLE_PATTERN.match(line.strip()))
    has_separator = any(TABLE_SEPARATOR.match(line.strip()) for line in lines)

    # Au moins 2 lignes de table et un séparateur
    return table_lines >= 2 and has_separator


def _split_into_sentences(text: str) -> List[str]:
    """Découpe un texte en phrases."""
    # Pattern pour fin de phrase (. ! ? suivis d'espace ou fin)
    sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$')
    sentences = sentence_pattern.split(text)
    return [s.strip() for s in sentences if s.strip()]


def _detect_document_structure(text: str) -> List[Dict[str, Any]]:
    """
    Détecte la structure d'un document (titres, sections, listes, tables).

    Returns:
        Liste de blocs avec type ('header', 'paragraph', 'list', 'table') et contenu
    """
    paragraphs = _split_into_paragraphs(text)

    blocks = []
    current_header = None

    for para in paragraphs:
        if _is_header(para):
            blocks.append({
                "type": "header",
                "content": para,
                "level": _detect_header_level(para)
            })
            current_header = para
        elif _is_table_block(para):
            blocks.append({
                "type": "table",
                "content": para,
                "parent_header": current_header
            })
        elif _is_list_block(para):
            blocks.append({
                "type": "list",
                "content": para,
                "parent_header": current_header
            })
        else:
            blocks.append({
                "type": "paragraph",
                "content": para,
                "parent_header": current_header
            })

    return blocks


def _detect_header_level(header: str) -> int:
    """Détecte le niveau d'un header (1, 2, 3...)."""
    header = header.strip()

    # Compter les points dans la numérotation: "1.2.3" -> niveau 3
    match = re.match(r'^(\d+(?:\.\d+)*)', header)
    if match:
        return match.group(1).count('.') + 1

    # Chapitre/Section/Part = niveau 1-2
    if re.match(r'^(?:Chapter|Part)', header, re.IGNORECASE):
        return 1
    if re.match(r'^(?:Section|Appendix|Annex)', header, re.IGNORECASE):
        return 2

    # Titres en majuscules = niveau 1
    if header.isupper():
        return 1

    return 2  # Par défaut niveau 2


def smart_chunk_generic(
    text: str,
    source_file: str = "",
    chunk_size: int = 1500,
    min_chunk_size: int = 200,
    overlap: int = 100,
    add_source_prefix: bool = True,
    preserve_lists: bool = True,
    preserve_headers: bool = True,
) -> List[Dict[str, Any]]:
    """
    Chunking intelligent pour documents génériques.

    Features:
    1. Détection des titres/headers - garde le titre avec son contenu
    2. Préservation des listes - ne coupe pas une liste
    3. Coupure aux phrases - coupe en fin de phrase
    4. Contexte source - ajoute [Source: filename] en préfixe
    5. Détection de structure - respecte les sections

    Args:
        text: Texte à chunker
        source_file: Nom du fichier source (pour le préfixe)
        chunk_size: Taille max d'un chunk
        min_chunk_size: Taille min (fusion si en dessous)
        overlap: Chevauchement entre chunks
        add_source_prefix: Ajouter préfixe [Source: ...]
        preserve_lists: Ne pas couper les listes
        preserve_headers: Garder le header avec son contenu

    Returns:
        Liste de dicts avec text, header, type, etc.
    """
    if not text or not text.strip():
        return []

    # Préfixe source
    source_prefix = f"[Source: {source_file}]" if add_source_prefix and source_file else ""
    prefix_len = len(source_prefix) + 2 if source_prefix else 0
    effective_max = chunk_size - prefix_len

    # Détecter la structure
    blocks = _detect_document_structure(text)

    if not blocks:
        # Fallback sur simple_chunk
        raw_chunks = simple_chunk(text, chunk_size=effective_max, overlap=overlap)
        return [{"text": f"{source_prefix}\n\n{ch}".strip() if source_prefix else ch,
                 "chunk_index": i, "header": None, "type": "text"}
                for i, ch in enumerate(raw_chunks)]

    chunks = []
    current_text = ""
    current_header = None
    chunk_index = 0

    def flush_chunk():
        nonlocal current_text, current_header, chunk_index
        if current_text and len(current_text.strip()) >= min_chunk_size:
            # Ajouter le header au début si présent
            chunk_content = current_text.strip()
            if current_header and preserve_headers:
                chunk_content = f"[{current_header}]\n\n{chunk_content}"

            # Ajouter le préfixe source
            if source_prefix:
                chunk_content = f"{source_prefix}\n\n{chunk_content}"

            chunks.append({
                "text": chunk_content.strip(),
                "chunk_index": chunk_index,
                "header": current_header,
                "type": "structured",
            })
            chunk_index += 1
        current_text = ""

    for block in blocks:
        block_type = block["type"]
        content = block["content"]

        if block_type == "header":
            # Nouveau header = nouveau chunk potentiel
            if current_text:
                flush_chunk()
            current_header = content
            # Ne pas ajouter le header au contenu, il sera ajouté en préfixe
            continue

        # Vérifier si on peut ajouter ce bloc au chunk courant
        potential_len = len(current_text) + len(content) + 2

        if block_type == "list" and preserve_lists:
            # Liste: essayer de la garder entière
            if len(content) > effective_max:
                # Liste trop grande: flush et chunker la liste
                flush_chunk()
                list_chunks = _chunk_list(content, effective_max)
                for i, lc in enumerate(list_chunks):
                    chunk_content = lc
                    if current_header and preserve_headers:
                        chunk_content = f"[{current_header}]\n\n{chunk_content}"
                    if source_prefix:
                        chunk_content = f"{source_prefix}\n\n{chunk_content}"
                    chunks.append({
                        "text": chunk_content.strip(),
                        "chunk_index": chunk_index,
                        "header": current_header,
                        "type": "list",
                    })
                    chunk_index += 1
                continue
            elif potential_len > effective_max:
                flush_chunk()

        elif block_type == "table":
            # Table: garder entière comme unité sémantique
            if len(content) > effective_max:
                # Table trop grande: flush et la mettre dans son propre chunk
                flush_chunk()
                chunk_content = content
                if current_header and preserve_headers:
                    chunk_content = f"[{current_header}]\n\n{chunk_content}"
                if source_prefix:
                    chunk_content = f"{source_prefix}\n\n{chunk_content}"
                chunks.append({
                    "text": chunk_content.strip(),
                    "chunk_index": chunk_index,
                    "header": current_header,
                    "type": "table",
                })
                chunk_index += 1
                continue
            elif potential_len > effective_max:
                flush_chunk()

        elif potential_len > effective_max:
            # Bloc trop grand pour le chunk courant
            if block_type == "paragraph" and len(content) > effective_max:
                # Paragraphe énorme: flush et découper par phrases
                flush_chunk()
                para_chunks = _chunk_by_sentences(content, effective_max)
                for pc in para_chunks:
                    current_text = pc
                    flush_chunk()
                continue
            else:
                flush_chunk()

        # Ajouter le bloc au chunk courant
        if current_text:
            current_text += "\n\n" + content
        else:
            current_text = content

    # Flush final
    flush_chunk()

    # Fusionner les petits chunks
    chunks = _merge_small_chunks(chunks, min_chunk_size, chunk_size)

    # Ajouter l'overlap
    chunks = _add_overlap_to_smart_chunks(chunks, overlap)

    return chunks


def _chunk_list(list_text: str, max_size: int) -> List[str]:
    """Découpe une liste en gardant les items ensemble."""
    lines = list_text.strip().split('\n')

    chunks = []
    current = []
    current_len = 0

    for line in lines:
        line_len = len(line) + 1
        if current_len + line_len > max_size and current:
            chunks.append('\n'.join(current))
            current = [line]
            current_len = line_len
        else:
            current.append(line)
            current_len += line_len

    if current:
        chunks.append('\n'.join(current))

    return chunks


def _chunk_by_sentences(text: str, max_size: int) -> List[str]:
    """Découpe un texte en chunks en coupant aux fins de phrases."""
    sentences = _split_into_sentences(text)

    if not sentences:
        return _chunk_block(text, max_size)

    chunks = []
    current = []
    current_len = 0

    for sentence in sentences:
        sent_len = len(sentence) + 1

        if sent_len > max_size:
            # Phrase trop longue: flush et découper
            if current:
                chunks.append(' '.join(current))
                current = []
                current_len = 0
            sub_chunks = _chunk_block(sentence, max_size)
            chunks.extend(sub_chunks)
            continue

        if current_len + sent_len > max_size and current:
            chunks.append(' '.join(current))
            current = [sentence]
            current_len = sent_len
        else:
            current.append(sentence)
            current_len += sent_len

    if current:
        chunks.append(' '.join(current))

    return chunks


def _merge_small_chunks(chunks: List[Dict[str, Any]], min_size: int, max_size: int) -> List[Dict[str, Any]]:
    """Fusionne les chunks trop petits avec le précédent ou suivant."""
    if len(chunks) <= 1:
        return chunks

    merged = []
    i = 0

    while i < len(chunks):
        chunk = chunks[i]
        text = chunk["text"]

        if len(text) < min_size and merged:
            # Essayer de fusionner avec le précédent
            last = merged[-1]
            combined_len = len(last["text"]) + len(text) + 2

            if combined_len <= max_size + 200:
                last["text"] = last["text"] + "\n\n" + text
                i += 1
                continue

        if len(text) < min_size and i + 1 < len(chunks):
            # Essayer de fusionner avec le suivant
            next_chunk = chunks[i + 1]
            combined_len = len(text) + len(next_chunk["text"]) + 2

            if combined_len <= max_size + 200:
                chunk["text"] = text + "\n\n" + next_chunk["text"]
                chunk["header"] = chunk.get("header") or next_chunk.get("header")
                merged.append(chunk)
                i += 2
                continue

        merged.append(chunk)
        i += 1

    # Renuméroter les chunks
    for i, ch in enumerate(merged):
        ch["chunk_index"] = i

    return merged


def _add_overlap_to_smart_chunks(chunks: List[Dict[str, Any]], overlap: int) -> List[Dict[str, Any]]:
    """Ajoute un overlap entre les smart chunks."""
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    for i in range(1, len(chunks)):
        prev_text = chunks[i - 1]["text"]

        # Extraire la fin du chunk précédent
        if len(prev_text) > overlap:
            # Essayer de couper à une fin de phrase
            tail = prev_text[-overlap * 2:]
            sentence_end = max(
                tail.rfind('. '),
                tail.rfind('? '),
                tail.rfind('! ')
            )
            if sentence_end > overlap // 2:
                overlap_text = tail[sentence_end + 2:]
            else:
                overlap_text = prev_text[-overlap:]
        else:
            overlap_text = prev_text

        # Ajouter au début du chunk courant (après le préfixe si présent)
        current_text = chunks[i]["text"]

        # Trouver où insérer l'overlap (après [Source:] et [Header])
        insert_pos = 0
        if current_text.startswith("[Source:"):
            end_bracket = current_text.find("]\n\n")
            if end_bracket >= 0:  # >= 0 car find() retourne -1 si non trouvé
                insert_pos = end_bracket + 3

        if insert_pos < len(current_text) and current_text[insert_pos:].startswith("["):
            end_bracket = current_text.find("]\n\n", insert_pos)
            if end_bracket >= 0:  # >= 0 car find() retourne -1 si non trouvé
                insert_pos = end_bracket + 3

        # Insérer l'overlap
        prefix = current_text[:insert_pos]
        suffix = current_text[insert_pos:]
        chunks[i]["text"] = f"{prefix}...{overlap_text}\n\n{suffix}".strip()

    return chunks

def _build_context_prefix(section: Dict[str, Any]) -> str:
    """
    Construit un préfixe de contexte pour un chunk.
    Ex: "[CS 25.1309 - Equipment, systems and installations]"
    """
    sec_id = section.get("id", "").strip()
    sec_title = section.get("title", "").strip()

    if sec_id and sec_title:
        return f"[{sec_id} - {sec_title}]"
    elif sec_id:
        return f"[{sec_id}]"
    return ""


def _detect_subsections(text: str) -> List[Dict[str, Any]]:
    """
    Détecte les sous-sections dans un texte EASA.
    Ex: (a), (b), (1), (2), (i), (ii)

    Returns:
        Liste de dicts avec 'marker', 'start', 'end', 'content'
    """
    # Pattern pour détecter les marqueurs de sous-section en début de ligne
    subsection_pattern = re.compile(
        r'^(\s*)(\([a-z]\)|\([0-9]+\)|\([ivx]+\))\s*',
        re.MULTILINE | re.IGNORECASE
    )

    matches = list(subsection_pattern.finditer(text))

    if not matches:
        return []

    subsections = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        subsections.append({
            "marker": match.group(2),
            "indent": len(match.group(1)),
            "start": start,
            "end": end,
            "content": text[start:end].strip()
        })

    return subsections


def smart_chunk_section(
    section: Dict[str, Any],
    max_chunk_size: int = 1500,
    min_chunk_size: int = 200,
    overlap: int = 100,
    add_context_prefix: bool = True,
) -> List[Dict[str, Any]]:
    """
    Chunking intelligent d'une section EASA.

    Règles:
    1. Si section < max_chunk_size : garder en un seul chunk
    2. Sinon, découper par sous-sections (a), (b), etc.
    3. Chaque chunk reçoit le préfixe de contexte [CS xx.xxx - Title]
    4. Les sous-sections trop petites sont fusionnées

    Args:
        section: Dict avec 'id', 'kind', 'number', 'title', 'full_text'
        max_chunk_size: Taille max avant découpage
        min_chunk_size: Taille min (fusion si en dessous)
        overlap: Chevauchement entre chunks
        add_context_prefix: Ajouter le préfixe [CS xx.xxx] à chaque chunk

    Returns:
        Liste de dicts avec 'text', 'section_id', 'section_title', 'chunk_index'
    """
    sec_id = section.get("id", "").strip()
    sec_title = section.get("title", "").strip()
    sec_kind = section.get("kind", "").strip()
    full_text = section.get("full_text", "").strip()

    if not full_text:
        return []

    context_prefix = _build_context_prefix(section) if add_context_prefix else ""
    prefix_len = len(context_prefix) + 2 if context_prefix else 0  # +2 pour \n\n

    effective_max = max_chunk_size - prefix_len

    chunks = []

    # Cas 1: Section assez petite → un seul chunk
    if len(full_text) <= effective_max:
        chunk_text = f"{context_prefix}\n\n{full_text}" if context_prefix else full_text
        chunks.append({
            "text": chunk_text.strip(),
            "section_id": sec_id,
            "section_kind": sec_kind,
            "section_title": sec_title,
            "chunk_index": 0,
            "is_complete_section": True,
        })
        return chunks

    # Cas 2: Section longue → essayer de découper par sous-sections
    subsections = _detect_subsections(full_text)

    if subsections:
        # Découpage par sous-sections avec fusion des petites
        current_text = ""
        current_markers = []
        chunk_index = 0

        # Texte avant la première sous-section (intro)
        intro_text = full_text[:subsections[0]["start"]].strip()
        if intro_text:
            current_text = intro_text

        for i, subsec in enumerate(subsections):
            subsec_content = subsec["content"]

            # Vérifier si on peut ajouter cette sous-section au chunk courant
            potential_len = len(current_text) + len(subsec_content) + 2

            if potential_len <= effective_max:
                # Ajouter au chunk courant
                if current_text:
                    current_text += "\n\n" + subsec_content
                else:
                    current_text = subsec_content
                current_markers.append(subsec["marker"])
            else:
                # Flush le chunk courant si non vide
                if current_text and len(current_text) >= min_chunk_size:
                    chunk_text = f"{context_prefix}\n\n{current_text}" if context_prefix else current_text
                    chunks.append({
                        "text": chunk_text.strip(),
                        "section_id": sec_id,
                        "section_kind": sec_kind,
                        "section_title": sec_title,
                        "chunk_index": chunk_index,
                        "subsections": current_markers.copy(),
                        "is_complete_section": False,
                    })
                    chunk_index += 1
                    current_text = ""
                    current_markers = []

                # Si la sous-section elle-même est trop grande, la découper
                if len(subsec_content) > effective_max:
                    sub_chunks = _chunk_block(subsec_content, effective_max)
                    for j, sub_chunk in enumerate(sub_chunks):
                        chunk_text = f"{context_prefix}\n\n{sub_chunk}" if context_prefix else sub_chunk
                        chunks.append({
                            "text": chunk_text.strip(),
                            "section_id": sec_id,
                            "section_kind": sec_kind,
                            "section_title": sec_title,
                            "chunk_index": chunk_index,
                            "subsections": [subsec["marker"]],
                            "is_complete_section": False,
                        })
                        chunk_index += 1
                else:
                    current_text = subsec_content
                    current_markers = [subsec["marker"]]

        # Flush le dernier chunk
        if current_text:
            # Si trop petit, essayer de fusionner avec le précédent
            if len(current_text) < min_chunk_size and chunks:
                last_chunk = chunks[-1]
                combined = last_chunk["text"] + "\n\n" + current_text
                if len(combined) <= max_chunk_size + 200:  # Petite marge
                    last_chunk["text"] = combined
                    last_chunk["subsections"] = last_chunk.get("subsections", []) + current_markers
                else:
                    chunk_text = f"{context_prefix}\n\n{current_text}" if context_prefix else current_text
                    chunks.append({
                        "text": chunk_text.strip(),
                        "section_id": sec_id,
                        "section_kind": sec_kind,
                        "section_title": sec_title,
                        "chunk_index": chunk_index,
                        "subsections": current_markers,
                        "is_complete_section": False,
                    })
            else:
                chunk_text = f"{context_prefix}\n\n{current_text}" if context_prefix else current_text
                chunks.append({
                    "text": chunk_text.strip(),
                    "section_id": sec_id,
                    "section_kind": sec_kind,
                    "section_title": sec_title,
                    "chunk_index": chunk_index,
                    "subsections": current_markers,
                    "is_complete_section": False,
                })

    else:
        # Pas de sous-sections détectées → découpage classique avec contexte
        raw_chunks = simple_chunk(full_text, chunk_size=effective_max, overlap=overlap)

        for i, chunk in enumerate(raw_chunks):
            chunk_text = f"{context_prefix}\n\n{chunk}" if context_prefix else chunk
            chunks.append({
                "text": chunk_text.strip(),
                "section_id": sec_id,
                "section_kind": sec_kind,
                "section_title": sec_title,
                "chunk_index": i,
                "is_complete_section": False,
            })

    return chunks


def chunk_easa_sections(
    sections: List[Dict[str, Any]],
    max_chunk_size: int = 1500,
    min_chunk_size: int = 200,
    merge_small_sections: bool = True,
    add_context_prefix: bool = True,
) -> List[Dict[str, Any]]:
    """
    Chunking intelligent de toutes les sections EASA d'un document.

    Features:
    - Préserve le contexte (préfixe [CS xx.xxx - Title])
    - Fusionne les petites sections adjacentes
    - Découpe intelligemment les grandes sections par sous-sections

    Args:
        sections: Liste de sections EASA (from split_easa_sections)
        max_chunk_size: Taille max d'un chunk
        min_chunk_size: Taille min (fusion si en dessous)
        merge_small_sections: Fusionner les sections trop petites
        add_context_prefix: Ajouter préfixe de contexte

    Returns:
        Liste de chunks avec métadonnées
    """
    if not sections:
        return []

    all_chunks = []
    pending_small_sections = []
    pending_text = ""
    pending_ids = []

    for section in sections:
        full_text = section.get("full_text", "").strip()
        sec_id = section.get("id", "").strip()

        if not full_text:
            continue

        # Si la section est petite et merge activé
        if merge_small_sections and len(full_text) < min_chunk_size:
            context = _build_context_prefix(section)
            section_with_context = f"{context}\n{full_text}" if context else full_text

            # Essayer de fusionner avec les pending
            if len(pending_text) + len(section_with_context) + 4 <= max_chunk_size:
                if pending_text:
                    pending_text += "\n\n---\n\n" + section_with_context
                else:
                    pending_text = section_with_context
                pending_ids.append(sec_id)
                pending_small_sections.append(section)
            else:
                # Flush pending et commencer nouveau groupe
                if pending_text:
                    all_chunks.append({
                        "text": pending_text.strip(),
                        "section_id": " | ".join(pending_ids),
                        "section_kind": pending_small_sections[0].get("kind", "") if pending_small_sections else "",
                        "section_title": "Sections fusionnées",
                        "chunk_index": 0,
                        "merged_sections": pending_ids.copy(),
                        "is_complete_section": True,
                    })
                pending_text = section_with_context
                pending_ids = [sec_id]
                pending_small_sections = [section]
        else:
            # Flush pending avant de traiter une grande section
            if pending_text:
                all_chunks.append({
                    "text": pending_text.strip(),
                    "section_id": " | ".join(pending_ids),
                    "section_kind": pending_small_sections[0].get("kind", "") if pending_small_sections else "",
                    "section_title": "Sections fusionnées",
                    "chunk_index": 0,
                    "merged_sections": pending_ids.copy(),
                    "is_complete_section": True,
                })
                pending_text = ""
                pending_ids = []
                pending_small_sections = []

            # Chunker la section normalement
            section_chunks = smart_chunk_section(
                section,
                max_chunk_size=max_chunk_size,
                min_chunk_size=min_chunk_size,
                add_context_prefix=add_context_prefix,
            )
            all_chunks.extend(section_chunks)

    # Flush final pending
    if pending_text:
        all_chunks.append({
            "text": pending_text.strip(),
            "section_id": " | ".join(pending_ids),
            "section_kind": pending_small_sections[0].get("kind", "") if pending_small_sections else "",
            "section_title": "Sections fusionnées",
            "chunk_index": 0,
            "merged_sections": pending_ids.copy(),
            "is_complete_section": True,
        })

    return all_chunks


# =====================================================================
#  CROSS-REFERENCE DETECTION
# =====================================================================

# Patterns pour détecter les références croisées dans les documents EASA
# Améliorés pour capturer tous les formats: CS-E, CS-APU, AMC1, GM2, CS 25A.xxx, CAT.OP.MPA, etc.
CROSS_REF_PATTERNS = [
    # Références directes complètes:
    # - CS 25.571, CS-E 510, CS-APU 25.1309, CS 25A.631
    # - AMC 25.1309, AMC1 25.631, AMC2 25.1309
    # - GM 25.631, GM1 25.631, GM2 25.1309
    re.compile(
        r'\b(CS(?:-[A-Z]+)?|AMC\d{0,2}|GM\d{0,2})[-\s]?'
        r'(\d+[A-Z]?(?:[.\-]\d+[A-Za-z]?)?(?:\([a-z0-9]+\))*)',
        re.IGNORECASE
    ),
    # Références CAT/ORO/SPA/NCO/NCC/SPO: CAT.OP.MPA.100, ORO.GEN.105, etc.
    re.compile(
        r'\b(CAT|ORO|SPA|NCO|NCC|SPO)\.([A-Z]+(?:\.[A-Z]+)*\.\d+)',
        re.IGNORECASE
    ),
    # Références FCL: FCL.055, FCL.055.A
    re.compile(
        r'\b(FCL)[.\-]?(\d+(?:\.[A-Z]+)?)',
        re.IGNORECASE
    ),
    # Références avec introducteurs: "see CS 25.571", "refer to AMC 25.1309", "compliance with CS 25.613"
    re.compile(
        r'(?:see|refer\s+to|in\s+accordance\s+with|as\s+per|per|according\s+to|compliance\s+with|compliant\s+with)\s+'
        r'(CS(?:-[A-Z]+)?|AMC\d{0,2}|GM\d{0,2})[-\s]?'
        r'(\d+[A-Z]?(?:[.\-]\d+[A-Za-z]?)?)',
        re.IGNORECASE
    ),
    # Références FAR/JAR: "FAR 25.571", "JAR 25.571", "FAR 25-571"
    re.compile(r'\b(FAR|JAR|FAA)[-\s]?(\d+(?:[.\-]\d+)*)', re.IGNORECASE),
    # Références internes: "paragraph (a)", "sub-paragraph (1)", "point (a)(1)"
    re.compile(r'(?:paragraph|sub-paragraph|section|point|alinéa)\s*(\([a-z0-9]+\)(?:\([a-z0-9]+\))?)', re.IGNORECASE),
    # Références Appendix: "Appendix A", "Appendix B to CS 25"
    re.compile(r'(Appendix|APPENDIX)\s+([A-Z0-9]+)', re.IGNORECASE),
]


def extract_cross_references(text: str) -> List[Dict[str, Any]]:
    """
    Extrait toutes les références croisées d'un texte.

    Détecte:
    - Références EASA: CS 25.xxx, AMC, GM
    - Références FAR/JAR
    - Références internes: paragraph (a), section (1)

    Args:
        text: Texte à analyser

    Returns:
        Liste de dicts avec 'ref_type', 'ref_id', 'full_match', 'position'
    """
    if not text:
        return []

    references = []
    seen = set()  # Pour éviter les doublons

    for pattern in CROSS_REF_PATTERNS:
        for match in pattern.finditer(text):
            groups = match.groups()

            if len(groups) >= 2:
                ref_type = groups[0].upper()
                ref_number = groups[1]
                ref_id = f"{ref_type} {ref_number}".strip()
            else:
                ref_type = "internal"
                ref_id = groups[0] if groups else match.group(0)

            # Normaliser l'ID
            ref_id_normalized = ref_id.upper().replace(" ", "").replace("-", "")

            if ref_id_normalized not in seen:
                seen.add(ref_id_normalized)
                references.append({
                    "ref_type": ref_type,
                    "ref_id": ref_id,
                    "ref_id_normalized": ref_id_normalized,
                    "full_match": match.group(0),
                    "position": match.start(),
                })

    return references


def add_cross_references_to_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ajoute les références croisées détectées aux métadonnées d'un chunk.

    Args:
        chunk: Dict avec 'text' et autres métadonnées

    Returns:
        Chunk enrichi avec 'cross_references' et 'references_to'
    """
    text = chunk.get("text", "")
    refs = extract_cross_references(text)

    # Extraire les IDs de référence uniques
    ref_ids = list(set(r["ref_id"] for r in refs))

    # Séparer l'ID de section courant des références externes
    current_section = chunk.get("section_id", "")
    current_normalized = current_section.upper().replace(" ", "").replace("-", "") if current_section else ""

    # Filtrer pour ne garder que les références vers d'autres sections
    external_refs = [
        r for r in refs
        if r["ref_id_normalized"] != current_normalized
    ]

    chunk["cross_references"] = refs
    chunk["references_to"] = [r["ref_id"] for r in external_refs]

    return chunk


def add_cross_references_to_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ajoute les références croisées à une liste de chunks (parallélisé).
    """
    if len(chunks) < 10:
        # Pas de parallélisme pour peu de chunks
        return [add_cross_references_to_chunk(chunk) for chunk in chunks]

    # Paralléliser pour beaucoup de chunks
    max_workers = min(multiprocessing.cpu_count(), len(chunks), 8)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(add_cross_references_to_chunk, chunks))

    return results


def build_reference_index(chunks: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    """
    Construit un index inversé des références: ref_id → indices de chunks.

    Permet de retrouver rapidement quels chunks parlent d'une référence donnée.

    Args:
        chunks: Liste de chunks avec métadonnées

    Returns:
        Dict mapping ref_id → liste d'indices de chunks
    """
    index = {}

    for i, chunk in enumerate(chunks):
        # Indexer par section_id du chunk
        section_id = chunk.get("section_id", "")
        if section_id:
            normalized = section_id.upper().replace(" ", "").replace("-", "")
            if normalized not in index:
                index[normalized] = []
            if i not in index[normalized]:
                index[normalized].append(i)

        # Indexer par références mentionnées
        refs = chunk.get("references_to", [])
        for ref in refs:
            normalized = ref.upper().replace(" ", "").replace("-", "")
            # Note: on n'indexe pas ici car ce sont les chunks qui MENTIONNENT la ref,
            # pas les chunks qui SONT cette ref

    return index


# =====================================================================
#  QUERY-TIME CONTEXT EXPANSION
# =====================================================================

def get_neighboring_chunks(
    chunk_index: int,
    all_chunks: List[Dict[str, Any]],
    window: int = 1,
    same_source_only: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Récupère les chunks voisins (précédent et suivant).

    Args:
        chunk_index: Index du chunk central
        all_chunks: Liste complète des chunks
        window: Nombre de chunks avant/après à récupérer
        same_source_only: Ne prendre que les chunks du même fichier source

    Returns:
        Dict avec 'previous' et 'next' contenant les chunks voisins
    """
    if not all_chunks or chunk_index < 0 or chunk_index >= len(all_chunks):
        return {"previous": [], "next": []}

    current_chunk = all_chunks[chunk_index]
    current_source = current_chunk.get("source_file", "") or current_chunk.get("source", "")

    previous = []
    next_chunks = []

    # Chunks précédents
    for i in range(1, window + 1):
        idx = chunk_index - i
        if idx >= 0:
            neighbor = all_chunks[idx]
            neighbor_source = neighbor.get("source_file", "") or neighbor.get("source", "")

            if same_source_only and neighbor_source != current_source:
                break  # Arrêter si on change de source

            previous.insert(0, neighbor)  # Insérer au début pour garder l'ordre

    # Chunks suivants
    for i in range(1, window + 1):
        idx = chunk_index + i
        if idx < len(all_chunks):
            neighbor = all_chunks[idx]
            neighbor_source = neighbor.get("source_file", "") or neighbor.get("source", "")

            if same_source_only and neighbor_source != current_source:
                break

            next_chunks.append(neighbor)

    return {"previous": previous, "next": next_chunks}


def expand_chunk_context(
    chunk: Dict[str, Any],
    all_chunks: List[Dict[str, Any]],
    chunk_index: int,
    include_neighbors: bool = True,
    include_referenced: bool = True,
    neighbor_window: int = 1,
    reference_index: Optional[Dict[str, List[int]]] = None,
) -> Dict[str, Any]:
    """
    Expanse le contexte d'un chunk avec ses voisins et références.

    Args:
        chunk: Chunk à enrichir
        all_chunks: Liste complète des chunks
        chunk_index: Index du chunk dans la liste
        include_neighbors: Inclure les chunks voisins
        include_referenced: Inclure les chunks référencés
        neighbor_window: Nombre de voisins à inclure
        reference_index: Index des références (optionnel, sera construit si absent)

    Returns:
        Dict avec le chunk original + contexte étendu
    """
    result = {
        "chunk": chunk,
        "neighbors": {"previous": [], "next": []},
        "referenced_chunks": [],
        "expanded_text": chunk.get("text", ""),
    }

    # Ajouter les voisins
    if include_neighbors:
        neighbors = get_neighboring_chunks(
            chunk_index, all_chunks, window=neighbor_window
        )
        result["neighbors"] = neighbors

        # Construire le texte étendu avec contexte
        parts = []

        # Contexte précédent (résumé)
        if neighbors["previous"]:
            prev_texts = [c.get("text", "")[:200] + "..." for c in neighbors["previous"]]
            parts.append(f"[CONTEXTE PRÉCÉDENT]\n{' '.join(prev_texts)}\n")

        # Chunk principal
        parts.append(f"[CONTENU PRINCIPAL]\n{chunk.get('text', '')}\n")

        # Contexte suivant (résumé)
        if neighbors["next"]:
            next_texts = [c.get("text", "")[:200] + "..." for c in neighbors["next"]]
            parts.append(f"[CONTEXTE SUIVANT]\n{' '.join(next_texts)}")

        result["expanded_text"] = "\n".join(parts)

    # Ajouter les chunks référencés
    if include_referenced:
        refs = chunk.get("references_to", [])

        if refs and reference_index is None:
            reference_index = build_reference_index(all_chunks)

        if refs and reference_index:
            for ref_id in refs[:5]:  # Limiter à 5 références
                normalized = ref_id.upper().replace(" ", "").replace("-", "")
                if normalized in reference_index:
                    for idx in reference_index[normalized][:2]:  # Max 2 chunks par ref
                        if idx != chunk_index:  # Ne pas s'inclure soi-même
                            result["referenced_chunks"].append(all_chunks[idx])

    return result


def expand_search_results(
    results: List[Dict[str, Any]],
    all_chunks: List[Dict[str, Any]],
    include_neighbors: bool = True,
    include_referenced: bool = True,
    neighbor_window: int = 1,
) -> List[Dict[str, Any]]:
    """
    Expanse le contexte de tous les résultats de recherche.

    Args:
        results: Résultats de recherche (chunks avec scores)
        all_chunks: Liste complète des chunks indexés
        include_neighbors: Inclure les voisins
        include_referenced: Inclure les références
        neighbor_window: Fenêtre de voisinage

    Returns:
        Résultats enrichis avec contexte
    """
    if not results or not all_chunks:
        return results

    # Construire l'index des références une seule fois
    reference_index = build_reference_index(all_chunks) if include_referenced else None

    # Créer un mapping chunk_id → index pour retrouver rapidement
    chunk_id_to_index = {}
    for i, chunk in enumerate(all_chunks):
        chunk_id = chunk.get("chunk_id", "")
        if chunk_id:
            chunk_id_to_index[chunk_id] = i

    expanded_results = []

    for result in results:
        chunk_id = result.get("chunk_id", "") or result.get("metadata", {}).get("chunk_id", "")

        # Trouver l'index du chunk
        chunk_index = chunk_id_to_index.get(chunk_id, -1)

        if chunk_index >= 0:
            expanded = expand_chunk_context(
                result,
                all_chunks,
                chunk_index,
                include_neighbors=include_neighbors,
                include_referenced=include_referenced,
                neighbor_window=neighbor_window,
                reference_index=reference_index,
            )
            result["context_expansion"] = expanded
            result["expanded_text"] = expanded["expanded_text"]

        expanded_results.append(result)

    return expanded_results


def get_related_chunks_by_reference(
    chunk: Dict[str, Any],
    all_chunks: List[Dict[str, Any]],
    max_related: int = 5,
) -> List[Dict[str, Any]]:
    """
    Trouve les chunks liés par références croisées.

    Si chunk A référence CS 25.571, retourne les chunks qui:
    1. Sont la section CS 25.571
    2. Référencent aussi CS 25.571

    Args:
        chunk: Chunk source
        all_chunks: Tous les chunks
        max_related: Nombre max de chunks liés

    Returns:
        Liste de chunks liés
    """
    refs = chunk.get("references_to", [])
    if not refs:
        return []

    current_id = chunk.get("chunk_id", "")
    related = []
    seen_ids = {current_id}

    # Normaliser les références recherchées
    target_refs = set(r.upper().replace(" ", "").replace("-", "") for r in refs)

    for other_chunk in all_chunks:
        other_id = other_chunk.get("chunk_id", "")

        if other_id in seen_ids:
            continue

        # Vérifier si ce chunk EST une des références
        other_section = other_chunk.get("section_id", "")
        if other_section:
            other_normalized = other_section.upper().replace(" ", "").replace("-", "")
            if other_normalized in target_refs:
                related.append(other_chunk)
                seen_ids.add(other_id)
                continue

        # Vérifier si ce chunk RÉFÉRENCE les mêmes sections
        other_refs = other_chunk.get("references_to", [])
        if other_refs:
            other_normalized_refs = set(r.upper().replace(" ", "").replace("-", "") for r in other_refs)
            if target_refs & other_normalized_refs:  # Intersection non vide
                related.append(other_chunk)
                seen_ids.add(other_id)

        if len(related) >= max_related:
            break

    return related
