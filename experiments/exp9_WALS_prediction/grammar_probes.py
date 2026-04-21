#!/usr/bin/env python3
"""
Grammar probes v2: 10 probes per testable dimension per language.

Dimensions by language:
- EN: sv_num, article, colloc (3 dims = 30 probes)
- FR: sv_num, sv_pers, gender, num_adj, art_gen, colloc (6 dims = 60 probes)
- ES: sv_num, sv_pers, gender, num_adj, art_gen, colloc (6 dims = 60 probes)
- FI: sv_num, sv_pers, num_adj, case, colloc (5 dims = 50 probes)
- RU: sv_num, sv_pers, gender, num_adj, case, colloc (6 dims = 60 probes)
- ID: classifier, colloc (2 dims = 20 probes)
- VI: classifier, colloc (2 dims = 20 probes)
- ZH: classifier, colloc (2 dims = 20 probes)
"""

GRAMMAR_PROBES = {
    "en": [
        # === SV Agreement (number) - 10 probes ===
        {"prompt": "The cat ", "good": ["is", "sits", "sleeps"], "bad": ["are", "sit", "sleep"], "category": "sv_num"},
        {"prompt": "The dog ", "good": ["is", "runs", "barks"], "bad": ["are", "run", "bark"], "category": "sv_num"},
        {"prompt": "The bird ", "good": ["is", "flies", "sings"], "bad": ["are", "fly", "sing"], "category": "sv_num"},
        {"prompt": "The man ", "good": ["is", "walks", "talks"], "bad": ["are", "walk", "talk"], "category": "sv_num"},
        {"prompt": "The woman ", "good": ["is", "walks", "talks"], "bad": ["are", "walk", "talk"], "category": "sv_num"},
        {"prompt": "The cats ", "good": ["are", "sit", "sleep"], "bad": ["is", "sits", "sleeps"], "category": "sv_num"},
        {"prompt": "The dogs ", "good": ["are", "run", "bark"], "bad": ["is", "runs", "barks"], "category": "sv_num"},
        {"prompt": "The birds ", "good": ["are", "fly", "sing"], "bad": ["is", "flies", "sings"], "category": "sv_num"},
        {"prompt": "The men ", "good": ["are", "walk", "talk"], "bad": ["is", "walks", "talks"], "category": "sv_num"},
        {"prompt": "The women ", "good": ["are", "walk", "talk"], "bad": ["is", "walks", "talks"], "category": "sv_num"},
        # === Article (a/an) - 10 probes ===
        {"prompt": "I saw a ", "good": ["cat", "dog", "bird"], "bad": ["apple", "elephant", "owl"], "category": "article"},
        {"prompt": "I saw a ", "good": ["book", "tree", "house"], "bad": ["egg", "orange", "umbrella"], "category": "article"},
        {"prompt": "I saw a ", "good": ["car", "man", "girl"], "bad": ["ant", "eagle", "inch"], "category": "article"},
        {"prompt": "I saw a ", "good": ["table", "chair", "desk"], "bad": ["apple", "ear", "eye"], "category": "article"},
        {"prompt": "I saw a ", "good": ["pen", "cup", "box"], "bad": ["arm", "uncle", "ocean"], "category": "article"},
        {"prompt": "I saw an ", "good": ["apple", "elephant", "owl"], "bad": ["cat", "dog", "bird"], "category": "article"},
        {"prompt": "I saw an ", "good": ["egg", "orange", "umbrella"], "bad": ["book", "tree", "house"], "category": "article"},
        {"prompt": "I saw an ", "good": ["ant", "eagle", "inch"], "bad": ["car", "man", "girl"], "category": "article"},
        {"prompt": "I saw an ", "good": ["arm", "uncle", "ocean"], "bad": ["table", "chair", "desk"], "category": "article"},
        {"prompt": "I saw an ", "good": ["ice", "onion", "hour"], "bad": ["pen", "cup", "box"], "category": "article"},
        # === Collocations - 10 probes ===
        {"prompt": "black and ", "good": ["white"], "bad": ["red", "green", "blue"], "category": "colloc"},
        {"prompt": "day and ", "good": ["night"], "bad": ["morning", "evening", "sun"], "category": "colloc"},
        {"prompt": "up and ", "good": ["down"], "bad": ["left", "right", "over"], "category": "colloc"},
        {"prompt": "hot and ", "good": ["cold"], "bad": ["warm", "cool", "wet"], "category": "colloc"},
        {"prompt": "left and ", "good": ["right"], "bad": ["up", "down", "over"], "category": "colloc"},
        {"prompt": "back and ", "good": ["forth"], "bad": ["front", "side", "over"], "category": "colloc"},
        {"prompt": "bread and ", "good": ["butter"], "bad": ["cheese", "milk", "water"], "category": "colloc"},
        {"prompt": "salt and ", "good": ["pepper"], "bad": ["sugar", "spice", "water"], "category": "colloc"},
        {"prompt": "husband and ", "good": ["wife"], "bad": ["man", "woman", "child"], "category": "colloc"},
        {"prompt": "king and ", "good": ["queen"], "bad": ["prince", "lord", "knight"], "category": "colloc"},
    ],

    "fr": [
        # === SV Agreement (number) - 10 probes ===
        {"prompt": "Le chat ", "good": ["est", "dort", "mange"], "bad": ["sont", "dorment", "mangent"], "category": "sv_num"},
        {"prompt": "Le chien ", "good": ["est", "court", "aboie"], "bad": ["sont", "courent", "aboient"], "category": "sv_num"},
        {"prompt": "L'oiseau ", "good": ["est", "vole", "chante"], "bad": ["sont", "volent", "chantent"], "category": "sv_num"},
        {"prompt": "L'homme ", "good": ["est", "marche", "parle"], "bad": ["sont", "marchent", "parlent"], "category": "sv_num"},
        {"prompt": "La femme ", "good": ["est", "marche", "parle"], "bad": ["sont", "marchent", "parlent"], "category": "sv_num"},
        {"prompt": "Les chats ", "good": ["sont", "dorment", "mangent"], "bad": ["est", "dort", "mange"], "category": "sv_num"},
        {"prompt": "Les chiens ", "good": ["sont", "courent", "aboient"], "bad": ["est", "court", "aboie"], "category": "sv_num"},
        {"prompt": "Les oiseaux ", "good": ["sont", "volent", "chantent"], "bad": ["est", "vole", "chante"], "category": "sv_num"},
        {"prompt": "Les hommes ", "good": ["sont", "marchent", "parlent"], "bad": ["est", "marche", "parle"], "category": "sv_num"},
        {"prompt": "Les femmes ", "good": ["sont", "marchent", "parlent"], "bad": ["est", "marche", "parle"], "category": "sv_num"},
        # === SV Agreement (person) - 10 probes ===
        {"prompt": "Je ", "good": ["suis", "mange", "vais"], "bad": ["es", "manges", "vas"], "category": "sv_pers"},
        {"prompt": "Je ", "good": ["parle", "dors", "lis"], "bad": ["parles", "dors", "lis"], "category": "sv_pers"},
        {"prompt": "Tu ", "good": ["es", "manges", "vas"], "bad": ["suis", "mange", "vais"], "category": "sv_pers"},
        {"prompt": "Tu ", "good": ["parles", "dors", "lis"], "bad": ["parle", "dort", "lit"], "category": "sv_pers"},
        {"prompt": "Il ", "good": ["est", "mange", "va"], "bad": ["suis", "manges", "vais"], "category": "sv_pers"},
        {"prompt": "Elle ", "good": ["est", "mange", "va"], "bad": ["es", "manges", "vas"], "category": "sv_pers"},
        {"prompt": "Nous ", "good": ["sommes", "mangeons", "allons"], "bad": ["sont", "mangent", "vont"], "category": "sv_pers"},
        {"prompt": "Vous ", "good": ["êtes", "mangez", "allez"], "bad": ["sommes", "mangeons", "allons"], "category": "sv_pers"},
        {"prompt": "Ils ", "good": ["sont", "mangent", "vont"], "bad": ["est", "mange", "va"], "category": "sv_pers"},
        {"prompt": "Elles ", "good": ["sont", "mangent", "vont"], "bad": ["est", "mange", "va"], "category": "sv_pers"},
        # === Gender Agreement - 10 probes ===
        {"prompt": "Le chat ", "good": ["noir", "petit", "gros"], "bad": ["noire", "petite", "grosse"], "category": "gender"},
        {"prompt": "Le chien ", "good": ["blanc", "grand", "beau"], "bad": ["blanche", "grande", "belle"], "category": "gender"},
        {"prompt": "Le livre ", "good": ["rouge", "vieux", "nouveau"], "bad": ["vieille", "nouvelle"], "category": "gender"},
        {"prompt": "L'homme ", "good": ["grand", "petit", "vieux"], "bad": ["grande", "petite", "vieille"], "category": "gender"},
        {"prompt": "Le garçon ", "good": ["blond", "content", "fatigué"], "bad": ["blonde", "contente", "fatiguée"], "category": "gender"},
        {"prompt": "La maison ", "good": ["blanche", "grande", "belle"], "bad": ["blanc", "grand", "beau"], "category": "gender"},
        {"prompt": "La voiture ", "good": ["rouge", "petite", "vieille"], "bad": ["petit", "vieux"], "category": "gender"},
        {"prompt": "La femme ", "good": ["grande", "petite", "belle"], "bad": ["grand", "petit", "beau"], "category": "gender"},
        {"prompt": "La fille ", "good": ["blonde", "contente", "fatiguée"], "bad": ["blond", "content", "fatigué"], "category": "gender"},
        {"prompt": "La table ", "good": ["ronde", "carrée", "haute"], "bad": ["rond", "carré", "haut"], "category": "gender"},
        # === Number Agreement (adj) - 10 probes ===
        {"prompt": "Les chats ", "good": ["noirs", "petits", "gros"], "bad": ["noir", "petit"], "category": "num_adj"},
        {"prompt": "Les chiens ", "good": ["blancs", "grands", "beaux"], "bad": ["blanc", "grand", "beau"], "category": "num_adj"},
        {"prompt": "Les livres ", "good": ["rouges", "vieux", "nouveaux"], "bad": ["rouge", "vieux", "nouveau"], "category": "num_adj"},
        {"prompt": "Les hommes ", "good": ["grands", "petits", "vieux"], "bad": ["grand", "petit", "vieux"], "category": "num_adj"},
        {"prompt": "Les garçons ", "good": ["blonds", "contents", "fatigués"], "bad": ["blond", "content", "fatigué"], "category": "num_adj"},
        {"prompt": "Les maisons ", "good": ["blanches", "grandes", "belles"], "bad": ["blanche", "grande", "belle"], "category": "num_adj"},
        {"prompt": "Les voitures ", "good": ["rouges", "petites", "vieilles"], "bad": ["rouge", "petite", "vieille"], "category": "num_adj"},
        {"prompt": "Les femmes ", "good": ["grandes", "petites", "belles"], "bad": ["grande", "petite", "belle"], "category": "num_adj"},
        {"prompt": "Les filles ", "good": ["blondes", "contentes", "fatiguées"], "bad": ["blonde", "contente", "fatiguée"], "category": "num_adj"},
        {"prompt": "Les tables ", "good": ["rondes", "carrées", "hautes"], "bad": ["ronde", "carrée", "haute"], "category": "num_adj"},
        # === Article (gender) - 10 probes ===
        {"prompt": "Je vois le ", "good": ["chat", "chien", "livre"], "bad": ["maison", "voiture", "table"], "category": "art_gen"},
        {"prompt": "Je vois le ", "good": ["garçon", "homme", "père"], "bad": ["fille", "femme", "mère"], "category": "art_gen"},
        {"prompt": "Je vois le ", "good": ["soleil", "ciel", "jour"], "bad": ["lune", "nuit", "étoile"], "category": "art_gen"},
        {"prompt": "Je vois le ", "good": ["pain", "fromage", "vin"], "bad": ["viande", "salade", "bière"], "category": "art_gen"},
        {"prompt": "Je vois le ", "good": ["jardin", "arbre", "mur"], "bad": ["fleur", "plante", "porte"], "category": "art_gen"},
        {"prompt": "Je vois la ", "good": ["maison", "voiture", "table"], "bad": ["chat", "chien", "livre"], "category": "art_gen"},
        {"prompt": "Je vois la ", "good": ["fille", "femme", "mère"], "bad": ["garçon", "homme", "père"], "category": "art_gen"},
        {"prompt": "Je vois la ", "good": ["lune", "nuit", "étoile"], "bad": ["soleil", "ciel", "jour"], "category": "art_gen"},
        {"prompt": "Je vois la ", "good": ["viande", "salade", "bière"], "bad": ["pain", "fromage", "vin"], "category": "art_gen"},
        {"prompt": "Je vois la ", "good": ["fleur", "plante", "porte"], "bad": ["jardin", "arbre", "mur"], "category": "art_gen"},
        # === Collocations - 10 probes ===
        {"prompt": "noir et ", "good": ["blanc"], "bad": ["rouge", "vert", "bleu"], "category": "colloc"},
        {"prompt": "jour et ", "good": ["nuit"], "bad": ["matin", "soir", "soleil"], "category": "colloc"},
        {"prompt": "haut et ", "good": ["bas"], "bad": ["grand", "petit", "large"], "category": "colloc"},
        {"prompt": "chaud et ", "good": ["froid"], "bad": ["tiède", "frais", "sec"], "category": "colloc"},
        {"prompt": "gauche et ", "good": ["droite"], "bad": ["haut", "bas", "avant"], "category": "colloc"},
        {"prompt": "sel et ", "good": ["poivre"], "bad": ["sucre", "épice", "eau"], "category": "colloc"},
        {"prompt": "pain et ", "good": ["beurre"], "bad": ["fromage", "lait", "eau"], "category": "colloc"},
        {"prompt": "mari et ", "good": ["femme"], "bad": ["homme", "fille", "enfant"], "category": "colloc"},
        {"prompt": "roi et ", "good": ["reine"], "bad": ["prince", "duc", "chevalier"], "category": "colloc"},
        {"prompt": "frère et ", "good": ["sœur"], "bad": ["père", "mère", "fils"], "category": "colloc"},
    ],

    "es": [
        # === SV Agreement (number) - 10 probes ===
        {"prompt": "El gato ", "good": ["es", "duerme", "come"], "bad": ["son", "duermen", "comen"], "category": "sv_num"},
        {"prompt": "El perro ", "good": ["es", "corre", "ladra"], "bad": ["son", "corren", "ladran"], "category": "sv_num"},
        {"prompt": "El pájaro ", "good": ["es", "vuela", "canta"], "bad": ["son", "vuelan", "cantan"], "category": "sv_num"},
        {"prompt": "El hombre ", "good": ["es", "camina", "habla"], "bad": ["son", "caminan", "hablan"], "category": "sv_num"},
        {"prompt": "La mujer ", "good": ["es", "camina", "habla"], "bad": ["son", "caminan", "hablan"], "category": "sv_num"},
        {"prompt": "Los gatos ", "good": ["son", "duermen", "comen"], "bad": ["es", "duerme", "come"], "category": "sv_num"},
        {"prompt": "Los perros ", "good": ["son", "corren", "ladran"], "bad": ["es", "corre", "ladra"], "category": "sv_num"},
        {"prompt": "Los pájaros ", "good": ["son", "vuelan", "cantan"], "bad": ["es", "vuela", "canta"], "category": "sv_num"},
        {"prompt": "Los hombres ", "good": ["son", "caminan", "hablan"], "bad": ["es", "camina", "habla"], "category": "sv_num"},
        {"prompt": "Las mujeres ", "good": ["son", "caminan", "hablan"], "bad": ["es", "camina", "habla"], "category": "sv_num"},
        # === SV Agreement (person) - 10 probes ===
        {"prompt": "Yo ", "good": ["soy", "como", "voy"], "bad": ["eres", "comes", "vas"], "category": "sv_pers"},
        {"prompt": "Yo ", "good": ["hablo", "duermo", "leo"], "bad": ["hablas", "duermes", "lees"], "category": "sv_pers"},
        {"prompt": "Tú ", "good": ["eres", "comes", "vas"], "bad": ["soy", "como", "voy"], "category": "sv_pers"},
        {"prompt": "Tú ", "good": ["hablas", "duermes", "lees"], "bad": ["habla", "duerme", "lee"], "category": "sv_pers"},
        {"prompt": "Él ", "good": ["es", "come", "va"], "bad": ["soy", "comes", "voy"], "category": "sv_pers"},
        {"prompt": "Ella ", "good": ["es", "come", "va"], "bad": ["eres", "comes", "vas"], "category": "sv_pers"},
        {"prompt": "Nosotros ", "good": ["somos", "comemos", "vamos"], "bad": ["son", "comen", "van"], "category": "sv_pers"},
        {"prompt": "Vosotros ", "good": ["sois", "coméis", "vais"], "bad": ["somos", "comemos", "vamos"], "category": "sv_pers"},
        {"prompt": "Ellos ", "good": ["son", "comen", "van"], "bad": ["es", "come", "va"], "category": "sv_pers"},
        {"prompt": "Ellas ", "good": ["son", "comen", "van"], "bad": ["es", "come", "va"], "category": "sv_pers"},
        # === Gender Agreement - 10 probes ===
        {"prompt": "El gato ", "good": ["negro", "pequeño", "gordo"], "bad": ["negra", "pequeña", "gorda"], "category": "gender"},
        {"prompt": "El perro ", "good": ["blanco", "grande", "bonito"], "bad": ["blanca", "grande", "bonita"], "category": "gender"},
        {"prompt": "El libro ", "good": ["rojo", "viejo", "nuevo"], "bad": ["roja", "vieja", "nueva"], "category": "gender"},
        {"prompt": "El hombre ", "good": ["alto", "bajo", "viejo"], "bad": ["alta", "baja", "vieja"], "category": "gender"},
        {"prompt": "El niño ", "good": ["rubio", "contento", "cansado"], "bad": ["rubia", "contenta", "cansada"], "category": "gender"},
        {"prompt": "La casa ", "good": ["blanca", "grande", "bonita"], "bad": ["blanco", "grande", "bonito"], "category": "gender"},
        {"prompt": "La mesa ", "good": ["roja", "pequeña", "vieja"], "bad": ["rojo", "pequeño", "viejo"], "category": "gender"},
        {"prompt": "La mujer ", "good": ["alta", "baja", "vieja"], "bad": ["alto", "bajo", "viejo"], "category": "gender"},
        {"prompt": "La niña ", "good": ["rubia", "contenta", "cansada"], "bad": ["rubio", "contento", "cansado"], "category": "gender"},
        {"prompt": "La silla ", "good": ["redonda", "cuadrada", "alta"], "bad": ["redondo", "cuadrado", "alto"], "category": "gender"},
        # === Number Agreement (adj) - 10 probes ===
        {"prompt": "Los gatos ", "good": ["negros", "pequeños", "gordos"], "bad": ["negro", "pequeño", "gordo"], "category": "num_adj"},
        {"prompt": "Los perros ", "good": ["blancos", "grandes", "bonitos"], "bad": ["blanco", "grande", "bonito"], "category": "num_adj"},
        {"prompt": "Los libros ", "good": ["rojos", "viejos", "nuevos"], "bad": ["rojo", "viejo", "nuevo"], "category": "num_adj"},
        {"prompt": "Los hombres ", "good": ["altos", "bajos", "viejos"], "bad": ["alto", "bajo", "viejo"], "category": "num_adj"},
        {"prompt": "Los niños ", "good": ["rubios", "contentos", "cansados"], "bad": ["rubio", "contento", "cansado"], "category": "num_adj"},
        {"prompt": "Las casas ", "good": ["blancas", "grandes", "bonitas"], "bad": ["blanca", "grande", "bonita"], "category": "num_adj"},
        {"prompt": "Las mesas ", "good": ["rojas", "pequeñas", "viejas"], "bad": ["roja", "pequeña", "vieja"], "category": "num_adj"},
        {"prompt": "Las mujeres ", "good": ["altas", "bajas", "viejas"], "bad": ["alta", "baja", "vieja"], "category": "num_adj"},
        {"prompt": "Las niñas ", "good": ["rubias", "contentas", "cansadas"], "bad": ["rubia", "contenta", "cansada"], "category": "num_adj"},
        {"prompt": "Las sillas ", "good": ["redondas", "cuadradas", "altas"], "bad": ["redonda", "cuadrada", "alta"], "category": "num_adj"},
        # === Article (gender) - 10 probes ===
        {"prompt": "Yo veo el ", "good": ["gato", "perro", "libro"], "bad": ["casa", "mesa", "silla"], "category": "art_gen"},
        {"prompt": "Yo veo el ", "good": ["niño", "hombre", "padre"], "bad": ["niña", "mujer", "madre"], "category": "art_gen"},
        {"prompt": "Yo veo el ", "good": ["sol", "cielo", "día"], "bad": ["luna", "noche", "estrella"], "category": "art_gen"},
        {"prompt": "Yo veo el ", "good": ["pan", "queso", "vino"], "bad": ["carne", "leche", "cerveza"], "category": "art_gen"},
        {"prompt": "Yo veo el ", "good": ["jardín", "árbol", "muro"], "bad": ["flor", "planta", "puerta"], "category": "art_gen"},
        {"prompt": "Yo veo la ", "good": ["casa", "mesa", "silla"], "bad": ["gato", "perro", "libro"], "category": "art_gen"},
        {"prompt": "Yo veo la ", "good": ["niña", "mujer", "madre"], "bad": ["niño", "hombre", "padre"], "category": "art_gen"},
        {"prompt": "Yo veo la ", "good": ["luna", "noche", "estrella"], "bad": ["sol", "cielo", "día"], "category": "art_gen"},
        {"prompt": "Yo veo la ", "good": ["carne", "leche", "cerveza"], "bad": ["pan", "queso", "vino"], "category": "art_gen"},
        {"prompt": "Yo veo la ", "good": ["flor", "planta", "puerta"], "bad": ["jardín", "árbol", "muro"], "category": "art_gen"},
        # === Collocations - 10 probes ===
        {"prompt": "blanco y ", "good": ["negro"], "bad": ["rojo", "verde", "azul"], "category": "colloc"},
        {"prompt": "día y ", "good": ["noche"], "bad": ["mañana", "tarde", "sol"], "category": "colloc"},
        {"prompt": "arriba y ", "good": ["abajo"], "bad": ["grande", "pequeño", "largo"], "category": "colloc"},
        {"prompt": "caliente y ", "good": ["frío"], "bad": ["tibio", "fresco", "seco"], "category": "colloc"},
        {"prompt": "izquierda y ", "good": ["derecha"], "bad": ["arriba", "abajo", "delante"], "category": "colloc"},
        {"prompt": "sal y ", "good": ["pimienta"], "bad": ["azúcar", "especia", "agua"], "category": "colloc"},
        {"prompt": "pan y ", "good": ["mantequilla"], "bad": ["queso", "leche", "agua"], "category": "colloc"},
        {"prompt": "marido y ", "good": ["mujer"], "bad": ["hombre", "hija", "niño"], "category": "colloc"},
        {"prompt": "rey y ", "good": ["reina"], "bad": ["príncipe", "duque", "caballero"], "category": "colloc"},
        {"prompt": "hermano y ", "good": ["hermana"], "bad": ["padre", "madre", "hijo"], "category": "colloc"},
    ],

    "fi": [
        # === SV Agreement (number) - 10 probes ===
        {"prompt": "Kissa ", "good": ["on", "nukkuu", "syö"], "bad": ["ovat", "nukkuvat", "syövät"], "category": "sv_num"},
        {"prompt": "Koira ", "good": ["on", "juoksee", "haukkuu"], "bad": ["ovat", "juoksevat", "haukkuvat"], "category": "sv_num"},
        {"prompt": "Lintu ", "good": ["on", "lentää", "laulaa"], "bad": ["ovat", "lentävät", "laulavat"], "category": "sv_num"},
        {"prompt": "Mies ", "good": ["on", "kävelee", "puhuu"], "bad": ["ovat", "kävelevät", "puhuvat"], "category": "sv_num"},
        {"prompt": "Nainen ", "good": ["on", "kävelee", "puhuu"], "bad": ["ovat", "kävelevät", "puhuvat"], "category": "sv_num"},
        {"prompt": "Kissat ", "good": ["ovat", "nukkuvat", "syövät"], "bad": ["on", "nukkuu", "syö"], "category": "sv_num"},
        {"prompt": "Koirat ", "good": ["ovat", "juoksevat", "haukkuvat"], "bad": ["on", "juoksee", "haukkuu"], "category": "sv_num"},
        {"prompt": "Linnut ", "good": ["ovat", "lentävät", "laulavat"], "bad": ["on", "lentää", "laulaa"], "category": "sv_num"},
        {"prompt": "Miehet ", "good": ["ovat", "kävelevät", "puhuvat"], "bad": ["on", "kävelee", "puhuu"], "category": "sv_num"},
        {"prompt": "Naiset ", "good": ["ovat", "kävelevät", "puhuvat"], "bad": ["on", "kävelee", "puhuu"], "category": "sv_num"},
        # === SV Agreement (person) - 10 probes ===
        {"prompt": "Minä ", "good": ["olen", "syön", "menen"], "bad": ["olet", "syöt", "menet"], "category": "sv_pers"},
        {"prompt": "Minä ", "good": ["puhun", "nukun", "luen"], "bad": ["puhut", "nukut", "luet"], "category": "sv_pers"},
        {"prompt": "Sinä ", "good": ["olet", "syöt", "menet"], "bad": ["olen", "syön", "menen"], "category": "sv_pers"},
        {"prompt": "Sinä ", "good": ["puhut", "nukut", "luet"], "bad": ["puhuu", "nukkuu", "lukee"], "category": "sv_pers"},
        {"prompt": "Hän ", "good": ["on", "syö", "menee"], "bad": ["olen", "syöt", "menen"], "category": "sv_pers"},
        {"prompt": "Hän ", "good": ["puhuu", "nukkuu", "lukee"], "bad": ["puhut", "nukut", "luet"], "category": "sv_pers"},
        {"prompt": "Me ", "good": ["olemme", "syömme", "menemme"], "bad": ["ovat", "syövät", "menevät"], "category": "sv_pers"},
        {"prompt": "Te ", "good": ["olette", "syötte", "menette"], "bad": ["olemme", "syömme", "menemme"], "category": "sv_pers"},
        {"prompt": "He ", "good": ["ovat", "syövät", "menevät"], "bad": ["on", "syö", "menee"], "category": "sv_pers"},
        {"prompt": "He ", "good": ["puhuvat", "nukkuvat", "lukevat"], "bad": ["puhuu", "nukkuu", "lukee"], "category": "sv_pers"},
        # === Number Agreement (adj) - 10 probes ===
        {"prompt": "Iso ", "good": ["talo", "koira", "kissa"], "bad": ["talot", "koirat", "kissat"], "category": "num_adj"},
        {"prompt": "Pieni ", "good": ["auto", "lintu", "lapsi"], "bad": ["autot", "linnut", "lapset"], "category": "num_adj"},
        {"prompt": "Vanha ", "good": ["mies", "nainen", "puu"], "bad": ["miehet", "naiset", "puut"], "category": "num_adj"},
        {"prompt": "Uusi ", "good": ["kirja", "koti", "työ"], "bad": ["kirjat", "kodit", "työt"], "category": "num_adj"},
        {"prompt": "Kaunis ", "good": ["kukka", "päivä", "tyttö"], "bad": ["kukat", "päivät", "tytöt"], "category": "num_adj"},
        {"prompt": "Isot ", "good": ["talot", "koirat", "kissat"], "bad": ["talo", "koira", "kissa"], "category": "num_adj"},
        {"prompt": "Pienet ", "good": ["autot", "linnut", "lapset"], "bad": ["auto", "lintu", "lapsi"], "category": "num_adj"},
        {"prompt": "Vanhat ", "good": ["miehet", "naiset", "puut"], "bad": ["mies", "nainen", "puu"], "category": "num_adj"},
        {"prompt": "Uudet ", "good": ["kirjat", "kodit", "työt"], "bad": ["kirja", "koti", "työ"], "category": "num_adj"},
        {"prompt": "Kauniit ", "good": ["kukat", "päivät", "tytöt"], "bad": ["kukka", "päivä", "tyttö"], "category": "num_adj"},
        # === Case Agreement - 10 probes ===
        {"prompt": "Näen ison ", "good": ["talon", "koiran", "kissan"], "bad": ["talo", "koira", "kissa"], "category": "case"},
        {"prompt": "Näen pienen ", "good": ["auton", "linnun", "lapsen"], "bad": ["auto", "lintu", "lapsi"], "category": "case"},
        {"prompt": "Isossa ", "good": ["talossa", "autossa", "huoneessa"], "bad": ["talo", "auto", "huone"], "category": "case"},
        {"prompt": "Pienessä ", "good": ["kodissa", "kaupassa", "koulussa"], "bad": ["koti", "kauppa", "koulu"], "category": "case"},
        {"prompt": "Isoon ", "good": ["taloon", "autoon", "huoneeseen"], "bad": ["talo", "auto", "huone"], "category": "case"},
        {"prompt": "Pieneen ", "good": ["kotiin", "kauppaan", "kouluun"], "bad": ["koti", "kauppa", "koulu"], "category": "case"},
        {"prompt": "Isosta ", "good": ["talosta", "autosta", "huoneesta"], "bad": ["talo", "auto", "huone"], "category": "case"},
        {"prompt": "Pienestä ", "good": ["kodista", "kaupasta", "koulusta"], "bad": ["koti", "kauppa", "koulu"], "category": "case"},
        {"prompt": "Ison talon ", "good": ["edessä", "takana", "vieressä"], "bad": ["edessä", "takana", "vieressä"], "category": "case"},
        {"prompt": "Vanhan miehen ", "good": ["kanssa", "luona", "takia"], "bad": ["kanssa", "luona", "takia"], "category": "case"},
        # === Collocations - 10 probes ===
        {"prompt": "musta ja ", "good": ["valkoinen"], "bad": ["punainen", "sininen", "vihreä"], "category": "colloc"},
        {"prompt": "yö ja ", "good": ["päivä"], "bad": ["aamu", "ilta", "aurinko"], "category": "colloc"},
        {"prompt": "ylös ja ", "good": ["alas"], "bad": ["iso", "pieni", "pitkä"], "category": "colloc"},
        {"prompt": "kuuma ja ", "good": ["kylmä"], "bad": ["lämmin", "viileä", "kuiva"], "category": "colloc"},
        {"prompt": "vasen ja ", "good": ["oikea"], "bad": ["ylös", "alas", "eteen"], "category": "colloc"},
        {"prompt": "suola ja ", "good": ["pippuri"], "bad": ["sokeri", "mauste", "vesi"], "category": "colloc"},
        {"prompt": "leipä ja ", "good": ["voi"], "bad": ["juusto", "maito", "vesi"], "category": "colloc"},
        {"prompt": "mies ja ", "good": ["vaimo"], "bad": ["nainen", "tyttö", "lapsi"], "category": "colloc"},
        {"prompt": "kuningas ja ", "good": ["kuningatar"], "bad": ["prinssi", "herttua", "ritari"], "category": "colloc"},
        {"prompt": "veli ja ", "good": ["sisko"], "bad": ["isä", "äiti", "poika"], "category": "colloc"},
    ],

    "ru": [
        # === SV Agreement (number) - 10 probes ===
        {"prompt": "Кот ", "good": ["спит", "ест", "бежит"], "bad": ["спят", "едят", "бегут"], "category": "sv_num"},
        {"prompt": "Собака ", "good": ["спит", "ест", "бежит"], "bad": ["спят", "едят", "бегут"], "category": "sv_num"},
        {"prompt": "Птица ", "good": ["летит", "поёт", "сидит"], "bad": ["летят", "поют", "сидят"], "category": "sv_num"},
        {"prompt": "Мужчина ", "good": ["идёт", "говорит", "работает"], "bad": ["идут", "говорят", "работают"], "category": "sv_num"},
        {"prompt": "Женщина ", "good": ["идёт", "говорит", "работает"], "bad": ["идут", "говорят", "работают"], "category": "sv_num"},
        {"prompt": "Коты ", "good": ["спят", "едят", "бегут"], "bad": ["спит", "ест", "бежит"], "category": "sv_num"},
        {"prompt": "Собаки ", "good": ["спят", "едят", "бегут"], "bad": ["спит", "ест", "бежит"], "category": "sv_num"},
        {"prompt": "Птицы ", "good": ["летят", "поют", "сидят"], "bad": ["летит", "поёт", "сидит"], "category": "sv_num"},
        {"prompt": "Мужчины ", "good": ["идут", "говорят", "работают"], "bad": ["идёт", "говорит", "работает"], "category": "sv_num"},
        {"prompt": "Женщины ", "good": ["идут", "говорят", "работают"], "bad": ["идёт", "говорит", "работает"], "category": "sv_num"},
        # === SV Agreement (person) - 10 probes ===
        {"prompt": "Я ", "good": ["иду", "ем", "сплю"], "bad": ["идёшь", "ешь", "спишь"], "category": "sv_pers"},
        {"prompt": "Я ", "good": ["говорю", "читаю", "пишу"], "bad": ["говоришь", "читаешь", "пишешь"], "category": "sv_pers"},
        {"prompt": "Ты ", "good": ["идёшь", "ешь", "спишь"], "bad": ["иду", "ем", "сплю"], "category": "sv_pers"},
        {"prompt": "Ты ", "good": ["говоришь", "читаешь", "пишешь"], "bad": ["говорит", "читает", "пишет"], "category": "sv_pers"},
        {"prompt": "Он ", "good": ["идёт", "ест", "спит"], "bad": ["иду", "ешь", "сплю"], "category": "sv_pers"},
        {"prompt": "Она ", "good": ["идёт", "ест", "спит"], "bad": ["идёшь", "ешь", "спишь"], "category": "sv_pers"},
        {"prompt": "Мы ", "good": ["идём", "едим", "спим"], "bad": ["идут", "едят", "спят"], "category": "sv_pers"},
        {"prompt": "Вы ", "good": ["идёте", "едите", "спите"], "bad": ["идём", "едим", "спим"], "category": "sv_pers"},
        {"prompt": "Они ", "good": ["идут", "едят", "спят"], "bad": ["идёт", "ест", "спит"], "category": "sv_pers"},
        {"prompt": "Они ", "good": ["говорят", "читают", "пишут"], "bad": ["говорит", "читает", "пишет"], "category": "sv_pers"},
        # === Gender Agreement - 10 probes ===
        {"prompt": "Большой ", "good": ["дом", "стол", "кот"], "bad": ["машина", "книга", "кошка"], "category": "gender"},
        {"prompt": "Новый ", "good": ["друг", "город", "год"], "bad": ["подруга", "страна", "жизнь"], "category": "gender"},
        {"prompt": "Старый ", "good": ["человек", "мир", "лес"], "bad": ["женщина", "земля", "река"], "category": "gender"},
        {"prompt": "Красивый ", "good": ["мальчик", "цветок", "день"], "bad": ["девочка", "роза", "ночь"], "category": "gender"},
        {"prompt": "Хороший ", "good": ["отец", "брат", "сын"], "bad": ["мать", "сестра", "дочь"], "category": "gender"},
        {"prompt": "Большая ", "good": ["машина", "книга", "кошка"], "bad": ["дом", "стол", "кот"], "category": "gender"},
        {"prompt": "Новая ", "good": ["подруга", "страна", "жизнь"], "bad": ["друг", "город", "год"], "category": "gender"},
        {"prompt": "Старая ", "good": ["женщина", "земля", "река"], "bad": ["человек", "мир", "лес"], "category": "gender"},
        {"prompt": "Красивая ", "good": ["девочка", "роза", "ночь"], "bad": ["мальчик", "цветок", "день"], "category": "gender"},
        {"prompt": "Хорошая ", "good": ["мать", "сестра", "дочь"], "bad": ["отец", "брат", "сын"], "category": "gender"},
        # === Number Agreement (adj) - 10 probes ===
        {"prompt": "Большие ", "good": ["дома", "столы", "коты"], "bad": ["дом", "стол", "кот"], "category": "num_adj"},
        {"prompt": "Новые ", "good": ["друзья", "города", "годы"], "bad": ["друг", "город", "год"], "category": "num_adj"},
        {"prompt": "Старые ", "good": ["люди", "леса", "реки"], "bad": ["человек", "лес", "река"], "category": "num_adj"},
        {"prompt": "Красивые ", "good": ["цветы", "дни", "ночи"], "bad": ["цветок", "день", "ночь"], "category": "num_adj"},
        {"prompt": "Хорошие ", "good": ["отцы", "братья", "сыновья"], "bad": ["отец", "брат", "сын"], "category": "num_adj"},
        {"prompt": "Маленькие ", "good": ["дети", "птицы", "кошки"], "bad": ["ребёнок", "птица", "кошка"], "category": "num_adj"},
        {"prompt": "Белые ", "good": ["облака", "стены", "цветы"], "bad": ["облако", "стена", "цветок"], "category": "num_adj"},
        {"prompt": "Чёрные ", "good": ["коты", "собаки", "птицы"], "bad": ["кот", "собака", "птица"], "category": "num_adj"},
        {"prompt": "Высокие ", "good": ["деревья", "горы", "здания"], "bad": ["дерево", "гора", "здание"], "category": "num_adj"},
        {"prompt": "Длинные ", "good": ["дороги", "реки", "ночи"], "bad": ["дорога", "река", "ночь"], "category": "num_adj"},
        # === Case Agreement - 10 probes ===
        {"prompt": "Вижу большого ", "good": ["кота", "пса", "мальчика"], "bad": ["кот", "пёс", "мальчик"], "category": "case"},
        {"prompt": "Вижу красивую ", "good": ["девушку", "машину", "книгу"], "bad": ["девушка", "машина", "книга"], "category": "case"},
        {"prompt": "В большом ", "good": ["доме", "городе", "лесу"], "bad": ["дом", "город", "лес"], "category": "case"},
        {"prompt": "На высокой ", "good": ["горе", "башне", "крыше"], "bad": ["гора", "башня", "крыша"], "category": "case"},
        {"prompt": "С хорошим ", "good": ["другом", "человеком", "отцом"], "bad": ["друг", "человек", "отец"], "category": "case"},
        {"prompt": "Для новой ", "good": ["работы", "жизни", "книги"], "bad": ["работа", "жизнь", "книга"], "category": "case"},
        {"prompt": "О старом ", "good": ["друге", "городе", "времени"], "bad": ["друг", "город", "время"], "category": "case"},
        {"prompt": "К большому ", "good": ["дому", "озеру", "морю"], "bad": ["дом", "озеро", "море"], "category": "case"},
        {"prompt": "Из маленького ", "good": ["города", "села", "дома"], "bad": ["город", "село", "дом"], "category": "case"},
        {"prompt": "За высоким ", "good": ["забором", "деревом", "домом"], "bad": ["забор", "дерево", "дом"], "category": "case"},
        # === Collocations - 10 probes ===
        {"prompt": "чёрный и ", "good": ["белый"], "bad": ["красный", "синий", "зелёный"], "category": "colloc"},
        {"prompt": "день и ", "good": ["ночь"], "bad": ["утро", "вечер", "солнце"], "category": "colloc"},
        {"prompt": "вверх и ", "good": ["вниз"], "bad": ["большой", "маленький", "длинный"], "category": "colloc"},
        {"prompt": "горячий и ", "good": ["холодный"], "bad": ["тёплый", "прохладный", "сухой"], "category": "colloc"},
        {"prompt": "левый и ", "good": ["правый"], "bad": ["верхний", "нижний", "передний"], "category": "colloc"},
        {"prompt": "соль и ", "good": ["перец"], "bad": ["сахар", "специя", "вода"], "category": "colloc"},
        {"prompt": "хлеб и ", "good": ["масло"], "bad": ["сыр", "молоко", "вода"], "category": "colloc"},
        {"prompt": "муж и ", "good": ["жена"], "bad": ["мужчина", "дочь", "ребёнок"], "category": "colloc"},
        {"prompt": "царь и ", "good": ["царица"], "bad": ["принц", "герцог", "рыцарь"], "category": "colloc"},
        {"prompt": "брат и ", "good": ["сестра"], "bad": ["отец", "мать", "сын"], "category": "colloc"},
    ],

    "id": [
        # === Classifiers - 10 probes ===
        {"prompt": "seekor ", "good": ["kucing", "anjing", "burung"], "bad": ["buku", "meja", "rumah"], "category": "classifier"},
        {"prompt": "seekor ", "good": ["ikan", "kuda", "sapi"], "bad": ["kursi", "pintu", "jendela"], "category": "classifier"},
        {"prompt": "seekor ", "good": ["ayam", "bebek", "kambing"], "bad": ["pensil", "tas", "sepatu"], "category": "classifier"},
        {"prompt": "seekor ", "good": ["gajah", "harimau", "singa"], "bad": ["mobil", "sepeda", "kereta"], "category": "classifier"},
        {"prompt": "seekor ", "good": ["kelinci", "tikus", "ular"], "bad": ["televisi", "komputer", "telepon"], "category": "classifier"},
        {"prompt": "sebuah ", "good": ["buku", "meja", "rumah"], "bad": ["kucing", "anjing", "burung"], "category": "classifier"},
        {"prompt": "sebuah ", "good": ["kursi", "pintu", "jendela"], "bad": ["ikan", "kuda", "sapi"], "category": "classifier"},
        {"prompt": "sebuah ", "good": ["pensil", "tas", "sepatu"], "bad": ["ayam", "bebek", "kambing"], "category": "classifier"},
        {"prompt": "sebuah ", "good": ["mobil", "sepeda", "kereta"], "bad": ["gajah", "harimau", "singa"], "category": "classifier"},
        {"prompt": "sebuah ", "good": ["televisi", "komputer", "telepon"], "bad": ["kelinci", "tikus", "ular"], "category": "classifier"},
        # === Collocations - 10 probes ===
        {"prompt": "hitam dan ", "good": ["putih"], "bad": ["merah", "biru", "hijau"], "category": "colloc"},
        {"prompt": "siang dan ", "good": ["malam"], "bad": ["pagi", "sore", "matahari"], "category": "colloc"},
        {"prompt": "atas dan ", "good": ["bawah"], "bad": ["besar", "kecil", "panjang"], "category": "colloc"},
        {"prompt": "panas dan ", "good": ["dingin"], "bad": ["hangat", "sejuk", "kering"], "category": "colloc"},
        {"prompt": "kiri dan ", "good": ["kanan"], "bad": ["atas", "bawah", "depan"], "category": "colloc"},
        {"prompt": "garam dan ", "good": ["merica"], "bad": ["gula", "bumbu", "air"], "category": "colloc"},
        {"prompt": "roti dan ", "good": ["mentega"], "bad": ["keju", "susu", "air"], "category": "colloc"},
        {"prompt": "suami dan ", "good": ["istri"], "bad": ["pria", "anak", "wanita"], "category": "colloc"},
        {"prompt": "raja dan ", "good": ["ratu"], "bad": ["pangeran", "duke", "ksatria"], "category": "colloc"},
        {"prompt": "kakak dan ", "good": ["adik"], "bad": ["ayah", "ibu", "anak"], "category": "colloc"},
    ],

    "vi": [
        # === Classifiers - 10 probes ===
        {"prompt": "một con ", "good": ["mèo", "chó", "chim"], "bad": ["sách", "bàn", "nhà"], "category": "classifier"},
        {"prompt": "một con ", "good": ["cá", "ngựa", "bò"], "bad": ["ghế", "cửa", "cửa sổ"], "category": "classifier"},
        {"prompt": "một con ", "good": ["gà", "vịt", "dê"], "bad": ["bút", "túi", "giày"], "category": "classifier"},
        {"prompt": "một con ", "good": ["voi", "hổ", "sư tử"], "bad": ["xe", "xe đạp", "tàu"], "category": "classifier"},
        {"prompt": "một con ", "good": ["thỏ", "chuột", "rắn"], "bad": ["tivi", "máy tính", "điện thoại"], "category": "classifier"},
        {"prompt": "một quyển ", "good": ["sách", "vở", "tạp chí"], "bad": ["mèo", "chó", "chim"], "category": "classifier"},
        {"prompt": "một cái ", "good": ["bàn", "ghế", "cửa"], "bad": ["cá", "ngựa", "bò"], "category": "classifier"},
        {"prompt": "một cái ", "good": ["bút", "túi", "giày"], "bad": ["gà", "vịt", "dê"], "category": "classifier"},
        {"prompt": "một chiếc ", "good": ["xe", "xe đạp", "thuyền"], "bad": ["voi", "hổ", "sư tử"], "category": "classifier"},
        {"prompt": "một ngôi ", "good": ["nhà", "chùa", "đền"], "bad": ["thỏ", "chuột", "rắn"], "category": "classifier"},
        # === Collocations - 10 probes ===
        {"prompt": "đen và ", "good": ["trắng"], "bad": ["đỏ", "xanh", "vàng"], "category": "colloc"},
        {"prompt": "ngày và ", "good": ["đêm"], "bad": ["sáng", "chiều", "mặt trời"], "category": "colloc"},
        {"prompt": "trên và ", "good": ["dưới"], "bad": ["to", "nhỏ", "dài"], "category": "colloc"},
        {"prompt": "nóng và ", "good": ["lạnh"], "bad": ["ấm", "mát", "khô"], "category": "colloc"},
        {"prompt": "trái và ", "good": ["phải"], "bad": ["trên", "dưới", "trước"], "category": "colloc"},
        {"prompt": "muối và ", "good": ["tiêu"], "bad": ["đường", "gia vị", "nước"], "category": "colloc"},
        {"prompt": "bánh mì và ", "good": ["bơ"], "bad": ["phô mai", "sữa", "nước"], "category": "colloc"},
        {"prompt": "chồng và ", "good": ["vợ"], "bad": ["đàn ông", "con", "phụ nữ"], "category": "colloc"},
        {"prompt": "vua và ", "good": ["hoàng hậu"], "bad": ["hoàng tử", "công tước", "hiệp sĩ"], "category": "colloc"},
        {"prompt": "anh và ", "good": ["em"], "bad": ["bố", "mẹ", "con"], "category": "colloc"},
    ],

    "zh": [
        # === Classifiers - 10 probes ===
        {"prompt": "一只 ", "good": ["猫", "狗", "鸟"], "bad": ["书", "桌", "房"], "category": "classifier"},
        {"prompt": "一只 ", "good": ["鱼", "鸡", "鸭"], "bad": ["椅", "门", "窗"], "category": "classifier"},
        {"prompt": "一只 ", "good": ["兔", "虎", "象"], "bad": ["笔", "包", "鞋"], "category": "classifier"},
        {"prompt": "一只 ", "good": ["羊", "牛", "马"], "bad": ["车", "船", "机"], "category": "classifier"},
        {"prompt": "一只 ", "good": ["蛇", "鼠", "蛙"], "bad": ["电视", "电脑", "手机"], "category": "classifier"},
        {"prompt": "一本 ", "good": ["书", "杂志", "词典"], "bad": ["猫", "狗", "鸟"], "category": "classifier"},
        {"prompt": "一张 ", "good": ["桌", "椅", "床"], "bad": ["鱼", "鸡", "鸭"], "category": "classifier"},
        {"prompt": "一支 ", "good": ["笔", "枪", "箭"], "bad": ["兔", "虎", "象"], "category": "classifier"},
        {"prompt": "一辆 ", "good": ["车", "自行车", "摩托车"], "bad": ["羊", "牛", "马"], "category": "classifier"},
        {"prompt": "一间 ", "good": ["房", "屋", "店"], "bad": ["蛇", "鼠", "蛙"], "category": "classifier"},
        # === Collocations - 10 probes ===
        {"prompt": "黑与 ", "good": ["白"], "bad": ["红", "蓝", "绿"], "category": "colloc"},
        {"prompt": "日与 ", "good": ["夜"], "bad": ["晨", "午", "阳"], "category": "colloc"},
        {"prompt": "上与 ", "good": ["下"], "bad": ["大", "小", "长"], "category": "colloc"},
        {"prompt": "冷与 ", "good": ["热"], "bad": ["温", "凉", "干"], "category": "colloc"},
        {"prompt": "左与 ", "good": ["右"], "bad": ["上", "下", "前"], "category": "colloc"},
        {"prompt": "盐与 ", "good": ["胡椒"], "bad": ["糖", "料", "水"], "category": "colloc"},
        {"prompt": "面包与 ", "good": ["黄油"], "bad": ["奶酪", "牛奶", "水"], "category": "colloc"},
        {"prompt": "丈夫与 ", "good": ["妻子"], "bad": ["男人", "孩子", "女人"], "category": "colloc"},
        {"prompt": "国王与 ", "good": ["王后"], "bad": ["王子", "公爵", "骑士"], "category": "colloc"},
        {"prompt": "兄与 ", "good": ["弟"], "bad": ["父", "母", "子"], "category": "colloc"},
    ],
}

# Synthetic languages use EN probes (they're English-based)
GRAMMAR_PROBES["synth_a"] = GRAMMAR_PROBES["en"].copy()
GRAMMAR_PROBES["synth_b"] = GRAMMAR_PROBES["en"].copy()
GRAMMAR_PROBES["synth_c"] = GRAMMAR_PROBES["en"].copy()
GRAMMAR_PROBES["synth_d"] = GRAMMAR_PROBES["en"].copy()


def get_probe_summary():
    """Print summary of probes per language."""
    for lang, probes in GRAMMAR_PROBES.items():
        categories = {}
        for p in probes:
            cat = p["category"]
            categories[cat] = categories.get(cat, 0) + 1
        total = len(probes)
        print(f"{lang}: {total} probes - {dict(categories)}")


if __name__ == "__main__":
    get_probe_summary()
