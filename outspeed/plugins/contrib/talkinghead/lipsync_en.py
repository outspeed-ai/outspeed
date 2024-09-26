import unicodedata

import regex  # Note: We use the 'regex' module, not the built-in 're'


class LipsyncEn:
    def __init__(self):
        # English words to Oculus visemes, algorithmic rules adapted from:
        #   NRL Report 7948, "Automatic Translation of English Text to Phonetics by Means of Letter-to-Sound Rules" (1976)
        #   by HONEY SUE EL.OVITZ, RODNEY W. JOHNSON, ASTRID McHUGH, AND JOHN E. SHORE
        #   Available at: https://apps.dtic.mil/sti/pdfs/ADA021929.pdf
        self.rules = {
            "A": [
                "[A] =aa",
                " [ARE] =aa RR",
                " [AR]O=aa RR",
                "[AR]#=E RR",
                " ^[AS]#=E SS",
                "[A]WA=aa",
                "[AW]=aa",
                " :[ANY]=E nn I",
                "[A]^+#=E",
                "#:[ALLY]=aa nn I",
                " [AL]#=aa nn",
                "[AGAIN]=aa kk E nn",
                "#:[AG]E=I kk",
                "[A]^+:#=aa",
                ":[A]^+ =E",
                "[A]^%=E",
                " [ARR]=aa RR",
                "[ARR]=aa RR",
                " :[AR] =aa RR",
                "[AR] =E",
                "[AR]=aa RR",
                "[AIR]=E RR",
                "[AI]=E",
                "[AY]=E",
                "[AU]=aa",
                "#:[AL] =aa nn",
                "#:[ALS] =aa nn SS",
                "[ALK]=aa kk",
                "[AL]^=aa nn",
                " :[ABLE]=E PP aa nn",
                "[ABLE]=aa PP aa nn",
                "[ANG]+=E nn kk",
                "[A]=aa",
            ],
            "B": [
                " [BE]^#=PP I",
                "[BEING]=PP I I nn",
                " [BOTH] =PP O TH",
                " [BUS]#=PP I SS",
                "[BUIL]=PP I nn",
                "[B]=PP",
            ],
            "C": [
                " [CH]^=kk",
                "^E[CH]=kk",
                "[CH]=CH",
                " S[CI]#=SS aa",
                "[CI]A=SS",
                "[CI]O=SS",
                "[CI]EN=SS",
                "[C]+=SS",
                "[CK]=kk",
                "[COM]%=kk aa PP",
                "[C]=kk",
            ],
            "D": [
                "#:[DED] =DD I DD",
                ".E[D] =DD",
                "#^:E[D] =DD",
                " [DE]^#=DD I",
                " [DO] =DD U",
                " [DOES]=DD aa SS",
                " [DOING]=DD U I nn",
                " [DOW]=DD aa",
                "[DU]A=kk U",
                "[D]=DD",
            ],
            "E": [
                "#:[E] =",
                "'^:[E] =",
                " :[E] =I",
                "#[ED] =DD",
                "#:[E]D =",
                "[EV]ER=E FF",
                "[E]^%=I",
                "[ERI]#=I RR I",
                "[ERI]=E RR I",
                "#:[ER]#=E",
                "[ER]#=E RR",
                "[ER]=E",
                " [EVEN]=I FF E nn",
                "#:[E]W=",
                "@[EW]=U",
                "[EW]=I U",
                "[E]O=I",
                "#:&[ES] =I SS",
                "#:[E]S =",
                "#:[ELY] =nn I",
                "#:[EMENT]=PP E nn DD",
                "[EFUL]=FF U nn",
                "[EE]=I",
                "[EARN]=E nn",
                " [EAR]^=E",
                "[EAD]=E DD",
                "#:[EA] =I aa",
                "[EA]SU=E",
                "[EA]=I",
                "[EIGH]=E",
                "[EI]=I",
                " [EYE]=aa",
                "[EY]=I",
                "[EU]=I U",
                "[E]=E",
            ],
            "F": ["[FUL]=FF U nn", "[F]=FF"],
            "G": [
                "[GIV]=kk I FF",
                " [G]I^=kk",
                "[GE]T=kk E",
                "SU[GGES]=kk kk E SS",
                "[GG]=kk",
                " B#[G]=kk",
                "[G]+=kk",
                "[GREAT]=kk RR E DD",
                "#[GH]=",
                "[G]=kk",
            ],
            "H": [" [HAV]=I aa FF", " [HERE]=I I RR", " [HOUR]=aa EE", "[HOW]=I aa", "[H]#=I", "[H]="],
            "I": [
                " [IN]=I nn",
                " [I] =aa",
                "[IN]D=aa nn",
                "[IER]=I E",
                "#:R[IED] =I DD",
                "[IED] =aa DD",
                "[IEN]=I E nn",
                "[IE]T=aa E",
                " :[I]%=aa",
                "[I]%=I",
                "[IE]=I",
                "[I]^+:#=I",
                "[IR]#=aa RR",
                "[IZ]%=aa SS",
                "[IS]%=aa SS",
                "[I]D%=aa",
                "+^[I]^+=I",
                "[I]T%=aa",
                "#^:[I]^+=I",
                "[I]^+=aa",
                "[IR]=E",
                "[IGH]=aa",
                "[ILD]=aa nn DD",
                "[IGN] =aa nn",
                "[IGN]^=aa nn",
                "[IGN]%=aa nn",
                "[IQUE]=I kk",
                "[I]=I",
            ],
            "J": ["[J]=kk"],
            "K": [" [K]N=", "[K]=kk"],
            "L": ["[LO]C#=nn O", "L[L]=", "#^:[L]%=aa nn", "[LEAD]=nn I DD", "[L]=nn"],
            "M": ["[MOV]=PP U FF", "[M]=PP"],
            "N": [
                "E[NG]+=nn kk",
                "[NG]R=nn kk",
                "[NG]#=nn kk",
                "[NGL]%=nn kk aa nn",
                "[NG]=nn",
                "[NK]=nn kk",
                " [NOW] =nn aa",
                "[N]=nn",
            ],
            "O": [
                "[OF] =aa FF",
                "[OROUGH]=E O",
                "#:[OR] =E",
                "#:[ORS] =E SS",
                "[OR]=aa RR",
                " [ONE]=FF aa nn",
                "[OW]=O",
                " [OVER]=O FF E",
                "[OV]=aa FF",
                "[O]^%=O",
                "[O]^EN=O",
                "[O]^I#=O",
                "[OL]D=O nn",
                "[OUGHT]=aa DD",
                "[OUGH]=aa FF",
                " [OU]=aa",
                "H[OU]S#=aa",
                "[OUS]=aa SS",
                "[OUR]=aa RR",
                "[OULD]=U DD",
                "^[OU]^L=aa",
                "[OUP]=U OO",
                "[OU]=aa",
                "[OY]=O",
                "[OING]=O I nn",
                "[OI]=O",
                "[OOR]=aa RR",
                "[OOK]=U kk",
                "[OOD]=U DD",
                "[OO]=U",
                "[O]E=O",
                "[O] =O",
                "[OA]=O",
                " [ONLY]=O nn nn I",
                " [ONCE]=FF aa nn SS",
                "[ON'T]=O nn DD",
                "C[O]N=aa",
                "[O]NG=aa",
                " ^:[O]N=aa",
                "I[ON]=aa nn",
                "#:[ON] =aa nn",
                "#^[ON]=aa nn",
                "[O]ST =O",
                "[OF]^=aa FF",
                "[OTHER]=aa TH E",
                "[OSS] =aa SS",
                "#^:[OM]=aa PP",
                "[O]=aa",
            ],
            "P": ["[PH]=FF", "[PEOP]=PP I PP", "[POW]=PP aa", "[PUT] =PP U DD", "[P]=PP"],
            "Q": ["[QUAR]=kk FF aa RR", "[QU]=kk FF", "[Q]=kk"],
            "R": [" [RE]^#=RR I", "[R]=RR"],
            "S": [
                "[SH]=SS",
                "#[SION]=SS aa nn",
                "[SOME]=SS aa PP",
                "#[SUR]#=SS E",
                "[SUR]#=SS E",
                "#[SU]#=SS U",
                "#[SSU]#=SS U",
                "#[SED] =SS DD",
                "#[S]#=SS",
                "[SAID]=SS E DD",
                "^[SION]=SS aa nn",
                "[S]S=",
                ".[S] =SS",
                "#:.E[S] =SS",
                "#^:##[S] =SS",
                "#^:#[S] =SS",
                "U[S] =SS",
                " :#[S] =SS",
                " [SCH]=SS kk",
                "[S]C+=",
                "#[SM]=SS PP",
                "#[SN]'=SS aa nn",
                "[S]=SS",
            ],
            "T": [
                " [THE] =TH aa",
                "[TO] =DD U",
                "[THAT] =TH aa DD",
                " [THIS] =TH I SS",
                " [THEY]=TH E",
                " [THERE]=TH E RR",
                "[THER]=TH E",
                "[THEIR]=TH E RR",
                " [THAN] =TH aa nn",
                " [THEM] =TH E PP",
                "[THESE] =TH I SS",
                " [THEN]=TH E nn",
                "[THROUGH]=TH RR U",
                "[THOSE]=TH O SS",
                "[THOUGH] =TH O",
                " [THUS]=TH aa SS",
                "[TH]=TH",
                "#:[TED] =DD I DD",
                "S[TI]#N=CH",
                "[TI]O=SS",
                "[TI]A=SS",
                "[TIEN]=SS aa nn",
                "[TUR]#=CH E",
                "[TU]A=CH U",
                " [TWO]=DD U",
                "[T]=DD",
            ],
            "U": [
                " [UN]I=I U nn",
                " [UN]=aa nn",
                " [UPON]=aa PP aa nn",
                "@[UR]#=U RR",
                "[UR]#=I U RR",
                "[UR]=E",
                "[U]^ =aa",
                "[U]^^=aa",
                "[UY]=aa",
                " G[U]#=",
                "G[U]%=",
                "G[U]#=FF",
                "#N[U]=I U",
                "@[U]=I",
                "[U]=I U",
            ],
            "V": ["[VIEW]=FF I U", "[V]=FF"],
            "W": [
                " [WERE]=FF E",
                "[WA]S=FF aa",
                "[WA]T=FF aa",
                "[WHERE]=FF E RR",
                "[WHAT]=FF aa DD",
                "[WHOL]=I O nn",
                "[WHO]=I U",
                "[WH]=FF",
                "[WAR]=FF aa RR",
                "[WOR]^=FF E",
                "[WR]=RR",
                "[W]=FF",
            ],
            "X": [" [X]=SS", "[X]=kk SS"],
            "Y": [
                "[YOUNG]=I aa nn",
                " [YOU]=I U",
                " [YES]=I E SS",
                " [Y]=I",
                "#^:[Y] =I",
                "#^:[Y]I=I",
                " :[Y] =aa",
                " :[Y]#=aa",
                " :[Y]^+:#=I",
                " :[Y]^#=I",
                "[Y]=I",
            ],
            "Z": ["[Z]=SS"],
        }

        self.ops = {
            "#": "[AEIOUY]+",  # One or more vowels AEIOUY
            ".": "[BDVGJLMNRWZ]",  # One voiced consonant BDVGJLMNRWZ
            "%": "(?:ER|E|ES|ED|ING|ELY)",  # One of ER, E, ES, ED, ING, ELY
            "&": "(?:[SCGZXJ]|CH|SH)",  # One of S, C, G, Z, X, J, CH, SH
            "@": "(?:[TSRDLZNJ]|TH|CH|SH)",  # One of T, S, R, D, L, Z, N, J, TH, CH, SH
            "^": "[BCDFGHJKLMNPQRSTVWXZ]",  # One consonant BCDFGHJKLMNPQRSTVWXZ
            "+": "[EIY]",  # One of E, I, Y
            ":": "[BCDFGHJKLMNPQRSTVWXZ]*",  # Zero or more consonants BCDFGHJKLMNPQRSTVWXZ
            " ": r"\b",  # Start/end of the word
        }

        # Convert rules to regex
        for key in self.rules:
            new_rules = []
            for rule in self.rules[key]:
                posL = rule.index("[")
                posR = rule.index("]")
                posE = rule.index("=")
                strLeft = rule[:posL]
                strLetters = rule[posL + 1 : posR]
                strRight = rule[posR + 1 : posE]
                strVisemes = rule[posE + 1 :]

                o = {"regex": "", "move": 0, "visemes": []}

                exp = ""
                exp += "".join([self.ops.get(x, x) for x in strLeft])
                ctxLetters = list(strLetters)
                ctxLetters[0] = ctxLetters[0].lower()
                exp += "".join(ctxLetters)
                o["move"] = len(ctxLetters)
                exp += "".join([self.ops.get(x, x) for x in strRight])
                o["regex"] = regex.compile(exp)

                if len(strVisemes):
                    for viseme in strVisemes.split(" "):
                        o["visemes"].append(viseme)

                new_rules.append(o)
            self.rules[key] = new_rules

        # Viseme durations in relative unit (1=average)
        self.visemeDurations = {
            "aa": 0.95,
            "E": 0.90,
            "I": 0.92,
            "O": 0.96,
            "U": 0.95,
            "PP": 1.08,
            "SS": 1.23,
            "TH": 1,
            "DD": 1.05,
            "FF": 1.00,
            "kk": 1.21,
            "nn": 0.88,
            "RR": 0.88,
            "sil": 1,
        }

        # Pauses in relative units (1=average)
        self.specialDurations = {" ": 1, ",": 3, "-": 0.5, "'": 0.5}

        # English number words
        self.digits = ["oh", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        self.ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        self.tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
        self.teens = [
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
        ]

        # Symbols to English
        self.symbols = {"%": "percent", "€": "euros", "&": "and", "+": "plus", "$": "dollars"}
        self.symbolsReg = regex.compile(r"[%€&\+\$]")

    def convert_digit_by_digit(self, num):
        num = list(str(num))
        numWords = ""
        for m in num:
            numWords += self.digits[int(m)] + " "
        numWords = numWords.rstrip()  # Remove trailing space
        return numWords

    def convert_sets_of_two(self, num):
        firstNumHalf = str(num)[:2]
        secondNumHalf = str(num)[2:4]
        numWords = self.convert_tens(int(firstNumHalf))
        numWords += " " + self.convert_tens(int(secondNumHalf))
        return numWords

    def convert_millions(self, num):
        if num >= 1000000:
            return self.convert_millions(num // 1000000) + " million " + self.convert_thousands(num % 1000000)
        else:
            return self.convert_thousands(num)

    def convert_thousands(self, num):
        if num >= 1000:
            return self.convert_hundreds(num // 1000) + " thousand " + self.convert_hundreds(num % 1000)
        else:
            return self.convert_hundreds(num)

    def convert_hundreds(self, num):
        if num > 99:
            return self.ones[num // 100] + " hundred " + self.convert_tens(num % 100)
        else:
            return self.convert_tens(num)

    def convert_tens(self, num):
        if num < 10:
            return self.ones[num]
        elif 10 <= num < 20:
            return self.teens[num - 10]
        else:
            return self.tens[num // 10] + " " + self.ones[num % 10]

    def convertNumberToWords(self, num):
        num = int(num)
        if num == 0:
            return "zero"
        elif (99 < num < 1000) or (10000 < num < 1000000):
            return self.convert_digit_by_digit(num)
        elif (1000 < num < 2000) or (2009 < num < 3000):
            return self.convert_sets_of_two(num)
        else:
            return self.convert_millions(num)

    def preProcessText(self, s):
        s = regex.sub(r'[#_*":;]', "", s)
        s = self.symbolsReg.sub(lambda m: " " + self.symbols[m.group()] + " ", s)
        s = regex.sub(r"(\d),(\d)", r"\1 point \2", s)  # Number separator
        s = regex.sub(r"\d+", lambda m: self.convertNumberToWords(m.group()), s)  # Numbers to words
        s = regex.sub(r"(\D)\1\1+", r"\1\1", s)  # max 2 repeating chars
        s = regex.sub("  +", " ", s)  # Only one repeating space

        # Remove non-English diacritics
        s = unicodedata.normalize("NFD", s)
        s = "".join([c for c in s if not unicodedata.combining(c)])
        s = unicodedata.normalize("NFC", s)

        s = s.strip()
        return s

    def wordsToVisemes(self, w):
        o = {"words": w.upper(), "visemes": [], "times": [], "durations": [], "i": 0}
        t = 0

        chars = list(o["words"])
        while o["i"] < len(chars):
            c = chars[o["i"]]
            ruleset = self.rules.get(c)
            if ruleset:
                matched_rule = False
                for rule in ruleset:
                    test = o["words"][: o["i"]] + c.lower() + o["words"][o["i"] + 1 :]
                    matches = rule["regex"].search(test)
                    if matches:
                        matched_rule = True
                        for viseme in rule["visemes"]:
                            if o["visemes"] and o["visemes"][-1] == viseme:
                                d = 0.7 * self.visemeDurations.get(viseme, 1)
                                o["durations"][-1] += d
                                t += d
                            else:
                                d = self.visemeDurations.get(viseme, 1)
                                o["visemes"].append(viseme)
                                o["times"].append(t)
                                o["durations"].append(d)
                                t += d
                        o["i"] += rule["move"]
                        break
                if not matched_rule:
                    o["i"] += 1
                    t += self.specialDurations.get(c, 0)
            else:
                o["i"] += 1
                t += self.specialDurations.get(c, 0)
        return o
