two = ["am",
"an",
"as",
"at",
"be",
"by",
"do",
"go",
"he",
"if",
"in",
"is",
"it",
"me",
"my",
"no",
"of",
"on",
"or",
"so",
"to",
"up",
"us",
"we"]

three_a = ["the",
"and"
"for",
"are",
"but",
"not",
"you",
"all",
"any",
"can",
"had",
"her",
"was",
"one",
"our",
"out",
"day",
"get",
"has",
"him",
"his",
"how",
"man",
"new",
"now",
"old",
"see",
"two",
"way",
"who",
"boy",
"did",
"its",
"let",
"put",
"say",
"she",
"too",
"use"]

three_c = ["can",
"day",
"did",
"had",
"has",
"him",
"his",
"man",
"say",
"was",
"way"]
one = ["i", "a"]

sentences = []
for o in one:
    enc = {"i": "i", "w": "w", "a": "a", "n": "n", "z": "z", "f": "f", "r": "r", "j": "j", "k": "k", "b": "b", "m": "m", "e": "e", "h": "h",
           "u": "u", "y": "y", "x": "x", "s": "s",
           "l": "l", "p": "p", "v": "v", "t": "t", "o": "o", "c": "c"}
    enc["i"] = o
    for t in two:
        if o not in t:
            enc["h"] = t[0]
            enc["l"] = t[1]
            for t2 in two:
                if enc["h"] == t2[0] and t != t2:
                    enc["v"] = t[1]
                    for t3 in two:
                        if enc["h"] not in t3 and enc["l"] not in t3 and enc["v"] not in t3:
                            enc["y"] = t3[0]
                            enc["m"] = t3[1]
                            for three in three_a:
                                print(three)
                                if enc["h"] == three[0]:
                                    enc["z"] == three[1]
                                    enc["a"] == three[2]
                                    for three2 in three_a:
                                        if three2[1] == enc["i"]:
                                            enc["t"] = three2[0]
                                            enc["j"] = three2[2]
                                            sentence = []
                                            w = {}
                                            w[1] = enc["i"]+enc["w"]+enc["a"]+enc["w"]
                                            w[2] = enc["n"]+enc["z"]+enc["f"]+enc["r"]+enc["n"]
                                            w[3] = enc["j"]+enc["k"]+enc["w"]+enc["b"]+enc["m"]+enc["e"]
                                            w[4] = enc["h"]+enc["z"]+enc["a"]
                                            w[5] = enc["u"]+enc["r"]+enc["m"]+enc["i"]
                                            w[6] = enc["e"]+enc["y"]+enc["h"]+enc["m"]+enc["x"]+enc["s"]+enc["f"]
                                            w[7] = enc["h"]+enc["l"]
                                            w[8] = enc["r"]+enc["p"]+enc["u"]+enc["k"]+enc["v"]+enc["b"]
                                            w[9] = enc["h"]+enc["v"]
                                            w[10] = enc["x"]+enc["z"]+enc["t"]+enc["k"]+enc["t"]+enc["u"]+enc["s"]+enc["e"]+enc["i"]
                                            w[11] = enc["y"]+enc["m"]
                                            w[12] = enc["m"]+enc["p"]+enc["b"]+enc["f"]+enc["r"]+enc["r"]+enc["o"]
                                            w[13] = enc["i"]
                                            w[14] = enc["t"]+enc["i"]+enc["j"]
                                            w[15] = enc["c"]+enc["f"]+enc["v"]+enc["f"]
                                            for word in w.keys():
                                                sentence.append(w[word])
                                            sentences.append(sentence)
                    
                    