import pandas as pd
len("message")
str_cod = "IWAW NZFRN JKWBME HZA URMI EYHMXSF HL RPUKVB HV XZTKTUSEI YM MPBFRRO I TIJ CFVF"
# str_cod = ''.join(reversed(str_cod))
letter_count = pd.Series(list(str_cod)).value_counts()

two = ["am", "an",
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

three = ["the",
"and",
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
"man",
"say",
"was",
"let",
"way"]

four = ["that",
"this",
"what",
"when",
"they",
"then",
"have",
"your",
"them",
"will",
"like",
"from",
"were",
"with",
"come",
"more",
"time",
"make",
"some",
"word",
"said",
"look",
"many",
"each",
"long",
"know", 
"want", 
"been", 
"good", 
"much"]

five = []

one = ["i", "a"]

cod_ls = str_cod.lower().split(" ")

characters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r', "s", 't','u','v','w','x','y','z']
d = dict.fromkeys(characters, None)

dictionaries = []
for word in cod_ls: # Por cada palabra de la frase
    if len(word) == 1: # Si la longitud de la palabra es 2
        for common_words in one: # Entonces debemos generar un diccionario con el equivalente de la misma letra en las palabras 
            # más comunes para esa longitud
            d_temp = d.copy()
            abort = 0
            for i, letter in enumerate(word):  # Enumera cada letra de la palabra
                print("-"*10)
                print("Coded Word: " + word)
                print("Common Word: " + common_words)
                if d_temp[letter] == None:
                    d_temp[letter] = common_words[i]
                elif d_temp[letter] == common_words[i]:
                    pass
                else:
                    print("La palabra más común " + common_words + " no coincide en la estructura de la palabra ingresada: " + word)
                    abort = 1
                    pass
            if abort == 0:
                dictionaries.append(d_temp)
                # Probando traducción:
                print("Traduciendo: "+ word)
                print(''.join([d_temp[c] for c in word]))
    if len(word) == 2: # Si la longitud de la palabra es 2
        for common_words in two: # Entonces debemos generar un diccionario con el equivalente de la misma letra en las palabras 
            # más comunes para esa longitud
            d_temp = d.copy()
            abort = 0
            for i, letter in enumerate(word):  # Enumera cada letra de la palabra
                print("-"*10)
                print("Coded Word: " + word)
                print("Common Word: " + common_words)
                if d_temp[letter] == None:
                    d_temp[letter] = common_words[i]
                elif d_temp[letter] == common_words[i]:
                    pass
                else:
                    print("La palabra más común " + common_words + " no coincide en la estructura de la palabra ingresada: " + word)
                    abort = 1
                    pass
            if abort == 0:
                dictionaries.append(d_temp)
                # Probando traducción:
                print("Traduciendo: "+ word)
                print(''.join([d_temp[c] for c in word]))
    if len(word) == 3: # Si la longitud de la palabra es 2
        for common_words in three: # Entonces debemos generar un diccionario con el equivalente de la misma letra en las palabras 
            # más comunes para esa longitud
            d_temp = d.copy()
            abort = 0
            for i, letter in enumerate(word):  # Enumera cada letra de la palabra
                print("-"*10)
                print("Coded Word: " + word)
                print("Common Word: " + common_words)
                if d_temp[letter] == None:
                    d_temp[letter] = common_words[i]
                elif d_temp[letter] == common_words[i]:
                    pass
                else:
                    print("La palabra más común " + common_words + " no coincide en la estructura de la palabra ingresada: " + word)
                    abort = 1
                    pass
            if abort == 0:
                dictionaries.append(d_temp)
                # Probando traducción:
                print("Traduciendo: "+ word)
                print(''.join([d_temp[c] for c in word]))
    if len(word) == 4: # Si la longitud de la palabra es 2
        for common_words in four: # Entonces debemos generar un diccionario con el equivalente de la misma letra en las palabras 
            # más comunes para esa longitud
            d_temp = d.copy()
            abort = 0
            for i, letter in enumerate(word):  # Enumera cada letra de la palabra
                print("-"*10)
                print("Coded Word: " + word)
                print("Common Word: " + common_words)
                if d_temp[letter] == None:
                    d_temp[letter] = common_words[i]
                elif d_temp[letter] == common_words[i]:
                    pass
                else:
                    print("La palabra más común " + common_words + " no coincide en la estructura de la palabra ingresada: " + word)
                    abort = 1
                    pass
            if abort == 0:
                dictionaries.append(d_temp)
                # Probando traducción:
                print("Traduciendo: "+ word)
                print(''.join([d_temp[c] for c in word]))
                


sentences = []
for d_t in dictionaries:
    # print([k for k,v in d_t.items() if v != None])    
    # print([c for c in d_t.values() if c != None])
    uncod_ls = []
    for word in cod_ls:
        new_word = ''
        for i, letter in enumerate(word):
            if d_t[letter]:
                new_word = new_word + d_t[letter]
            else:
                new_word = new_word + letter
        # print("Word: " + word)
        # print("New Word: " + new_word)
        uncod_ls.append(new_word)
    sentences.append(uncod_ls)

df = pd.DataFrame(sentences)
df = df[df[12].isin(["i", "a"])] # Condición para que  la palabra de una sola letra sea "i" o "a"



# len_dicts = []

# for i, d_t1 in enumerate(dictionaries):
#     print("Diccionario: " + str(i))
#     d_t1.update(d)
#     final_dict = d_t1.copy()
#     for j, d_t in enumerate(dictionaries):
#         print("Diccionario: " + str(j))
#         join = 1
#         new_keys = list(d_t.keys())
#         for key in new_keys:
#             if not final_dict[key]:
#                 pass
#             elif final_dict[key] == d_t[key]:
#                 # print(final_dict[key])
#                 # print(d_t[key])
#                 pass
#             else:
#                 print(final_dict[key])
#                 print(d_t[key])
#                 join = 0
#         # print(join)
#         if join == 1:
#             final_dict.update(d_t)
#         tot_len = len([v for v in final_dict.values() if v != None])
#         len_dicts.append(tot_len)

# df = pd.DataFrame(sentences)


