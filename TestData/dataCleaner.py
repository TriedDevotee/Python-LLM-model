text = []
allowedchars = {"a", 1, "b", 1, "c", 1, "d", 1, "e", 1, "f", 1, "g", 1,"h", 1, "i", 1, "j", 1, "k", 1,"l", 1, "m", 1, "n", 1, "o", 1,"p", 1, "q", 1, "r", 1, "s", 1,"t", 1, "u", 1, "v", 1, "w", 1,"x", 1, "y", 1, "z", 1, " ", 1}
with open("StupidData.txt", "r", encoding="utf-8", errors="ignore") as f:
    line = f.readline()
    print("running...")
    while line != "":
        line = line.lower()
        text.append("")

        last_plus = -1

        for i in range(len(line)):
            if line[i] == "+":
                last_plus = i

        for i in range(last_plus+1, len(line)):
            if line[i] in allowedchars:
                text[len(text) - 1] += (line[i])
        line = f.readline()
        print(line)

f = open("StupidDataCleaned.txt", "w")
for i in range(len(text)):
    f.write(text[i])
    f.write(" ")
