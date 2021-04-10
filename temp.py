from datetime import datetime as dt

labels = ["blank", "down", "eight", "five", "four",
          "left", "nine", "off", "ok", "on", "one",
          "right", "seven", "six", "three", "two",
          "up", "zero"]

labels = {0: "blank", 1: "down", 2: "eight", 3: "five", 4: "four",
          5: "left", 6: "nine", 7: "off", 8: "ok", 9: "on", 10: "one",
          11: "right", 12: "seven", 13: "six", 14: "three", 15: "two",
          16: "up", 17: "zero"}


today = dt.now().strftime("%d%b")
print(today)

print("""
█▀ ▀█▀ ▄▀█ █▀█ ▀█▀ █ █▄░█ █▀▀   ▀█▀ █▀█ ▄▀█ █ █▄░█ █ █▄░█ █▀▀
▄█ ░█░ █▀█ █▀▄ ░█░ █ █░▀█ █▄█   ░█░ █▀▄ █▀█ █ █░▀█ █ █░▀█ █▄█\n\n""")
