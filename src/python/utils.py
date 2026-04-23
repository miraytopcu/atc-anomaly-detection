import numpy as np

def calculate_heading_delta(current, prev):
    if current is None or prev is None:
        return 0 # eğer elimizde current ya da previous veri yoksa hesap yapamayız, çökmeyi önlemek için return 0
    return (current - prev + 180) % 360 - 180

def calculate_rate(current, prev, dt):
    if current is None or prev is None or dt <= 0:
        return 0
    return (current - prev) / dt # dt: iki data arasındaki saniye farkı