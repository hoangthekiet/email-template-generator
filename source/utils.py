import colorsys
import numpy as np


def resolve_url(url: str) -> str:
    """
    Ensures that the given URL starts with 'http' or 'https'.
    """
    if url.startswith('http'):
        return url
    return 'https://' + url

def my_literal_eval_hsl(hsl: str) -> tuple[int, float, float]:
    """
    Parses a string representing an HSL color value and converts it into a tuple.

    Parameters
    ----------
    hsl: A string in the format "H,S%,L%"

    Returns
    -------
    A tuple containing three HSL values
    """
    h, s, l = tuple(hsl.split(','))
    h = int(h)
    s = float(s.rstrip('%'))
    l = float(l.rstrip('%'))
    return h, s, l

def hex_to_rgb(h: str) -> tuple:
    return tuple(int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb: tuple) -> str:
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

def rgb_to_hsl(rgb: tuple) -> str:
    h, l, s = colorsys.rgb_to_hls(rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0)
    return  round(h * 360), round(s * 100, 1), round(l * 100, 1)

def hsl_to_hex(hsl):
    h, s, l = hsl
    s /= 100
    l /= 100

    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = l - c / 2

    if 0 <= h < 60:
        r, g, b = c, x, 0
    elif 60 <= h < 120:
        r, g, b = x, c, 0
    elif 120 <= h < 180:
        r, g, b = 0, c, x
    elif 180 <= h < 240:
        r, g, b = 0, x, c
    elif 240 <= h < 300:
        r, g, b = x, 0, c
    elif 300 <= h < 360:
        r, g, b = c, 0, x

    r = round((r + m) * 255)
    g = round((g + m) * 255)
    b = round((b + m) * 255)

    return f'#{r:02x}{g:02x}{b:02x}'

def hex_to_hsl(h: str) -> tuple:
    rgb = hex_to_rgb(h)
    return rgb_to_hsl(rgb)

def hex_to_array(hex: str) -> np.ndarray:
    h, s, l = hex_to_hsl(hex)
    return np.array([float(h) / 360.0, s / 100.0, l / 100.0])

def hex_to_visual(hex: str) -> str:
    """
    Returns a URL pointing to a 100x100 pixel image filled with the given color.
    """
    return f"https://placehold.co/100x100/{hex.lstrip('#')}/{hex.lstrip('#')}.png"