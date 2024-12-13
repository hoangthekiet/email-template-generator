import ast
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import torch
from bs4 import BeautifulSoup
from tqdm import tqdm

from .model import ColorModel
from .utils import resolve_url, my_literal_eval_hsl, hsl_to_hex, rgb_to_hex, rgb_to_hsl, hex_to_rgb, hex_to_array


class StyleExtractor:
    """
    A class representing a hybrid method for extracting the style of a website.
    """

    MODEL_DIR = "checkpoints/cp_485_1212_1.pt"

    def __init__(self, url: str):
        """
        Parameters
        ----------
        url: The URL of the website to extract styles from.
        """
        self.url = resolve_url(url)
        response = requests.get(self.url)
        self.soup = BeautifulSoup(response.content, 'html.parser')
        self.css_files = [link['href'] for link in self.soup.find_all('link', rel='stylesheet')]
        self.model = None


    def get_css_content(self, css_file: str) -> str:
        # Standardize CSS url
        """
        Standardize a CSS file URL and retrieve its content.

        Parameters
        ----------
        css_file: CSS file URL
        
        Returns
        -------
        CSS content
        """
        if css_file.startswith('//'):
            css_file = 'https:' + css_file
        elif not css_file.startswith('http'):
            css_file = self.url + css_file
        # Get content
        try:
            css_content = requests.get(css_file).text
        except:
            print('Error retrieving CSS', css_file)
            css_content = ""
        return css_content


    def get_colors_showit(self) -> Dict[str, str]:
        """
        Get color profile from website built on ShowIt.
        e.g. https://www.hannahschneiderwellness.com/
        
        Returns
        -------
        A dictionary with keys representing color names,
            each associated with their respective hex color values.
            (Returns an empty dictionary if colors cannot be extracted.)
        """
        data = re.findall(r'\s\{([a-zA-Z\-]*color):rgba(\([\d,]+\))', self.soup.prettify())
        if not data:
            return {}

        df = pd.DataFrame(data, columns=['name', 'rgba'])
        df['rgba'] = df.rgba.apply(ast.literal_eval)
        df = df[df.rgba.apply(lambda x: x[-1] == 1)] # only keep opaque colors
        df['hsl'] = df.rgba.apply(rgb_to_hsl)
        df['hex'] = df.rgba.apply(rgb_to_hex)
        df = df.groupby(list(df.columns)).size().reset_index(name='counts').sort_values(by='counts', ascending=False, ignore_index=True)

        try:
            canvas_color = df[((df.name=='background-color') & (df.hsl.apply(lambda x: x[1] < 25 and x[2] > 50)))].iloc[0].hex
            backdrop_color = df[((df.name=='background-color') & (df.hex!=canvas_color) & (df.hsl.apply(lambda x: x[1] < 50 and x[2] > 75)))].iloc[0].hex
            font_color = df[((df.name!='background-color') & (df.hsl.apply(lambda x: x[1] < 25 and x[2] < 25)))].iloc[0].hex
            link_color = df[((df.name!='background-color') & (df.hex!=font_color) & (df.hsl.apply(lambda x: x[1] < 50 and x[2] < 50)))].iloc[0].hex
            accent_color = df[df.hsl.apply(lambda x: x[1] > 25 and 25 < x[2] < 75)].iloc[0].hex
        except:
            return {}

        return {'backdrop_color': backdrop_color,
                'canvas_color': canvas_color,
                'font_color': font_color,
                'link_color': link_color,
                'accent_color': accent_color}


    def get_colors_squarespace(self) -> Dict[str, str]:
        """
        Get color profile from website built on SquareSpace.
        e.g. https://www.hannahschneiderwellness.com/

        Returns
        -------
        A dictionary with keys representing color names,
            each associated with their respective hex color values.
            (Returns an empty dictionary if colors cannot be extracted.)
        """
        data = []
        print('Loading CSS files...')
        for css_file in tqdm(self.css_files):
            css_content = self.get_css_content(css_file)
            data.extend(re.findall(r"--([a-zA-Z\-]+)-hsl:([^;]+)\%;", css_content))
        if not data:
            return {}
        
        df = pd.DataFrame(data, columns=['name', 'hsl'])
        df['hsl'] = df.hsl.apply(my_literal_eval_hsl)
        df['hex'] = df.hsl.apply(hsl_to_hex)

        try:
            backdrop_color = df[df.name.isin(['lightAccent', 'safeLightAccent'])].iloc[0].hex
            canvas_color = df[df.name.isin(['white', 'fallback-white'])].iloc[0].hex
            font_color = df[df.name.isin(['black', 'fallback-black'])].iloc[0].hex
            link_color = df[df.name.isin(['darkAccent', 'safeDarkAccent'])].iloc[0].hex
            accent_color = df[df.name=='accent'].iloc[0].hex
        except:
            return {}
        
        return {'backdrop_color': backdrop_color,
                'canvas_color': canvas_color,
                'font_color': font_color,
                'link_color': link_color,
                'accent_color': accent_color}


    def get_html_colors(self) -> List[str]:
        """
        Extracts all hexadecimal color codes and RGB/RGBA color values from the HTML content.

        Returns
        -------
        A list of hexadecimal color codes found within the HTML content.
        """
        prettified_soup = self.soup.prettify()
        html_colors = re.findall(r'(#[0-9a-fA-F]{6})', prettified_soup)
        html_colors += ['#' + c
                        for c in re.findall(r'\"([0-9a-fA-F]{6})\"', prettified_soup)]
        html_colors += [rgb_to_hex(ast.literal_eval(c))
                        for c in re.findall(r'rgba?(\(\d+,\d+,\d+(?:,1)?\))', prettified_soup)]
        # print('Found', len(html_colors), 'HTML colors!')
        return html_colors


    def get_css_colors(self) -> List[str]:
        """
        Extracts all hexadecimal color codes and HSL color values from the CSS content.

        Returns
        -------
        A list of hexadecimal color codes found within the CSS content.
        """
        css_colors = []
        # Extract colors from each CSS file
        for css_file in tqdm(self.css_files):
            css_content = self.get_css_content(css_file)
            css_colors += re.findall(r'#[0-9a-fA-F]{6}', css_content)
            css_colors += [hsl_to_hex(my_literal_eval_hsl(c))
                           for c in re.findall(r'hsl:([^;]+)\%;', css_content)]
        print('Found', len(css_colors), 'CSS colors!')
        return css_colors


    def get_all_colors(self) -> List[str]:
        """
        Extracts all hexadecimal color codes from the website.

        Returns
        -------
        A list of hexadecimal color codes found within the CSS content.
        """
        all_colors = self.get_html_colors() + self.get_css_colors()
        return all_colors


    def get_color_counts(self, colors: List[str], is_grouped: bool = False) -> List[Tuple[str, int]]:
        """
        Get list of hexadecimal color codes and respective count.
        
        Parameters
        ----------
        colors: List of hexadecimal color codes
        is_grouped: If True, group similar colors together, by default False
        
        Returns
        -------
        List of tuples, where the first element is the color hex string and
            the second element is their count of appearances.
        """
        color_counts = {}
        for color in colors:
            if color.upper() in color_counts:
                color_counts[color.upper()] += 1
            else:
                color_counts[color.upper()] = 1
        color_counts = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)

        if is_grouped:                    
            # Detect similar colors
            def __is_similar(c1: str, c2: str, threshold=300) -> bool:
                """
                Similarity assessment based on Euclidean distance
                between two colors in RGB vector space.
                """
                a1 = np.array(hex_to_rgb(c1.lstrip('#')))
                a2 = np.array(hex_to_rgb(c2.lstrip('#')))
                distance = np.sum((a1 - a2)**2, axis=0)
                return distance < threshold

            similarity_mask = [False] * len(color_counts)
            for i in range(len(color_counts) - 1):
                if similarity_mask[i]:
                    continue
                for j in range(i + 1, len(color_counts)):
                    if __is_similar(color_counts[i][0], color_counts[j][0]):
                        color_counts[i] = (color_counts[i][0], color_counts[i][1] + color_counts[j][1])
                        similarity_mask[j] = True
            color_counts = [color_counts[i] for i in range(len(color_counts)) if not similarity_mask[i]]
            color_counts = sorted(color_counts, key=lambda x: x[1], reverse=True)

        return color_counts
    

    def predict_profile(self, color_counts, min_colors=5, max_colors=20) -> Dict[str, str]:        
        """
        Predict a color profile given a list of hexadecimal color codes and their counts,
        using a Multi-layer Perceptron model.
        
        Parameters
        ----------
        color_counts: List of tuples, where the first element is the color hex string and
            the second element is their count of appearances
        min_colors: The minimum number of colors to consider for prediction, by default 5
        max_colors: The maximum number of colors to consider for prediction, by default 20
        
        Returns
        -------
        A dictionary containing the predicted color profile, with keys representing color names.
            If less than `min_colors` colors are given, an empty dictionary is returned.
        """
        if self.model is None:
            self.model = ColorModel()
            self.model.load_state_dict(torch.load(self.MODEL_DIR, weights_only=True), strict=False)
        
        hex_colors, counts = tuple(list(t) for t in zip(*color_counts[:max_colors]))

        if len(hex_colors) >= min_colors:
            hsl_colors = np.array(list(map(hex_to_array, hex_colors)))
            counts_np = np.array(counts)
            counts_np = counts_np / counts_np.max()
            input = torch.tensor(np.concatenate((hsl_colors, counts_np.reshape(-1, 1)), axis=1)).float()

            output = self.model(input)
            predictions = output.argmax(dim=0).tolist()
            
            return {'backdrop_color': hex_colors[predictions[0]],
                    'canvas_color': hex_colors[predictions[1]],
                    'font_color': hex_colors[predictions[2]],
                    'link_color': hex_colors[predictions[3]],
                    'accent_color': hex_colors[predictions[4]]}
        
        return {}


    def get_color_profile(self) -> Dict[str, str]:
        """
        Get a color profile for the given URL.
        
        First, it checks if the website is built on web builders (e.g. ShowIt or Squarespace),
        and if so, it uses their predefined methods to extract the color profile.
        If not, it extracts all colors from the website, counts their appearances,
        and then uses an AI model to predict the color profile.
        
        Returns
        -------
        A dictionary containing the color profile, with keys representing color names.
            If no color profile can be extracted or predicted, an empty dictionary is returned.
        """
        extracted_colors = (self.get_colors_showit() or
                            self.get_colors_squarespace())
        if not extracted_colors:
            # Predict color profile with model
            all_colors = self.get_all_colors()
            color_counts = self.get_color_counts(all_colors)
            extracted_colors = self.predict_profile(color_counts)
        return extracted_colors
